import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import math


# class SinusoidalEmbeddings(nn.Module):
#     def __init__(self, time_steps: int, embed_dim: int):
#         super().__init__()
#         position = torch.arange(time_steps).unsqueeze(1).float()
#         div = torch.exp(torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim))
#         embeddings = torch.zeros(time_steps, embed_dim, requires_grad=False)
#         embeddings[:, 0::2] = torch.sin(position * div)
#         embeddings[:, 1::2] = torch.cos(position * div)
#         self.embeddings = embeddings
#
#     def forward(self, x, t):
#         embeds = self.embeddings[t].to(x.device)
#         return embeds[:, :, None, None]


class CrossAttentionMultiHead(nn.Module):
    def __init__(self, text_emb_dim, channels_current_layer, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (channels_current_layer // num_heads) ** -0.5  # Масштабируем по размеру одной головы
        # Приведение текстового эмбеддинга к C
        self.Wq = nn.Linear(channels_current_layer, channels_current_layer)  # Query из изображения
        self.Wk = nn.Linear(text_emb_dim, channels_current_layer)  # Key из текста
        self.Wv = nn.Linear(text_emb_dim, channels_current_layer)  # Value из текста
        self.group_norm = nn.GroupNorm(num_groups=8, num_channels=channels_current_layer, affine=True)
        # self.text_emb_dim = text_emb_dim
        # self.Channels_current_layer = channels_current_layer

    def forward(self, x, text_emb, attention_mask):
        B, C, H, W = x.shape
        x_flat = x.view(B, C, H * W).permute(0, 2, 1)  # (B, H*W, C)
        # Вычисляем Q, K, V
        Q = self.Wq(x_flat)  # (B, H*W, C)
        K = self.Wk(text_emb)  # (B, T, C)
        V = self.Wv(text_emb)  # (B, T, C)
        # Разделение на головы и масштабирование
        Q = Q.view(B, -1, self.num_heads, C // self.num_heads).transpose(1, 2)  # (B, num_heads, H*W, C//num_heads)
        K = K.view(B, -1, self.num_heads, C // self.num_heads).transpose(1, 2)  # (B, num_heads, T, C//num_heads)
        V = V.view(B, -1, self.num_heads, C // self.num_heads).transpose(1, 2)  # (B, num_heads, T, C//num_heads)
        # Attention
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # (B, num_heads, H*W, T)

        attention_mask = attention_mask[:, None, None, :].expand_as(attn_scores)  # (B, 1, 1, T)
        attn_scores = attn_scores.masked_fill(attention_mask == 0, float('-inf'))

        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_out = torch.matmul(attn_probs, V)  # (B, num_heads, H*W, C//num_heads)
        # Объединяем головы
        attn_out = attn_out.transpose(1, 2).reshape(B, H * W, C)  # (B, H*W, C)
        attn_out = attn_out.permute(0, 2, 1).view(B, C, H,
                                                  W)  # (B, C, H, W) (H и W не меняются, поэтому делаем преобразование без доп. проверок)
        attn_out = self.group_norm(attn_out)
        return attn_out


class SelfAttentionBlock(nn.Module):
    """
    Perform GroupNorm and Multiheaded Self Attention operation.
    """

    def __init__(self, num_channels: int, num_groups: int = 8, num_heads: int = 4):
        super(SelfAttentionBlock, self).__init__()
        # GroupNorm
        self.g_norm = nn.GroupNorm(num_groups, num_channels)
        # Self-Attention
        self.attn = nn.MultiheadAttention(
            num_channels,
            num_heads,
            batch_first=True
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.reshape(B, C, H * W)
        x = self.g_norm(x)
        x = x.transpose(1, 2)
        x, _ = self.attn(x, x, x)
        x = x.transpose(1, 2).reshape(B, C, H, W)
        return x


class UNetEncBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=0,
                              stride=1)  # Уменьшение изобр. на 2 пикселя (правильно)
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)  # Преобразуем `t` к текущем получившемуся кол-ву каналов
        self.group_norm = nn.GroupNorm(num_groups=32, num_channels=out_channels, affine=True)
        self.silu = nn.SiLU()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_emb_dim = time_emb_dim

    # x=(B, inC, H, W)->x=(B, outC, H-2, W-2);
    # time_emb=(B, tim_emb)->time_emb=(B, outC, 1, 1) (tim_emb должен быть равен time_emb_dim)
    # это правильный порядок применения преобразований!!!
    def forward(self, x, time_emb):
        x = self.conv(x)
        t_emb = self.time_mlp(time_emb)[:, :, None, None]  # (B, outC) -> (B, outC, 1, 1)
        x = x + t_emb  # Добавляем `t` к фичам
        x = self.group_norm(x)
        x = self.silu(x)
        return x


class UNetBottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1,
                              stride=1)  # Не уменьшаем размер изобр. (правильно)
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)  # Преобразуем `t`
        self.group_norm = nn.GroupNorm(num_groups=32, num_channels=out_channels, affine=True)
        self.silu = nn.SiLU()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_emb_dim = time_emb_dim

    # x=(B, inC, H, W)->x=(B, outC, H-2, W-2);
    # time_emb=(B, tim_emb)->time_emb=(B, outC, 1, 1) (tim_emb должен быть равен time_emb_dim)
    # это правильный порядок применения преобразований!!!
    def forward(self, x, time_emb):
        x = self.conv(x)
        t_emb = self.time_mlp(time_emb)[:, :, None, None]  # (B, outC) -> (B, outC, 1, 1)
        x = x + t_emb  # Добавляем `t` к фичам
        x = self.group_norm(x)
        x = self.silu(x)
        return x


class UNetDecBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=1, padding=0,
                                         output_padding=0)  # правильно
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)  # Преобразуем `t`
        self.group_norm = nn.GroupNorm(num_groups=32, num_channels=out_channels, affine=True)
        self.silu = nn.SiLU()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_emb_dim = time_emb_dim

    # x=(B, inC, H, W)->x=(B, outC, H, W); time_emb=(B, te)->time_emb=(B, outC, 1, 1) (te должен быть равен time_emb_dim)
    # это правильный порядок!!!
    def forward(self, x, time_emb):
        x = self.deconv(x)
        t_emb = self.time_mlp(time_emb)[:, :, None, None]  # (B, outC) -> (B, outC, 1, 1)
        x = x + t_emb
        x = self.group_norm(x)
        x = self.silu(x)
        return x


class TimeEmbedding(nn.Module):
    """
    Maps the Time Embedding to the Required output Dimension.
    """

    def __init__(self,
                 n_out: int,  # Output Dimension
                 t_emb_dim: int  # Time Embedding Dimension
                 ):
        super(TimeEmbedding, self).__init__()
        # Time Embedding Block
        self.te_block = nn.Sequential(
            nn.SiLU(),
            nn.Linear(t_emb_dim, n_out)
        )

    def forward(self, x):
        return self.te_block(x)


# class ResNetBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, time_emb_dim, mode):
#         super().__init__()
#         self.mode = mode
#         if mode == 'down':
#             self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=0,
#                                     stride=1)  # Уменьшение изобр. на 2 пикселя (правильно)
#             self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=0,
#                                     stride=1)  # Уменьшение изобр. на 2 пикселя (правильно)
#             self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=0, stride=1)
#         elif mode == 'bottleneck':
#             self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1,
#                                     stride=1)  # Не уменьша изобр. (правильно)
#             self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1,
#                                     stride=1)  # Не уменьша изобр. (правильно)
#             self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1)
#         elif mode == 'up':
#             self.conv_1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=1, padding=0,
#                                              output_padding=0)  # Увелич. на 2 пикселя (правильно)
#             self.conv_2 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=1, padding=0,
#                                              output_padding=0)  # Увелич. на 2 пикселя (правильно)
#             self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=2, stride=1)
#
#         if in_channels % 8 != 0:
#             self.group_norm_1 = nn.GroupNorm(num_groups=in_channels, num_channels=in_channels, affine=True)
#         else:
#             self.group_norm_1 = nn.GroupNorm(num_groups=8, num_channels=in_channels, affine=True)
#         self.group_norm_2 = nn.GroupNorm(num_groups=8, num_channels=out_channels, affine=True)
#         self.group_norm_3 = nn.GroupNorm(num_groups=8, num_channels=out_channels, affine=True)
#
#         self.te_block = TimeEmbedding(out_channels, time_emb_dim)
#
#         self.self_attension = SelfAttentionBlock(out_channels)
#
#         self.silu_1 = nn.SiLU()
#         self.silu_2 = nn.SiLU()
#
#
#
#     def forward(self, x, time_emb):
#         residual = x
#
#         x = self.group_norm_1(x)
#         x = self.silu_1(x)
#         x = self.conv_1(x)
#
#         x = x + self.te_block(time_emb)[:, :, None, None]
#
#         x = self.group_norm_2(x)
#         x = self.silu_2(x)
#         x = self.conv_2(x)
#
#         x = self.group_norm_3(x)
#         x = self.self_attension(x)
#
#         x = x + self.residual(residual)
#         return x


class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.te_block = TimeEmbedding(in_channels, time_emb_dim)
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1,
                                stride=1)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1,
                                stride=1)
        if in_channels % 8 != 0:
            self.group_norm_1 = nn.GroupNorm(num_groups=in_channels, num_channels=in_channels, affine=True)
        else:
            self.group_norm_1 = nn.GroupNorm(num_groups=8, num_channels=in_channels, affine=True)
        self.group_norm_2 = nn.GroupNorm(num_groups=8, num_channels=out_channels, affine=True)

        self.silu_1 = nn.SiLU()
        self.silu_2 = nn.SiLU()

    def forward(self, x, time_emb):
        x = x + self.te_block(time_emb)[:, :, None, None]
        r = self.group_norm_1(x)
        r = self.silu_1(r)
        r = self.conv_1(r)
        r = self.group_norm_2(r)
        r = self.silu_2(r)
        r = self.conv_2(r)
        r = r + self.residual(x)
        return r


class ResNetMiddleBlock(ResNetBlock):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__(in_channels, out_channels, time_emb_dim)
        self.self_attention = SelfAttentionBlock(out_channels)

    def forward(self, x, time_emb):
        x = x + self.te_block(time_emb)[:, :, None, None]
        r = self.group_norm_1(x)
        r = self.silu_1(r)
        r = self.conv_1(r)

        r = self.self_attention(r)

        # x = x + self.te_block(time_emb)[:, :, None, None]
        r = self.group_norm_2(r)
        r = self.silu_2(r)
        r = self.conv_2(r)

        # x = self.group_norm_3(x)
        # x = self.self_attension(x)

        r = r + self.residual(x)
        return r


class DeepBottleneck(nn.Module):
    def __init__(self, text_emb_dim, time_emb_dim):
        super().__init__()

        # self.residual = nn.Conv2d(256, 512,
        #                           kernel_size=1)  # Residual Block (для совпадения количества каналов, применяем слой свёртки) (задан правильно)
        self.deep_block_1 = ResNetBlock(256, 512, time_emb_dim)
        # self.self_attension = SelfAttentionBlock(512)
        self.cross_attn_multi_head_1 = CrossAttentionMultiHead(text_emb_dim, 512)
        # self.group_norm = nn.GroupNorm(num_groups=8, num_channels=512, affine=True)
        self.deep_block_2 = ResNetBlock(512, 512, time_emb_dim)

        # self.deep_block_1 = UNetBottleneckBlock(256, 512,
        #                                         time_emb_dim)  # этот блок считает свёртку, но не уменьшает изображение (conv 3*3)
        # self.deep_block_2 = UNetBottleneckBlock(512, 512, time_emb_dim)
        # self.cross_attn_multi_head_1 = CrossAttentionMultiHead(text_emb_dim,
        #                                                        512)  # Cross Attention на уровне bottleneck
        # self.cross_attn_multi_head_1 = CrossAttentionMultiHead(text_emb_dim, 512)

        # self.group_norm_1 = nn.GroupNorm(num_groups=8, num_channels=512, affine=True)
        # self.silu_1 = nn.SiLU()
        # self.group_norm_2 = nn.GroupNorm(num_groups=8, num_channels=512, affine=True)
        # self.silu_2 = nn.SiLU()

    # здесь правильный порядок преобразований!!!
    def forward(self, x, text_emb, time_emb, attention_mask):
        # res = x  # Для Residual Connection
        x = self.deep_block_1(x, time_emb)
        # x = self.self_attension(x)
        x = self.cross_attn_multi_head_1(x, text_emb, attention_mask)
        # x = self.group_norm(x)
        x = self.deep_block_2(x, time_emb)
        # r = self.residual(res)
        # x = x + r
        return x


class MyUNet(nn.Module):
    def __init__(self, txt_emb_dim, time_emb_dim, device):
        super().__init__()

        self.device = device

        # --- Downsampling (Сжатие) ---
        self.single_conv = nn.Conv2d(3, 64, kernel_size=3, padding=1,
                                     stride=1)
        self.encoder_1 = ResNetBlock(64, 64, time_emb_dim)
        self.down_conv_1 = nn.Conv2d(64, 64, kernel_size=3, padding=1,
                                     stride=2)  # уменьшаем изображение в 2 раза (для диффузионок)

        self.encoder_2 = ResNetMiddleBlock(64, 128, time_emb_dim)
        self.down_conv_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1,
                                     stride=2)  # уменьшаем изображение в 2 раза (для диффузионок)

        self.encoder_3 = ResNetBlock(128, 256, time_emb_dim)
        self.down_conv_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1,
                                     stride=2)  # уменьшаем изображение в 2 раза (для диффузионок)

        self.deep_bottleneck = DeepBottleneck(txt_emb_dim, time_emb_dim)

        self.up_1 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2, padding=0,
                                       output_padding=0)  # правильно (up conv 2*2) Этот блок увеличивает изображение в 2 раза
        self.decoder_1 = ResNetBlock(256 + 512, 512, time_emb_dim)

        self.up_2 = nn.ConvTranspose2d(512, 384, kernel_size=2, stride=2, padding=0,
                                       output_padding=0)  # правильно
        self.decoder_2 = ResNetMiddleBlock(128 + 384, 384, time_emb_dim)

        self.cross_attn_1 = CrossAttentionMultiHead(txt_emb_dim, 384)

        self.up_3 = nn.ConvTranspose2d(384, 192, kernel_size=2, stride=2, padding=0,
                                       output_padding=0)  # правильно
        self.single_conv_dec_1 = nn.Conv2d(64 + 192, 96, kernel_size=3, padding=1,
                                           stride=1)
        self.final = nn.Sequential(
            nn.Conv2d(96, 3, kernel_size=3, padding=1, stride=1),
            # nn.Tanh()
        )

        # self.final = nn.Sequential(
        #     nn.Conv2d(96, 3, kernel_size=1),
        #     nn.Tanh()
        # )

    def forward(self, x, text_emb, time_emb, attension_mask):
        x = self.single_conv(x)
        x = self.encoder_1(x, time_emb)
        x_skip_conn_1 = x
        x = self.down_conv_1(x)

        x = self.encoder_2(x, time_emb)
        x_skip_conn_2 = x
        x = self.down_conv_2(x)

        x = self.encoder_3(x, time_emb)
        x_skip_conn_3 = x
        x = self.down_conv_3(x)

        x = self.deep_bottleneck(x, text_emb, time_emb, attension_mask)

        x = self.up_1(x)
        x = torch.cat([x, x_skip_conn_3], dim=1)
        x = self.decoder_1(x, time_emb)

        x = self.up_2(x)
        x = torch.cat([x, x_skip_conn_2], dim=1)
        x = self.decoder_2(x, time_emb)
        x = self.cross_attn_1(x, text_emb, attension_mask)

        x = self.up_3(x)
        x = torch.cat([x, x_skip_conn_1], dim=1)
        x = self.single_conv_dec_1(x)
        x = self.final(x)
        return x
