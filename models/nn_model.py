import torch
import torch.nn as nn
import matplotlib.pyplot as plt


class CrossAttentionMultiHead(nn.Module):
    def __init__(self, text_emb_dim, channels_current_layer, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (channels_current_layer // num_heads) ** -0.5  # Масштабируем по размеру одной головы
        # Приведение текстового эмбеддинга к C
        self.Wq = nn.Linear(channels_current_layer, channels_current_layer)  # Query из изображения
        self.Wk = nn.Linear(text_emb_dim, channels_current_layer)  # Key из текста
        self.Wv = nn.Linear(text_emb_dim, channels_current_layer)  # Value из текста
        self.text_emb_dim = text_emb_dim
        self.Channels_current_layer = channels_current_layer

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
        return attn_out


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


class DeepBottleneck(nn.Module):
    def __init__(self, text_emb_dim, time_emb_dim):
        super().__init__()
        self.deep_block_1 = UNetBottleneckBlock(256, 512,
                                                time_emb_dim)  # этот блок считает свёртку, но не уменьшает изображение (conv 3*3)
        self.deep_block_2 = UNetBottleneckBlock(512, 512, time_emb_dim)
        self.cross_attn_multi_head_1 = CrossAttentionMultiHead(text_emb_dim,
                                                               512)  # Cross Attention на уровне bottleneck
        self.cross_attn_multi_head_2 = CrossAttentionMultiHead(text_emb_dim, 512)
        self.residual = nn.Conv2d(256, 512,
                                  kernel_size=1)  # Residual Block (для совпадения количества каналов, применяем слой свёртки) (задан правильно)
        self.group_norm_1 = nn.GroupNorm(num_groups=32, num_channels=512, affine=True)
        self.silu_1 = nn.SiLU()
        self.group_norm_2 = nn.GroupNorm(num_groups=32, num_channels=512, affine=True)
        self.silu_2 = nn.SiLU()

    # здесь правильный порядок преобразований!!!
    def forward(self, x, text_emb, time_emb, attention_mask):
        res = x  # Для Residual Connection
        x = self.deep_block_1(x, time_emb)
        x = self.cross_attn_multi_head_1(x, text_emb, attention_mask)
        x = self.group_norm_1(x)
        x = self.silu_1(x)

        x = self.deep_block_2(x, time_emb)
        x = self.cross_attn_multi_head_2(x, text_emb, attention_mask)
        x = self.group_norm_2(x)
        x = self.silu_2(x)
        r = self.residual(res)
        x = x + r
        return x


class MyUNet(nn.Module):
    def __init__(self, txt_emb_dim, time_emb_dim, device):
        super().__init__()

        self.device = device

        # self.time_embedding = SinusoidalTimeEmbedding(
        #     256, device)  # (это число (256) должно соотвествовать 3-ему инициализированному числу в UNetBlock)
        # это число означает размерность эмбеддинга одного числа t

        # --- Downsampling (Сжатие) ---
        self.unet_enc_conv_1 = UNetEncBlock(3, 64,
                                            time_emb_dim)  # каждый такой блок уменьшает изображение на 2 пикселя (conv 3*3)
        self.unet_enc_conv_2 = UNetEncBlock(64, 64, time_emb_dim)
        self.pool_1 = nn.MaxPool2d(2, 2)  # уменьшаем изображение в 2 раза (max pool 2*2)
        self.unet_enc_conv_3 = UNetEncBlock(64, 128, time_emb_dim)
        self.unet_enc_conv_4 = UNetEncBlock(128, 128, time_emb_dim)
        self.pool_2 = nn.MaxPool2d(2, 2)
        self.unet_enc_conv_5 = UNetEncBlock(128, 256, time_emb_dim)
        self.unet_enc_conv_6 = UNetEncBlock(256, 256, time_emb_dim)
        self.pool_3 = nn.MaxPool2d(2, 2)

        # --- Bottleneck (Самая узкая часть) ---
        self.deep_bottleneck = DeepBottleneck(txt_emb_dim, time_emb_dim)

        # --- Upsampling (Расширение) ---
        self.unet_dec_up_conv_1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2, padding=0,
                                                     output_padding=0)  # правильно (up conv 2*2) Этот блок увеличивает изображение в 2 раза
        self.unet_dec_deconv_1 = UNetDecBlock(256 + 256, 256,
                                              time_emb_dim)  # каждый такой блок увеличивает изображение на 2 пикселя (deconv 3*3)
        self.unet_dec_deconv_2 = UNetDecBlock(256, 256, time_emb_dim)
        self.cross_attn_1 = CrossAttentionMultiHead(txt_emb_dim, 256)
        self.group_norm_1 = nn.GroupNorm(num_groups=32, num_channels=256, affine=True)
        self.silu_1 = nn.SiLU()

        self.unet_dec_up_conv_2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, padding=0,
                                                     output_padding=0)  # правильно
        self.unet_dec_deconv_3 = UNetDecBlock(128 + 128, 128, time_emb_dim)
        self.unet_dec_deconv_4 = UNetDecBlock(128, 128, time_emb_dim)
        self.cross_attn_2 = CrossAttentionMultiHead(txt_emb_dim, 128)
        self.group_norm_2 = nn.GroupNorm(num_groups=32, num_channels=128, affine=True)
        self.silu_2 = nn.SiLU()

        self.unet_dec_up_conv_3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, padding=0,
                                                     output_padding=0)  # правильно
        self.unet_dec_deconv_5 = UNetDecBlock(64 + 64, 64, time_emb_dim)
        self.unet_dec_deconv_6 = UNetDecBlock(64, 64, time_emb_dim)
        self.cross_attn_3 = CrossAttentionMultiHead(txt_emb_dim, 64)
        self.group_norm_3 = nn.GroupNorm(num_groups=32, num_channels=64, affine=True)
        self.silu_3 = nn.SiLU()

        # self.final = nn.Conv2d(64, 3, kernel_size=1, padding=0,
        #                        stride=1)  # Финальный слой (правильно)

        self.final = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=1, padding=0, stride=1),
            nn.Tanh()
        )

    def forward(self, x, text_emb, time_emb, attension_mask):
        # --- Encoder (Downsampling) ---
        # print(time_t.device)

        x = self.unet_enc_conv_1(x, time_emb)
        x = self.unet_enc_conv_2(x, time_emb)
        x_skip_conn_1 = x
        x = self.pool_1(x)

        x = self.unet_enc_conv_3(x, time_emb)
        x = self.unet_enc_conv_4(x, time_emb)
        x_skip_conn_2 = x
        x = self.pool_2(x)

        x = self.unet_enc_conv_5(x, time_emb)
        x = self.unet_enc_conv_6(x, time_emb)
        x_skip_conn_3 = x
        x = self.pool_3(x)

        x = self.deep_bottleneck(x, text_emb, time_emb, attension_mask)

        # это правильный порядок блока!!!
        x = self.unet_dec_up_conv_1(x)
        x_cat_1 = torch.cat([x, x_skip_conn_3], dim=1)
        x = self.unet_dec_deconv_1(x_cat_1, time_emb)
        x = self.unet_dec_deconv_2(x, time_emb)
        x = self.cross_attn_1(x, text_emb, attension_mask)
        x = self.group_norm_1(x)
        x = self.silu_1(x)

        x = self.unet_dec_up_conv_2(x)
        x_cat_2 = torch.cat([x, x_skip_conn_2], dim=1)
        x = self.unet_dec_deconv_3(x_cat_2, time_emb)
        x = self.unet_dec_deconv_4(x, time_emb)
        x = self.cross_attn_2(x, text_emb, attension_mask)
        x = self.group_norm_2(x)
        x = self.silu_2(x)

        x = self.unet_dec_up_conv_3(x)
        x_cat_3 = torch.cat([x, x_skip_conn_1], dim=1)
        x = self.unet_dec_deconv_5(x_cat_3, time_emb)
        x = self.unet_dec_deconv_6(x, time_emb)
        x = self.cross_attn_3(x, text_emb, attension_mask)
        x = self.group_norm_3(x)
        x = self.silu_3(x)

        x = self.final(x)

        return x


def show_image(tensor_img):
    """ Визуализация тензора изображения """
    # img = tensor_img.cpu().detach().numpy()
    img = tensor_img.cpu().detach().numpy().transpose(1, 2, 0)  # Приводим к (H, W, C)
    img = (img - img.min()) / (
            img.max() - img.min())  # Нормализация к [0,1] (matplotlib ждёт данные в формате [0, 1], другие не примет)
    plt.imshow(img)
    plt.axis("off")  # Убираем оси
    plt.show()
    plt.pause(3600)
