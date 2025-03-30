import torch
import torch.nn as nn


class CrossAttentionMultiHead(nn.Module):
    def __init__(self, text_emb_dim, channels_current_layer, batch_size):
        super().__init__()
        self.bs = batch_size
        num_heads = 4
        self.num_heads = num_heads
        self.scale = (channels_current_layer // num_heads) ** -0.5  # Масштабируем по размеру одной головы
        # Приведение текстового эмбеддинга к C
        self.Wq = nn.Linear(channels_current_layer, channels_current_layer)  # Query из изображения
        self.Wk = nn.Linear(text_emb_dim, channels_current_layer)  # Key из текста
        self.Wv = nn.Linear(text_emb_dim, channels_current_layer)  # Value из текста

        # Этот и подобный код инициализирует разные нормировки в зависимости от размера батча
        if self.bs < 16:
            self.norm = nn.GroupNorm(num_groups=8, num_channels=channels_current_layer, affine=True)
        else:
            self.norm = nn.BatchNorm2d(num_features=channels_current_layer)

        self.attn_dropout = nn.Dropout(p=0.1)
        self.output_dropout = nn.Dropout(p=0.1)

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
        attn_probs = self.attn_dropout(attn_probs)
        attn_out = torch.matmul(attn_probs, V)  # (B, num_heads, H*W, C//num_heads)
        attn_out = self.output_dropout(attn_out)
        # Объединяем головы
        attn_out = attn_out.transpose(1, 2).reshape(B, H * W, C)  # (B, H*W, C)
        attn_out = attn_out.permute(0, 2, 1).view(B, C, H,
                                                  W)  # (B, C, H, W) (H и W не меняются, поэтому делаем преобразование без доп. проверок)
        attn_out = self.norm(attn_out)
        return attn_out


class SelfAttentionBlock(nn.Module):
    """
    Perform GroupNorm and Multiheaded Self Attention operation.
    """

    def __init__(self, num_channels: int, num_groups: int = 8, num_heads: int = 4):
        super(SelfAttentionBlock, self).__init__()
        # GroupNorm
        self.norm = nn.GroupNorm(num_groups, num_channels)
        # Self-Attention
        self.attn = nn.MultiheadAttention(
            num_channels,
            num_heads,
            batch_first=True,
            dropout=0.1
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.reshape(B, C, H * W)
        x = self.norm(x)
        x = x.transpose(1, 2)
        x, _ = self.attn(x, x, x)
        x = x.transpose(1, 2).reshape(B, C, H, W)
        return x


class TimeEmbedding(nn.Module):
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


class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, batch_size):
        super().__init__()
        self.bs = batch_size
        self.te_block = TimeEmbedding(in_channels, time_emb_dim)
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1,
                                stride=1)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1,
                                stride=1)
        if self.bs < 16:
            if in_channels % 8 != 0:
                self.norm_1 = nn.GroupNorm(num_groups=in_channels, num_channels=in_channels, affine=True)
            else:
                self.norm_1 = nn.GroupNorm(num_groups=8, num_channels=in_channels, affine=True)
            self.norm_2 = nn.GroupNorm(num_groups=8, num_channels=out_channels, affine=True)
        else:
            self.norm_1 = nn.BatchNorm2d(num_features=in_channels)
            self.norm_2 = nn.BatchNorm2d(num_features=out_channels)

        self.silu_1 = nn.SiLU()
        self.silu_2 = nn.SiLU()

        self.dropout = nn.Dropout(p=0.1)  # 10% нейронов будут обнулены

    def forward(self, x, time_emb):
        x = x + self.te_block(time_emb)[:, :, None, None]
        r = self.norm_1(x)
        r = self.silu_1(r)
        r = self.conv_1(r)

        r = self.dropout(r)

        r = self.norm_2(r)
        r = self.silu_2(r)
        r = self.conv_2(r)
        r = r + self.residual(x)
        return r


class ResNetMiddleBlock(ResNetBlock):
    def __init__(self, in_channels, out_channels, time_emb_dim, batch_size):
        super().__init__(in_channels, out_channels, time_emb_dim, batch_size)
        self.self_attention = SelfAttentionBlock(out_channels)

    def forward(self, x, time_emb):
        x = x + self.te_block(time_emb)[:, :, None, None]
        r = self.norm_1(x)
        r = self.silu_1(r)
        r = self.conv_1(r)

        r = self.self_attention(r)
        r = self.dropout(r)

        r = self.norm_2(r)
        r = self.silu_2(r)
        r = self.conv_2(r)

        r = r + self.residual(x)
        return r


class DeepBottleneck(nn.Module):
    def __init__(self, in_C, out_C, text_emb_dim, time_emb_dim, batch_size):
        super().__init__()
        self.batch_size = batch_size
        self.deep_block_1 = ResNetBlock(in_C, out_C, time_emb_dim, self.batch_size)
        self.cross_attn_multi_head_1 = CrossAttentionMultiHead(text_emb_dim, out_C, self.batch_size)
        # self.deep_block_2 = ResNetBlock(512, 512, time_emb_dim)

    # здесь правильный порядок преобразований!!!
    def forward(self, x, text_emb, time_emb, attention_mask):
        x = self.deep_block_1(x, time_emb)
        x = self.cross_attn_multi_head_1(x, text_emb, attention_mask)
        # x = self.deep_block_2(x, time_emb)
        return x


class MyUNet(nn.Module):
    def __init__(self, txt_emb_dim,
                 time_emb_dim,
                 orig_img_channels,
                 channels_div,
                 batch_size):
        super().__init__()

        self.batch_size = batch_size
        self.orig_img_channels = orig_img_channels
        self.channels_div = channels_div  # во сколько раз делим кол-во каналов от оригинального кол-ва
        # self.orig_img_channels = 3

        # --- Downsampling (Сжатие) ---
        # self.single_conv = nn.Conv2d(self.orig_img_channels, 64, kernel_size=3, padding=1,
        #                              stride=1)

        self.prepare = nn.Conv2d(self.orig_img_channels, 64 // self.channels_div, kernel_size=1, padding=0,
                                 stride=1)  # Не меняем изображение, просто подготавливаем его для дальнейшей обработки, увеличивая кол-во каналов (линейное преобразование)
        self.encoder_1 = ResNetBlock(64 // self.channels_div,
                                     64 // self.channels_div,
                                     time_emb_dim,
                                     self.batch_size)
        self.down_conv_1 = nn.Conv2d(64 // self.channels_div, 64 // self.channels_div, kernel_size=3, padding=1,
                                     stride=2)  # уменьшаем изображение в 2 раза (для диффузионок)

        self.encoder_2 = ResNetMiddleBlock(64 // self.channels_div,
                                           128 // self.channels_div,
                                           time_emb_dim,
                                           self.batch_size)
        self.down_conv_2 = nn.Conv2d(128 // self.channels_div, 128 // self.channels_div, kernel_size=3, padding=1,
                                     stride=2)  # уменьшаем изображение в 2 раза (для диффузионок)

        self.encoder_3 = ResNetBlock(128 // self.channels_div,
                                     256 // self.channels_div,
                                     time_emb_dim,
                                     self.batch_size)
        self.down_conv_3 = nn.Conv2d(256 // self.channels_div, 256 // self.channels_div, kernel_size=3, padding=1,
                                     stride=2)  # уменьшаем изображение в 2 раза (для диффузионок)

        self.deep_bottleneck = DeepBottleneck(256 // self.channels_div,
                                              512 // self.channels_div,
                                              txt_emb_dim,
                                              time_emb_dim,
                                              self.batch_size)

        self.up_1 = nn.Sequential(
            nn.ConvTranspose2d(512 // self.channels_div, 512 // self.channels_div, kernel_size=3, stride=2, padding=1,
                               output_padding=1),
            # Более мягкая версия увеличения в 2 раза, помогающая убрать шахматные артефакты
            nn.Conv2d(512 // self.channels_div, 512 // self.channels_div, kernel_size=3, padding=1, stride=1),
            # Сглаживающая свёртка
            nn.SiLU()  # Максимальная мягкость
        )

        # self.up_1 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2, padding=0,
        #                                output_padding=0)  # правильно (up conv 2*2) Этот блок увеличивает изображение в 2 раза
        self.decoder_1 = ResNetBlock(256 // self.channels_div + 512 // self.channels_div,
                                     512 // self.channels_div,
                                     time_emb_dim,
                                     self.batch_size)

        self.cross_attn_1 = CrossAttentionMultiHead(txt_emb_dim,
                                                    512 // self.channels_div,
                                                    self.batch_size)

        self.up_2 = nn.Sequential(
            nn.ConvTranspose2d(512 // self.channels_div, 384 // self.channels_div, kernel_size=3, stride=2, padding=1,
                               output_padding=1),
            # Более мягкая версия увеличения в 2 раза, помогающая убрать шахматные артефакты
            nn.Conv2d(384 // self.channels_div, 384 // self.channels_div, kernel_size=3, padding=1, stride=1),
            # Сглаживающая свёртка
            nn.SiLU()  # Максимальная мягкость
        )

        # self.up_2 = nn.ConvTranspose2d(512, 384, kernel_size=2, stride=2, padding=0,
        #                                output_padding=0)  # правильно
        self.decoder_2 = ResNetMiddleBlock(128 // self.channels_div + 384 // self.channels_div,
                                           384 // self.channels_div,
                                           time_emb_dim,
                                           self.batch_size)

        self.cross_attn_2 = CrossAttentionMultiHead(txt_emb_dim,
                                                    384 // self.channels_div,
                                                    self.batch_size)

        self.up_3 = nn.Sequential(
            nn.ConvTranspose2d(384 // self.channels_div, 192 // self.channels_div, kernel_size=3, stride=2, padding=1,
                               output_padding=1),
            # Более мягкая версия увеличения в 2 раза, помогающая убрать шахматные артефакты
            nn.Conv2d(192 // self.channels_div, 192 // self.channels_div, kernel_size=3, padding=1, stride=1),
            # Сглаживающая свёртка
            nn.SiLU()  # Максимальная мягкость
        )

        # self.up_3 = nn.ConvTranspose2d(384, 192, kernel_size=2, stride=2, padding=0,
        #                                output_padding=0)  # правильно
        self.single_conv_dec_1 = nn.Conv2d(64 // self.channels_div + 192 // self.channels_div, 96 // self.channels_div,
                                           kernel_size=3, padding=1,
                                           stride=1)
        self.final = nn.Sequential(
            nn.Conv2d(96 // self.channels_div, self.orig_img_channels, kernel_size=3, padding=1, stride=1),
        )  # В ddpm без финальной активации

        # self.final = nn.Sequential(
        #     nn.Conv2d(96, 3, kernel_size=1),
        #     nn.Tanh()
        # )

    def forward(self, x, text_emb, time_emb, attension_mask):
        x = self.prepare(x)
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
        x = self.cross_attn_1(x, text_emb, attension_mask)

        x = self.up_2(x)
        x = torch.cat([x, x_skip_conn_2], dim=1)
        x = self.decoder_2(x, time_emb)
        x = self.cross_attn_2(x, text_emb, attension_mask)

        x = self.up_3(x)
        x = torch.cat([x, x_skip_conn_1], dim=1)
        x = self.single_conv_dec_1(x)
        x = self.final(x)
        return x
