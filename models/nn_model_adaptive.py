import torch
import torch.nn as nn
import models.hyperparams as hyperparams


class CrossAttentionMultiHead(nn.Module):
    def __init__(self, text_dim, img_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (img_dim // num_heads) ** -0.5

        # Проекции для изображения
        self.to_q = nn.Linear(img_dim, img_dim)

        # Проекции для текста
        self.to_kv = nn.Linear(text_dim, img_dim * 2)

        # Нормализация
        self.norm = nn.LayerNorm(img_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, text_emb, mask=None):
        B, C, H, W = x.shape
        x_flat = x.permute(0, 2, 3, 1).view(B, H * W, C)  # [B, H*W, C]

        # Query из изображения
        q = self.to_q(x_flat)

        # Key/Value из текста
        k, v = self.to_kv(text_emb).chunk(2, dim=-1)

        # Multi-head attention
        q = q.view(B, -1, self.num_heads, C // self.num_heads).transpose(1, 2)  # [B, nh, H*W, C//nh]
        k = k.view(B, -1, self.num_heads, C // self.num_heads).transpose(1, 2)  # [B, nh, L, C//nh]
        v = v.view(B, -1, self.num_heads, C // self.num_heads).transpose(1, 2)  # [B, nh, L, C//nh]

        # Attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, nh, H*W, L]

        # Применение маски (если она есть)
        if mask is not None:
            # mask: [B, L] → [B, 1, 1, L] (подходит для broadcating)
            mask = mask[:, None, None, :].to(attn.dtype)
            attn = attn.masked_fill(mask == 0, float('-inf'))  # -inf → softmax → 0

        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, H * W, C)
        out = out.permute(0, 2, 1).view(B, C, H, W)
        return out


class MyAdaptUNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # сигнатура конфига останется прежней, просто некоторые поля в этом адаптивном блоке не будут использоваться
        self.txt_emb_dim = self.config['TEXT_EMB_DIM']
        self.time_emb_dim = self.config['TIME_EMB_DIM']
        self.batch_size = self.config['BATCH_SIZE']
        self.orig_img_channels = self.config['ORIG_C']
        self.apply_fft = self.config['FFT']

        out_C = 2 * self.orig_img_channels
        self.act_1 = nn.Softplus()

        self.conv_1 = nn.Conv2d(self.orig_img_channels, out_C, kernel_size=3, padding=1, stride=1)

        self.CA = CrossAttentionMultiHead(self.txt_emb_dim, out_C, 2)
        self.dropout = nn.Dropout(0.1)

        self.act_2 = nn.GELU()
        self.conv_2 = nn.Conv2d(out_C, self.orig_img_channels, kernel_size=3, padding=1, stride=1)
        self.final = nn.Conv2d(self.orig_img_channels, self.orig_img_channels * 2, kernel_size=3, padding=1, stride=1)

    def forward(self, text_emb, attn_mask):  # time_emb будет None
        device = text_emb.device
        x = torch.zeros(hyperparams.BATCH_SIZE, hyperparams.CHANNELS, hyperparams.IMG_SIZE,
                        hyperparams.IMG_SIZE).to(device)
        # 1. Берем FFT
        x_fft = torch.fft.fft2(x)
        amp = torch.sqrt(x_fft.real ** 2 + x_fft.imag ** 2 + 1e-8)
        phase = torch.atan2(x_fft.imag, x_fft.real)
        x = amp
        # 2. Преобразуем только амплитуду
        x = self.act_1(x)
        x = self.conv_1(x)
        in_attn = x
        x = self.CA(x, text_emb, attn_mask)
        x = self.dropout(x)
        out_attn = x
        x = in_attn + out_attn
        x = self.act_2(x)
        x = self.conv_2(x)

        # применение IFFT
        # 3. Восстанавливаем комплексный спектр
        real_part = x * torch.cos(phase)
        imag_part = x * torch.sin(phase)
        x_fft_processed = torch.complex(real_part, imag_part)
        # 4. IFFT
        x = torch.fft.ifft2(x_fft_processed).real
        x = self.final(x)
        log_D, mu = torch.chunk(x, chunks=2, dim=1)  # по канальному измерению
        return log_D, mu
