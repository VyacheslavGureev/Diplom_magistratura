import torch
import torch.nn as nn
import time
import models.hyperparams as hyperparams


# TODO: Предварительно всё правильно

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

        # Инициализация
        # nn.init.xavier_uniform_(self.to_q.weight, gain=0.02)
        # nn.init.xavier_uniform_(self.to_kv.weight, gain=0.02)
        # nn.init.zeros_(self.to_q.bias)
        # nn.init.zeros_(self.to_kv.bias)

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
        # out = x_flat + 1.0 * out  # Малый вес для стабильности (0,5 пока подходит)
        out = out.permute(0, 2, 1).view(B, C, H, W)
        return out


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

        # self.silu_1 = nn.SiLU()
        # self.silu_2 = nn.SiLU()

        self.act_1 = nn.Tanh()
        self.act_2 = nn.Tanh()

        self.dropout = nn.Dropout(p=0.1)  # 10% нейронов будут обнулены

    def forward(self, x, time_emb):
        if time_emb != None:
            x = x + self.te_block(time_emb)[:, :, None, None]
        r = self.norm_1(x)
        r = self.act_1(r)
        r = self.conv_1(r)

        r = self.dropout(r)

        r = self.norm_2(r)
        r = self.act_2(r)
        r = self.conv_2(r)
        r = r + self.residual(x)
        return r


class DeepBottleneck(nn.Module):
    def __init__(self, in_C, out_C, text_emb_dim, time_emb_dim, batch_size):
        super().__init__()
        self.batch_size = batch_size
        self.deep_block_1 = ResNetBlock(in_C, out_C, time_emb_dim, self.batch_size)
        self.cross_attn_multi_head_1 = CrossAttentionMultiHead(text_emb_dim, out_C)

    # здесь правильный порядок преобразований!!!
    def forward(self, x, text_emb, time_emb, attention_mask):
        x = self.deep_block_1(x, time_emb)
        if text_emb != None:
            x = self.cross_attn_multi_head_1(x, text_emb, attention_mask)
        return x


class SoftUpsample(nn.Module):
    def __init__(self, in_C, out_C):
        super().__init__()
        # Увеличиваем размер изображения в 2 раза, можем уменьшить кол-во каналов
        self.very_soft_up = nn.Sequential(
            # Более мягкая версия увеличения в 2 раза, помогающая убрать шахматные артефакты
            nn.ConvTranspose2d(in_C, out_C, kernel_size=3, stride=2, padding=1,
                               output_padding=1),
            # Сглаживающая свёртка
            nn.Conv2d(out_C, out_C, kernel_size=3, padding=1, stride=1),
            # Максимальная мягкость
            nn.SiLU()
        )

    def forward(self, x):
        x = self.very_soft_up(x)
        return x


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

        # self.norm_1 = nn.LayerNorm([hyperparams.IMG_SIZE, hyperparams.IMG_SIZE])
        self.conv_1 = nn.Conv2d(self.orig_img_channels, out_C, kernel_size=3, padding=1, stride=1)

        self.CA = CrossAttentionMultiHead(self.txt_emb_dim, out_C, 2)
        self.dropout = nn.Dropout(0.1)

        # self.norm_2 = nn.LayerNorm([hyperparams.IMG_SIZE, hyperparams.IMG_SIZE])
        self.act_2 = nn.GELU()
        self.conv_2 = nn.Conv2d(out_C, self.orig_img_channels, kernel_size=3, padding=1, stride=1)
        self.final = nn.Conv2d(self.orig_img_channels, self.orig_img_channels * 2, kernel_size=3, padding=1, stride=1)


        # self.prepare = nn.Sequential(
        #     nn.Softplus(),
        #     nn.Conv2d(self.orig_img_channels, out_C, kernel_size=3, padding=1, stride=1),
        # )
        # in_C = out_C
        # self.ca = CrossAttentionMultiHead(self.txt_emb_dim, in_C, 2)
        # self.final = nn.Sequential(
        #     nn.GELU(),
        #     nn.Conv2d(in_C, self.orig_img_channels, kernel_size=3, padding=1, stride=1)
        # )

        self.mu = 0  # среднее
        self.D = 1  # дисперсия

    def create_down_block(self, in_channels, out_channels, time_emb_dim, batch_size, use_self_attn):
        if not use_self_attn:
            return ResNetBlock(in_channels, out_channels, time_emb_dim, batch_size)
        else:
            return ResNetBlock(in_channels, out_channels, time_emb_dim, batch_size)
            # return ResNetMiddleBlock(in_channels, out_channels, time_emb_dim, batch_size)

    def create_bottleneck_block(self, in_channels, out_channels, text_emb_dim, time_emb_dim, batch_size):
        return DeepBottleneck(in_channels, out_channels, text_emb_dim, time_emb_dim, batch_size)

    def create_up_block(self, in_channels, out_channels, time_emb_dim, batch_size,
                        use_self_attn):
        if use_self_attn:
            return ResNetBlock(in_channels, out_channels, time_emb_dim, batch_size)
            # return ResNetMiddleBlock(in_channels, out_channels, time_emb_dim, batch_size)
        else:
            return ResNetBlock(in_channels, out_channels, time_emb_dim, batch_size)

    def get_current_variance(self, train_loader, text_descr_loader, device):
        was_training = self.training  # True если train(), False если eval()
        self.eval()
        all_outputs = []
        start_time_variance = time.time()
        with torch.no_grad():
            for text_embs, attention_mask in text_descr_loader:
                start_time_noises = time.time()
                for _ in range(100):  # 100 разных шумов на один текст
                    text_embs, attention_mask = text_embs.to(device), attention_mask.to(device)
                    noise = torch.randn(hyperparams.BATCH_SIZE, hyperparams.CHANNELS, hyperparams.IMG_SIZE,
                                        hyperparams.IMG_SIZE).to(device)
                    # with torch.cuda.amp.autocast():  # Включаем AMP (включение повышает дисперсию, а это плохо)
                    output = self(noise, text_embs, None, attention_mask)
                    all_outputs.append(output.detach())
                print(f'Много шумов на один текст: {time.time() - start_time_noises}')
        print(f'Замер дисперсии: {time.time() - start_time_variance}')
        outputs_tensor = torch.cat(all_outputs, dim=0)
        self.mu = outputs_tensor.mean()  # mu - это скаляр
        self.D = outputs_tensor.std()  # D - это тоже скаляр
        print(f'D: {self.D}, mu: {self.mu}')
        if was_training:
            self.train()
        else:
            self.eval()
        return self.mu, self.D

    def forward(self, text_emb, attn_mask):  # time_emb будет None
        # was_training = self.training  # True если train(), False если eval()
        # self.eval()

        # if self.apply_fft:
        device = text_emb.device
        x = torch.zeros(hyperparams.BATCH_SIZE, hyperparams.CHANNELS, hyperparams.IMG_SIZE,
                            hyperparams.IMG_SIZE).to(device)
        # x = torch.randn(hyperparams.BATCH_SIZE, hyperparams.CHANNELS, hyperparams.IMG_SIZE,
        #                     hyperparams.IMG_SIZE).to(device)
        # 1. Берем FFT
        x_fft = torch.fft.fft2(x)
        amp = torch.sqrt(x_fft.real ** 2 + x_fft.imag ** 2 + 1e-8)
        phase = torch.atan2(x_fft.imag, x_fft.real)
        x = amp
        # 2. Прогоняем только амплитуду через нейронку
        x = self.act_1(x)
        # x = self.norm_1(x)
        x = self.conv_1(x)
        in_attn = x
        x = self.CA(x, text_emb, attn_mask)
        x = self.dropout(x)
        out_attn = x
        x = in_attn + out_attn
        # x = self.norm_2(x)
        x = self.act_2(x)
        x = self.conv_2(x)

        # x = self.prepare(x)
        # x = self.ca(x, text_emb, attension_mask)
        # x = self.final(x)

        # применение IFFT
        # if self.apply_fft:
            # 3. Восстанавливаем комплексный спектр
        real_part = x * torch.cos(phase)
        imag_part = x * torch.sin(phase)
        x_fft_processed = torch.complex(real_part, imag_part)
        # 4. IFFT
        x = torch.fft.ifft2(x_fft_processed).real
        x = self.final(x)
        log_D, mu = torch.chunk(x, chunks=2, dim=1)  # по канальному измерению

        # if was_training:
        #     self.train()
        # else:
        #     self.eval()
        return log_D, mu
