import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split, DataLoader

# --- Гиперпараметры ---
IMG_SIZE = 32  # Размер изображений
T = 1000  # Количество шагов в диффузии
BATCH_SIZE = 32
LR = 1e-4
TEXT_EMB_DIM = 768


# --- Определение форвардного процесса (зашумление) ---
def forward_diffusion(x0, t, alphas_bar, noise=None):
    """ Добавляет стандартный гауссовский шум к изображению """
    if noise is None:
        noise = torch.randn_like(x0)
    xt = torch.sqrt(alphas_bar[t]) * x0 + torch.sqrt(1 - alphas_bar[t]) * noise
    return xt


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, t):
        """t — это тензор со значениями [0, T], размерность (B,)"""
        half_dim = self.embed_dim // 2
        # freqs = torch.exp(-torch.arange(half_dim, dtype=torch.float32) * (torch.log(torch.tensor(10000.0)) / half_dim))
        freqs = torch.exp(-torch.arange(half_dim, dtype=torch.float32) * (torch.log(torch.tensor(10000.0)) / half_dim))
        angles = t[:, None] * freqs[None, :]
        time_embedding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        return time_embedding  # (B, embed_dim)


class UNetEncBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)  # Преобразуем `t`
        self.relu = nn.ReLU()

    # x=(B, inC, H, W)->x=(B, outC, H, W); time_emb=(B, te)->time_emb=(B, outC, 1, 1) (te должен быть равен time_emb_dim)
    def forward(self, x, time_emb):
        t_emb = self.time_mlp(time_emb)[:, :, None, None]  # (B, outC) -> (B, outC, 1, 1)
        x = self.conv(x) + t_emb  # Добавляем `t` к фичам
        return self.relu(x)


class UNetDecBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)  # Преобразуем `t`
        self.relu = nn.ReLU()

    # x=(B, inC, H, W)->x=(B, outC, H, W); time_emb=(B, te)->time_emb=(B, outC, 1, 1) (te должен быть равен time_emb_dim)
    def forward(self, x, time_emb):
        t_emb = self.time_mlp(time_emb)[:, :, None, None]  # (B, outC) -> (B, outC, 1, 1)
        x = self.deconv(x) + t_emb  # Добавляем `t` к фичам
        return x


class CrossAttentionMultiHead(nn.Module):
    def __init__(self, text_emb_dim, C, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (C // num_heads) ** -0.5  # Масштабируем по размеру одной головы
        # Приведение текстового эмбеддинга к C
        self.Wq = nn.Linear(C, C)  # Query из изображения
        self.Wk = nn.Linear(text_emb_dim, C)  # Key из текста
        self.Wv = nn.Linear(text_emb_dim, C)  # Value из текста







    def forward(self, x, text_emb):
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
        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_out = torch.matmul(attn_probs, V)  # (B, num_heads, H*W, C//num_heads)
        # Объединяем головы
        attn_out = attn_out.transpose(1, 2).reshape(B, H * W, C)  # (B, H*W, C)

        B, HW, C = attn_out.shape
        attn_out = attn_out.permute(0, 2, 1).view(B, C, H,
                                                  W)  # (B, C, H, W) (H и W не меняются, поэтому делаем преобразование без доп. проверок)
        return attn_out

    def forward(self, x, text_emb, attention_mask=None):
        B, C, H, W = x.shape
        x_flat = x.view(B, C, H * W).permute(0, 2, 1)  # (B, H*W, C)

        Q = self.Wq(x_flat)  # (B, H*W, C)
        K = self.Wk(text_emb)  # (B, T, C)
        V = self.Wv(text_emb)  # (B, T, C)

        Q = Q.view(B, -1, self.num_heads, C // self.num_heads).transpose(1, 2)
        K = K.view(B, -1, self.num_heads, C // self.num_heads).transpose(1, 2)
        V = V.view(B, -1, self.num_heads, C // self.num_heads).transpose(1, 2)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # (B, num_heads, H*W, T)

        if attention_mask is not None:
            attention_mask = attention_mask[:, None, None, :].expand_as(attn_scores)  # (B, 1, 1, T)
            attn_scores = attn_scores.masked_fill(attention_mask == 0, float('-inf'))

        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_out = torch.matmul(attn_probs, V)

        attn_out = attn_out.transpose(1, 2).reshape(B, H * W, C)
        attn_out = attn_out.permute(0, 2, 1).view(B, C, H, W)
        return attn_out


class DeepBottleneck(nn.Module):
    def __init__(self, text_emb_dim):
        super().__init__()

        self.deep_block_1 = UNetEncBlock(128, 256, 256)
        self.deep_block_2 = UNetEncBlock(256, 256, 256)
        self.deep_block_3 = UNetEncBlock(256, 256, 256)

        self.cross_attn_multi_head = CrossAttentionMultiHead(text_emb_dim, 256)  # Cross Attention на уровне bottleneck
        self.residual = nn.Conv2d(128, 256,
                                  kernel_size=1)  # Residual Block (для совпадения количества каналов, применяем слой свёртки)
        self.relu = nn.ReLU()

    def forward(self, x, text_emb, time_emb):
        res = x  # Для Residual Connection
        x = self.deep_block_1(x, time_emb)
        x = self.deep_block_2(x, time_emb)
        x = self.deep_block_3(x, time_emb)
        x = self.cross_attn_multi_head(x, text_emb)
        x = self.relu(x)
        x = x + self.residual(res)
        return x


class MyUNet(nn.Module):
    def __init__(self, TXT_EMB_DIM):
        super().__init__()

        self.time_embedding = SinusoidalTimeEmbedding(
            256)  # (это число (256) должно соотвествовать 3-ему инициализированному числу в UNetBlock)
        # это число означает размерность эмбеддинга одного числа t
        # --- Downsampling (Сжатие) ---
        self.unet_enc_1 = UNetEncBlock(3, 64, 256)
        self.unet_enc_2 = UNetEncBlock(64, 128, 256)

        self.pool = nn.MaxPool2d(2, 2)  # Уменьшает размер изображения в 2 раза

        # --- Bottleneck (Самая узкая часть) ---
        self.deep_bottleneck = DeepBottleneck(TXT_EMB_DIM)

        # --- Upsampling (Расширение) ---
        self.up1 = UNetDecBlock(256, 128, 256)
        self.cross_attn_upsamling_1 = CrossAttentionMultiHead(TXT_EMB_DIM, 128)
        self.dec1 = nn.Conv2d(128 + 128, 64, kernel_size=3, padding=1)  # Skip connection

        self.up2 = UNetDecBlock(64, 64, 256)
        self.cross_attn_upsamling_2 = CrossAttentionMultiHead(TXT_EMB_DIM, 64)
        self.dec2 = nn.Conv2d(64 + 64, 3, kernel_size=3, padding=1)  # Skip connection

    def forward(self, x, text_emb, time_t):
        # --- Encoder (Downsampling) ---
        time_emb = self.time_embedding(time_t)
        x1 = self.unet_enc_1(x, time_emb)  # time_t = (B, t), # x1 = (B, 64, H, W)
        x1_pooled = self.pool(x1)  # (B, 64, H/2, W/2)
        x2 = self.unet_enc_2(x1_pooled, time_emb)  # (B, 128, H/2, W/2)
        x2_pooled = self.pool(x2)  # (B, 128, H/4, W/4)

        # --- Bottleneck ---
        bottleneck = self.deep_bottleneck(x2_pooled, text_emb, time_emb)  # (B, 256, H/4, W/4)
        # --- Decoder (Upsampling) ---
        x_up1 = self.up1(bottleneck, time_emb)  # (B, 128, H/2, W/2)
        x_up1_attn = self.cross_attn_upsamling_1(x_up1, text_emb)  # (B, 128, H/2, W/2)
        x_concat1 = torch.cat([x_up1_attn, x2], dim=1)  # (B, 128+128, H/2, W/2)
        x_dec1 = F.relu(self.dec1(x_concat1))  # (B, 64, H/2, W/2)

        x_up2 = self.up2(x_dec1, time_emb)  # (B, 64, H, W)
        x_up2_attn = self.cross_attn_upsamling_2(x_up2, text_emb)  # (B, 64, H, W)
        x_concat2 = torch.cat([x_up2_attn, x1], dim=1)  # (B, 64+64, H, W)
        x_dec2 = F.relu(self.dec2(x_concat2))  # (B, 3, H, W)
        return x_dec2


# --- Создание модели ---
def create_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MyUNet(TEXT_EMB_DIM).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()
    return device, model, optimizer, criterion


# # --- Создание датасета для обучения ---
# def create_dataset(image_captions):
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     model = MyUNet(TEXT_EMB_DIM).to(device)
#     optimizer = optim.Adam(model.parameters(), lr=LR)
#     criterion = nn.MSELoss()
#     return device, model, optimizer, criterion


# # --- Функция обучения ---
# def train_ddpm(model, dataset, epochs=10):
#     beta = torch.linspace(0.0001, 0.02, T)  # Линейно возрастающие β_t
#     alpha = 1 - beta  # α_t
#     alphas_bar = torch.cumprod(alpha, dim=0)  # Накапливаемый коэффициент ᾱ_t
#     print(alphas_bar.shape)  # Должно быть [1000] (для каждого t своё значение)
#     model.train()
#     for epoch in range(epochs):
#         for x0, _ in dataset:
#             x0 = x0.to(device)
#             t = torch.randint(0, T, (BATCH_SIZE,), device=device)  # случайные шаги t
#
#             xt = forward_diffusion(x0, t, alphas_bar)  # добавляем шум
#             predicted_noise = model(xt)  # модель предсказывает шум
#             loss = criterion(predicted_noise, torch.randn_like(xt))  # сравниваем с реальным шумом
#
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#
#         print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")



def collate_fn(batch):
    images, text_embs, masks = zip(*batch)  # Разбираем батч по частям
    images = torch.stack(images)  # Объединяем картинки (B, C, H, W)
    text_embs = torch.stack(text_embs)  # Объединяем текстовые эмбеддинги (B, max_length, txt_emb_dim)
    masks = torch.stack(masks)  # Объединяем маски внимания (B, max_length)
    return images, text_embs, masks


# --- Функция обучения ---
def train_ddpm(model, optimizer, dataset, epochs):
    train_size = int(0.75 * len(dataset))
    val_size = len(dataset) - train_size
    # Разделяем датасет
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)  # Валидацию можно не перемешивать

    beta = torch.linspace(0.0001, 0.02, T)  # Линейно возрастающие β_t
    alpha = 1 - beta  # α_t
    alphas_bar = torch.cumprod(alpha, dim=0)  # Накапливаемый коэффициент ᾱ_t

    for epoch in range(epochs):
        model.train()  # Включаем режим обучения
        for images, text_embs, attention_mask in train_loader:
            optimizer.zero_grad()



            # Прямой проход (Forward)
            predicted_noise = model(images, encoder_hidden_states=text_embs, attention_mask=attention_mask)

            loss = loss_function(predicted_noise, target)  # Вычисляем ошибку
            loss.backward()  # Обратный проход (Backpropagation)
            optimizer.step()  # Обновляем веса модели

        # Оценка на валидационном датасете
        model.eval()  # Переключаем в режим валидации
        with torch.no_grad():
            for images, text_embs, attention_mask in val_loader:
                predicted_noise = model(images, encoder_hidden_states=text_embs, attention_mask=attention_mask)
                val_loss = loss_function(predicted_noise, target)

        print(f"Epoch {epoch + 1}, Train Loss: {loss.item()}, Val Loss: {val_loss.item()}")

    # for epoch in range(epochs):
    #     for x0, _ in dataset:
    #         x0 = x0.to(device)
    #         t = torch.randint(0, T, (BATCH_SIZE,), device=device)  # случайные шаги t
    #
    #         xt = forward_diffusion(x0, t, alphas_bar)  # добавляем шум
    #         predicted_noise = model(xt)  # модель предсказывает шум
    #         loss = criterion(predicted_noise, torch.randn_like(xt))  # сравниваем с реальным шумом
    #
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #
    #     print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")
