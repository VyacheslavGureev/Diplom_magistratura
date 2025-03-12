#
# import torch.fft
#
#
# # --- Attention в частотной области ---
# class FrequencyAttention(nn.Module):
#     def __init__(self, embed_dim):
#         super().__init__()
#         self.Wq = nn.Linear(embed_dim, embed_dim)
#         self.Wk = nn.Linear(embed_dim, embed_dim)
#         self.Wv = nn.Linear(embed_dim, embed_dim)
#         self.softmax = nn.Softmax(dim=-1)
#
#     def forward(self, noise, text_emb):
#         noise_fft = torch.fft.fft(noise, dim=-1)  # FFT преобразование
#         Q = self.Wq(noise_fft.real)  # Query = реальные частоты шума
#         K = self.Wk(text_emb)  # Key = текстовое описание
#         V = self.Wv(noise_fft.real)  # Value = частоты шума
#
#         attention_scores = self.softmax(Q @ K.T / torch.sqrt(torch.tensor(K.shape[-1], dtype=torch.float32)))
#         modified_freqs = attention_scores @ V  # Меняем частоты шума
#
#         # Собираем обратно (оставляем фазу неизменной)
#         modified_noise_fft = torch.complex(modified_freqs, noise_fft.imag)
#         adapted_noise = torch.fft.ifft(modified_noise_fft, dim=-1).real
#
#         return adapted_noise
#
#
# # --- Встраиваем FFT-Attention в диффузионку ---
# class FFTDiffusionModel(nn.Module):
#     def __init__(self, img_size, embed_dim):
#         super().__init__()
#         self.attn = FrequencyAttention(embed_dim)
#         self.unet = SimpleUNet(img_size)
#
#     def forward(self, xt, text_emb):
#         adapted_noise = self.attn(xt, text_emb)
#         xt_attn = xt + adapted_noise  # Добавляем адаптивный шум
#         predicted_noise = self.unet(xt_attn)  # Восстанавливаем шум через U-Net
#         return predicted_noise
#
#
# # --- Создание модели ---
# fft_model = FFTDiffusionModel(IMG_SIZE, 256).to(device)
# optimizer_fft = optim.Adam(fft_model.parameters(), lr=LR)
#
#
# # --- Функция обучения ---
# def train_fft_ddpm(model, dataset, epochs=10):
#     model.train()
#     for epoch in range(epochs):
#         for x0, text_emb in dataset:
#             x0, text_emb = x0.to(device), text_emb.to(device)
#             t = torch.randint(0, T, (BATCH_SIZE,), device=device)
#             xt = forward_diffusion(x0, t, alphas_bar)
#
#             predicted_noise = model(xt, text_emb)
#             loss = criterion(predicted_noise, torch.randn_like(xt))
#
#             optimizer_fft.zero_grad()
#             loss.backward()
#             optimizer_fft.step()
#
#         print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")
#
#
# print("FFT-Attention Diffusion Model готова! ✅")
#
#
# # --- Загружаем датасет (например, COCO-Text) ---
# train_dataset = DataLoader(COCOTextDataset(), batch_size=BATCH_SIZE, shuffle=True)
#
# # --- Обучаем обе модели ---
# print("Обучаем стандартный DDPM...")
# train_ddpm(model, train_dataset, epochs=10)
#
# print("Обучаем FFT-Attention Diffusion...")
# train_fft_ddpm(fft_model, train_dataset, epochs=10)
#
# # --- Сохраняем модели ---
# torch.save(model.state_dict(), "ddpm_model.pth")
# torch.save(fft_model.state_dict(), "fft_attention_model.pth")
#
# print("Модели обучены и сохранены! 🚀")










# 12.03.2025
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
#
# # --- Гиперпараметры ---
# IMG_SIZE = 32  # Размер изображений
# T = 1000  # Количество шагов в диффузии
# BATCH_SIZE = 32
# LR = 1e-4
#
#
# # --- Определение форвардного процесса (зашумление) ---
# def forward_diffusion(x0, t, alphas_bar, noise=None):
#     """ Добавляет стандартный гауссовский шум к изображению """
#     if noise is None:
#         noise = torch.randn_like(x0)
#     xt = torch.sqrt(alphas_bar[t]) * x0 + torch.sqrt(1 - alphas_bar[t]) * noise
#     return xt
#
#
# class SinusoidalTimeEmbedding(nn.Module):
#     def __init__(self, embed_dim):
#         super().__init__()
#         self.embed_dim = embed_dim
#
#     def forward(self, t):
#         """t — это тензор со значениями [0, T], размерность (B,)"""
#         half_dim = self.embed_dim // 2
#         freqs = torch.exp(-torch.arange(half_dim, dtype=torch.float32) * (torch.log(torch.tensor(10000.0)) / half_dim))
#         angles = t[:, None] * freqs[None, :]
#         time_embedding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
#         return time_embedding  # (B, embed_dim)
#
#
# class CrossAttention(nn.Module):
#     def __init__(self, channels, num_heads=8):
#         super().__init__()
#         self.num_heads = num_heads
#         self.scale = (channels // num_heads) ** -0.5
#         self.Wq = nn.Linear(channels, channels)  # Query для изображения
#         self.Wk = nn.Linear(channels, channels)  # Key для текста
#         self.Wv = nn.Linear(channels, channels)  # Value для текста
#
#         self.softmax = nn.Softmax(dim=-1)
#
#     def forward(self, image_features, text_features):
#         Q = self.Wq(image_features)  # (B, H*W, channels)
#         K = self.Wk(text_features)  # (B, T, channels)
#         V = self.Wv(text_features)  # (B, T, channels)
#
#         attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # (B, H*W, T)
#         attn_weights = self.softmax(attn_scores)  # (B, H*W, T)
#
#         output = torch.matmul(attn_weights, V)  # (B, H*W, channels)
#         return output
#
#
# class DeepBottleneck(nn.Module):
#     def __init__(self, text_emb_dim):
#         super().__init__()
#
#         self.deep_block_1 = UNetBlock(256, 256, 256)
#         self.deep_block_2 = UNetBlock(256, 256, 256)
#         self.deep_block_3 = UNetBlock(256, 256, 256)
#
#         # self.deep_block_1 = DeepBottleneckBlock(256, 256, 256)
#         # self.deep_block_2 = DeepBottleneckBlock(256, 256, 256)
#         # self.deep_block_3 = DeepBottleneckBlock(256, 256, 256)
#         # self.conv1 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
#         # self.conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
#         # self.conv3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
#         self.proj_text = nn.Linear(text_emb_dim, 256)  # Приводим CLIP эмбеддинг к C
#
#         self.cross_attn = CrossAttention(256)  # Cross Attention на уровне bottleneck
#         self.residual = nn.Conv2d(256, 256,
#                                   kernel_size=1)  # Residual Block (для совпадения количества каналов, применяем слой свёртки)
#
#     def forward(self, x, text_emb, time_emb):
#         res = x  # Для Residual Connection
#         x = self.deep_block_1(x, time_emb)
#         x = self.deep_block_2(x, time_emb)
#         x = self.deep_block_3(x, time_emb)
#         # x = F.relu(self.conv1(x))
#         # x = F.relu(self.conv2(x))
#         # x = F.relu(self.conv3(x))
#
#         B, C, H, W = x.shape
#         x_flat = x.view(B, C, H * W).permute(0, 2, 1)  # (B, H*W, C)
#         proj_text_emb = text_emb
#         if text_emb.size(2) != x_flat.size(2):
#             proj_text_emb = self.proj_text(text_emb)
#         x_attn = self.cross_attn(x_flat, proj_text_emb)  # Применяем Attention
#         x_attn = x_attn.permute(0, 2, 1).view(B, C, H, W)  # (B, C, H, W)
#
#         # x = apply_cross_attention(x, self.cross_attn, self.proj_text, text_emb)
#         # x = self.cross_attn(x, text_emb)  # Добавляем текстовый контекст
#         x = F.relu(x_attn)  # ???
#         x = x + self.residual(res)  # Residual Skip Connection
#         return x
#
#
# # class DeepBottleneckBlock(nn.Module):
# #     def __init__(self, in_channels, out_channels, time_emb_dim):
# #         super().__init__()
# #         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
# #         self.time_mlp = nn.Linear(time_emb_dim, out_channels)  # Преобразуем `t`
# #         self.relu = nn.ReLU()
# #
# #     def forward(self, x, time_emb):
# #         t_emb = self.time_mlp(time_emb)[:, :, None, None]  # (B, C) -> (B, C, 1, 1)
# #         x = self.conv(x) + t_emb  # Добавляем `t` к фичам
# #         return self.relu(x)
#
#
# class UNetBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, time_emb_dim):
#         super().__init__()
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
#         self.time_mlp = nn.Linear(time_emb_dim, out_channels)  # Преобразуем `t`
#         self.relu = nn.ReLU()
#
#     # x=(B, inC, H, W)->x=(B, outC, H, W); time_emb=(B, te)->time_emb=(B, outC, 1, 1) (te должен быть равен time_emb_dim)
#     def forward(self, x, time_emb):
#         t_emb = self.time_mlp(time_emb)[:, :, None, None]  # (B, outC) -> (B, outC, 1, 1)
#         x = self.conv(x) + t_emb  # Добавляем `t` к фичам
#         return self.relu(x)
#
#
# class UNet(nn.Module):
#     def __init__(self, TXT_EMB_DIM):
#         super().__init__()
#
#         self.time_embedding = SinusoidalTimeEmbedding(
#             256)  # (это число (256) должно соотвествовать 3-ему инициализированному числу в UNetBlock)
#         # --- Downsampling (Сжатие) ---
#         self.unet_enc_1 = UNetBlock(3, 64, 256)
#         self.unet_enc_2 = UNetBlock(64, 128, 256)
#
#         # self.enc1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
#         # self.enc2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
#         self.pool = nn.MaxPool2d(2, 2)  # Уменьшает размер изображения в 2 раза
#
#         # --- Bottleneck (Самая узкая часть) ---
#         self.deep_bottleneck = DeepBottleneck(TXT_EMB_DIM)
#         # self.bottleneck = nn.Conv2d(128, 256, kernel_size=3, padding=1)
#
#         # self.cross_attn_bottleneck = CrossAttention(embed_dim)
#         self.cross_attn_upsamling_1 = CrossAttention(128)
#         self.proj_text_up_1 = nn.Linear(TXT_EMB_DIM, 128)
#         self.cross_attn_upsamling_2 = CrossAttention(64)
#         self.proj_text_up_2 = nn.Linear(TXT_EMB_DIM, 64)
#
#         # --- Upsampling (Расширение) ---
#         self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
#         self.dec1 = nn.Conv2d(128 + 128, 64, kernel_size=3, padding=1)  # Skip connection
#         self.up2 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
#         self.dec2 = nn.Conv2d(64 + 64, 3, kernel_size=3, padding=1)  # Skip connection
#
#     def forward(self, x, text_emb, time_t):
#         # --- Encoder (Downsampling) ---
#         x1 = self.unet_enc_1(x, self.time_embedding(time_t))  # time_t = (B, t)
#         # x1 = F.relu(self.enc1(x))  # (B, 64, H, W)
#         x1_pooled = self.pool(x1)  # (B, 64, H/2, W/2)
#         x2 = self.unet_enc_2(x1_pooled, self.time_embedding(time_t))
#         # x2 = F.relu(self.enc2(x1_pooled))  # (B, 128, H/2, W/2)
#         x2_pooled = self.pool(x2)  # (B, 128, H/4, W/4)
#
#         # --- Bottleneck ---
#         bottleneck = self.deep_bottleneck(x2_pooled, text_emb)  # (B, 256, H/4, W/4)
#
#         # --- Decoder (Upsampling) ---
#         x_up1 = self.up1(bottleneck)  # (B, 128, H/2, W/2)
#
#         x = x_up1
#         B, C, H, W = x.shape
#         x_flat = x.view(B, C, H * W).permute(0, 2, 1)  # (B, H*W, C)
#         proj_text_emb = text_emb
#         if text_emb.size(2) != x_flat.size(2):
#             proj_text_emb = self.proj_text_up_1(text_emb)
#         x_attn = self.cross_attn_upsamling_1(x_flat, proj_text_emb)  # Применяем Attention
#         x_attn = x_attn.permute(0, 2, 1).view(B, C, H, W)  # (B, C, H, W)
#         x_up1_attn = x_attn
#         x_concat1 = torch.cat([x_up1_attn, x2], dim=1)
#         x_dec1 = F.relu(self.dec1(x_concat1))
#
#         x_up2 = self.up2(x_dec1)  # (B, 64, H, W)
#
#         x = x_up2
#         B, C, H, W = x.shape
#         x_flat = x.view(B, C, H * W).permute(0, 2, 1)  # (B, H*W, C)
#         proj_text_emb = text_emb
#         if text_emb.size(2) != x_flat.size(2):
#             proj_text_emb = self.proj_text_up_2(text_emb)
#         x_attn = self.cross_attn_upsamling_2(x_flat, proj_text_emb)  # Применяем Attention
#         x_attn = x_attn.permute(0, 2, 1).view(B, C, H, W)  # (B, C, H, W)
#         x_up2_attn = x_attn
#         x_concat2 = torch.cat([x_up2_attn, x1], dim=1)
#         x_dec2 = F.relu(self.dec2(x_concat2))
#
#         return x_dec2
#
#
# # --- Создание модели ---
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model = UNet(IMG_SIZE).to(device)
# optimizer = optim.Adam(model.parameters(), lr=LR)
# criterion = nn.MSELoss()
#
#
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
#
# # print("Базовая DDPM модель готова! ✅")
