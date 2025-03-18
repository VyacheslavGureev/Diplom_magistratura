import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
import matplotlib.pyplot as plt
import random

# --- Гиперпараметры ---
# IMG_SIZE = 32  # Размер изображений
T = 1000  # Количество шагов в диффузии
BATCH_SIZE = 8
LR = 1e-4
TEXT_EMB_DIM = 512


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, embed_dim, device):
        super().__init__()
        self.embed_dim = embed_dim
        self.device = device

    def forward(self, t):
        """t — это тензор со значениями [0, T], размерность (B,)"""
        half_dim = self.embed_dim // 2
        # freqs = torch.exp(-torch.arange(half_dim, dtype=torch.float32) * (torch.log(torch.tensor(10000.0)) / half_dim))
        freqs = torch.exp(
            -torch.arange(half_dim, dtype=torch.float32) * (torch.log(torch.tensor(10000.0)) / half_dim)).to(
            self.device)
        angles = t[:, None] * freqs[None, :]
        time_embedding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        return time_embedding  # (B, embed_dim)


class UNetEncBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)  # Преобразуем `t`
        self.bn = nn.BatchNorm2d(out_channels)  # BatchNorm для out_channels каналов
        self.relu = nn.ReLU()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_emb_dim = time_emb_dim

    # x=(B, inC, H, W)->x=(B, outC, H, W); time_emb=(B, te)->time_emb=(B, outC, 1, 1) (te должен быть равен time_emb_dim)
    def forward(self, x, time_emb):
        x = self.conv(x)
        t_emb = self.time_mlp(time_emb)[:, :, None, None]  # (B, outC) -> (B, outC, 1, 1)
        x = x + t_emb
        # x = self.conv(x) + t_emb  # Добавляем `t` к фичам
        x = self.bn(x)
        return self.relu(x)


class UNetDecBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)  # Преобразуем `t`
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_emb_dim = time_emb_dim

    # x=(B, inC, H, W)->x=(B, outC, H, W); time_emb=(B, te)->time_emb=(B, outC, 1, 1) (te должен быть равен time_emb_dim)
    def forward(self, x, time_emb):
        x = self.deconv(x)
        t_emb = self.time_mlp(time_emb)[:, :, None, None]  # (B, outC) -> (B, outC, 1, 1)
        x = x + t_emb
        # x = self.deconv(x) + t_emb  # Добавляем `t` к фичам
        x = self.bn(x)
        return self.relu(x)


class CrossAttentionMultiHead(nn.Module):
    def __init__(self, text_emb_dim, C, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (C // num_heads) ** -0.5  # Масштабируем по размеру одной головы
        # Приведение текстового эмбеддинга к C
        self.Wq = nn.Linear(C, C)  # Query из изображения
        self.Wk = nn.Linear(text_emb_dim, C)  # Key из текста
        self.Wv = nn.Linear(text_emb_dim, C)  # Value из текста
        self.text_emb_dim = text_emb_dim
        self.C = C

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


class DeepBottleneck(nn.Module):
    def __init__(self, text_emb_dim):
        super().__init__()

        self.deep_block_1 = UNetEncBlock(128, 256, 256)
        self.deep_block_2 = UNetEncBlock(256, 256, 256)
        self.deep_block_3 = UNetEncBlock(256, 256, 256)

        self.cross_attn_multi_head = CrossAttentionMultiHead(text_emb_dim, 256)  # Cross Attention на уровне bottleneck
        self.residual = nn.Conv2d(128, 256,
                                  kernel_size=1)  # Residual Block (для совпадения количества каналов, применяем слой свёртки)
        self.bn = nn.BatchNorm2d(256)
        self.relu = nn.ReLU()

    def forward(self, x, text_emb, time_emb, attention_mask):
        res = x  # Для Residual Connection
        x = self.deep_block_1(x, time_emb)
        x = self.deep_block_2(x, time_emb)
        x = self.deep_block_3(x, time_emb)
        x = self.cross_attn_multi_head(x, text_emb, attention_mask)
        x = self.bn(x)
        x = self.relu(x)
        r = self.residual(res)
        x = x + r
        return x


class MyUNet(nn.Module):
    def __init__(self, TXT_EMB_DIM, device):
        super().__init__()

        self.time_embedding = SinusoidalTimeEmbedding(
            256, device)  # (это число (256) должно соотвествовать 3-ему инициализированному числу в UNetBlock)
        # это число означает размерность эмбеддинга одного числа t
        # --- Downsampling (Сжатие) ---
        self.unet_enc_1 = UNetEncBlock(3, 64, 256)
        self.unet_enc_2 = UNetEncBlock(64, 128, 256)

        self.pool = nn.MaxPool2d(2, 2)  # Уменьшает размер изображения в 2 раза
        self.relu = nn.ReLU()

        # --- Bottleneck (Самая узкая часть) ---
        self.deep_bottleneck = DeepBottleneck(TXT_EMB_DIM)

        # --- Upsampling (Расширение) ---
        self.up1 = UNetDecBlock(256, 128, 256)
        self.cross_attn_upsamling_1 = CrossAttentionMultiHead(TXT_EMB_DIM, 128)
        self.dec1 = nn.Conv2d(128 + 128, 64, kernel_size=3, padding=1)  # Skip connection
        self.bn_dec_1 = nn.BatchNorm2d(64)

        self.up2 = UNetDecBlock(64, 64, 256)
        self.cross_attn_upsamling_2 = CrossAttentionMultiHead(TXT_EMB_DIM, 64)
        self.dec2 = nn.Conv2d(64 + 64, 3, kernel_size=3, padding=1)  # Skip connection
        self.bn_dec_2 = nn.BatchNorm2d(3)

        self.device = device

    def forward(self, x, text_emb, time_t, attension_mask):
        # --- Encoder (Downsampling) ---
        # print(time_t.device)

        time_emb = self.time_embedding(time_t)
        x1 = self.unet_enc_1(x, time_emb)  # time_t = (B, t), # x1 = (B, 64, H, W)
        x1_pooled = self.pool(x1)  # (B, 64, H/2, W/2)
        x2 = self.unet_enc_2(x1_pooled, time_emb)  # (B, 128, H/2, W/2)
        x2_pooled = self.pool(x2)  # (B, 128, H/4, W/4)

        # --- Bottleneck ---
        bottleneck = self.deep_bottleneck(x2_pooled, text_emb, time_emb, attension_mask)  # (B, 256, H/4, W/4)
        # --- Decoder (Upsampling) ---
        x_up1 = self.up1(bottleneck, time_emb)  # (B, 128, H/2, W/2)
        x_up1_attn = self.cross_attn_upsamling_1(x_up1, text_emb, attension_mask)  # (B, 128, H/2, W/2)
        x_concat1 = torch.cat([x_up1_attn, x2], dim=1)  # (B, 128+128, H/2, W/2)

        x_dec1 = self.relu(self.bn_dec_1(self.dec1(x_concat1)))  # (B, 64, H/2, W/2)

        x_up2 = self.up2(x_dec1, time_emb)  # (B, 64, H, W)
        x_up2_attn = self.cross_attn_upsamling_2(x_up2, text_emb, attension_mask)  # (B, 64, H, W)
        x_concat2 = torch.cat([x_up2_attn, x1], dim=1)  # (B, 64+64, H, W)
        x_dec2 = self.relu(self.bn_dec_2(self.dec2(x_concat2)))  # (B, 3, H, W)
        return x_dec2


class EncapsulatedModel:
    def __init__(self):
        # Создание модели с нуля
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(self.device)
        self.model = MyUNet(TEXT_EMB_DIM, self.device).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)
        self.criterion = nn.MSELoss()
        self.last_loss_item = -1
        self.epoch = -1


class EncapsulatedDataloaders:
    def __init__(self, train, val, test):
        self.train = train
        self.val = val
        self.test = test


class ModelManager():

    def __init__(self):
        pass

    # --- Создание модели ---
    def create_model(self):
        return EncapsulatedModel()

    def create_dataloaders(self, dataset, train_size_percent, val_size_percent):
        # Разделяем датасеты
        train_size = int(train_size_percent * len(dataset))
        val_size = int(val_size_percent * len(dataset))
        test_size = len(dataset) - train_size - val_size
        train_dataset, val_dataset, test_dataset = random_split(dataset,
                                                                [train_size, val_size, test_size])
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                  collate_fn=self.collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                collate_fn=self.collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                 collate_fn=self.collate_fn)  # Тестовый датасет можно не перемешивать
        e_loader = EncapsulatedDataloaders(train_loader, val_loader, test_loader)
        return e_loader

    def collate_fn(self, batch):
        if len(batch) % BATCH_SIZE != 0:
            additional_batch = random.choices(batch, k=BATCH_SIZE - (len(batch) % BATCH_SIZE))
            batch = batch + additional_batch
        images, text_embs, masks = zip(*batch)  # Разбираем батч по частям
        images = torch.stack(images)  # Объединяем картинки (B, C, H, W)
        text_embs = torch.stack(text_embs)  # Объединяем текстовые эмбеддинги (B, max_length, txt_emb_dim)
        masks = torch.stack(masks)  # Объединяем маски внимания (B, max_length)
        return images, text_embs, masks

    # --- Определение форвардного процесса (зашумление) ---
    def forward_diffusion(self, x0, t, alphas_bar, noise=None):
        """ Добавляет стандартный гауссовский шум к изображению """
        if noise is None:
            noise = torch.randn_like(x0)
        at = alphas_bar[t][:, None, None, None]
        xt = torch.sqrt(at) * x0 + torch.sqrt(1 - at) * noise
        return xt

    def train_model(self, e_model: EncapsulatedModel, e_loader, epochs):
        for epoch in range(epochs):
            train_loss = self.training_model(e_model, e_loader)
            val_loss = self.validating_model(e_model, e_loader)
            e_model.last_loss_item = val_loss.item()
            e_model.epoch = epoch + 1
            print(f"Epoch {epoch + 1}, Train Loss: {train_loss.item()}, Val Loss: {val_loss.item()}")

    def training_model(self, e_model: EncapsulatedModel, e_loader: EncapsulatedDataloaders):
        print("Тренировка")
        model = e_model.model
        device = e_model.device
        optimizer = e_model.optimizer
        criterion = e_model.criterion
        train_loader = e_loader.train

        model.train()  # Включаем режим обучения

        beta = torch.linspace(0.0001, 0.02, T)  # Линейно возрастающие b_t
        alpha = 1 - beta  # a_t
        alphas_bar = torch.cumprod(alpha, dim=0).to(device)  # Накапливаемый коэффициент a_t (T,)

        loss = None
        i = 0
        for images, text_embs, attention_mask in train_loader:
            # if i == 10:
            #     break

            optimizer.zero_grad()

            images, text_embs, attention_mask = images.to(device), text_embs.to(device), attention_mask.to(device)
            t = torch.randint(0, T, (BATCH_SIZE,), device=device)  # случайные шаги t

            xt = self.forward_diffusion(images, t, alphas_bar).to(device)  # добавляем шум
            predicted_noise = model(xt, text_embs, t, attention_mask)

            loss_train = criterion(predicted_noise, torch.randn_like(xt))  # сравниваем с реальным шумом
            loss = loss_train
            loss_train.backward()

            optimizer.step()

            i += 1
            print(f"Процентов {(i / len(train_loader)) * 100}")
        return loss

    def validating_model(self, e_model: EncapsulatedModel, e_loader: EncapsulatedDataloaders):
        print("Валидация")
        model = e_model.model
        device = e_model.device
        criterion = e_model.criterion
        val_loader = e_loader.val

        model.eval()  # Переключаем в режим валидации

        beta = torch.linspace(0.0001, 0.02, T)  # Линейно возрастающие b_t
        alpha = 1 - beta  # a_t
        alphas_bar = torch.cumprod(alpha, dim=0).to(device)  # Накапливаемый коэффициент a_t (T,)

        # Оценка на валидационном датасете
        loss = None
        i = 0
        with torch.no_grad():
            for images, text_embs, attention_mask in val_loader:
                # if i == 10:
                #     break

                images, text_embs, attention_mask = images.to(device), text_embs.to(device), attention_mask.to(device)
                t = torch.randint(0, T, (BATCH_SIZE,), device=device)  # случайные шаги t
                xt = self.forward_diffusion(images, t, alphas_bar).to(device)  # добавляем шум
                predicted_noise = model(xt, text_embs, t, attention_mask)
                loss_val = criterion(predicted_noise, torch.randn_like(xt))
                loss = loss_val

                i += 1
                print(f"Процентов {(i / len(val_loader)) * 100}")
        return loss

    def test_model(self, e_model: EncapsulatedModel, e_loader: EncapsulatedDataloaders):
        print("Тестирование")
        model = e_model.model
        device = e_model.device
        criterion = e_model.criterion
        test_loader = e_loader.test

        model.eval()

        beta = torch.linspace(0.0001, 0.02, T)  # Линейно возрастающие b_t
        alpha = 1 - beta  # a_t
        alphas_bar = torch.cumprod(alpha, dim=0).to(device)  # Накапливаемый коэффициент a_t (T,)

        test_loss = 0.0
        i = 0
        with torch.no_grad():
            for images, text_embs, attention_mask in test_loader:
                images, text_embs, attention_mask = images.to(device), text_embs.to(
                    device), attention_mask.to(device)
                t = torch.randint(0, T, (BATCH_SIZE,), device=device)  # случайные шаги t
                xt = self.forward_diffusion(images, t, alphas_bar).to(device)  # добавляем шум
                predicted_noise = model(xt, text_embs, t, attention_mask)
                loss_test = criterion(predicted_noise, torch.randn_like(xt))
                test_loss += loss_test.item()

                i += 1
                print(f"Процентов {(i / len(test_loader)) * 100}")
        # accuracy = 100 * correct / total
        avg_test_loss = test_loss / len(test_loader)
        # print(f'Test Accuracy: {accuracy:.2f}%, Test Loss: {avg_test_loss:.4f}')
        print(f'Test Loss: {avg_test_loss:.4f}')

    def save_my_model_in_middle_train(self, e_model: EncapsulatedModel, model_dir, model_file):
        # Сохранение
        model_filepath = model_dir + model_file
        model = e_model.model
        optimizer = e_model.optimizer
        epoch = e_model.epoch
        lossitem = e_model.last_loss_item
        model.cpu()
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss_item': lossitem,
        }, model_filepath)

    def load_my_model_in_middle_train(self, model_dir, model_file, device):
        # Загрузка
        model_filepath = model_dir + model_file
        checkpoint = torch.load(model_filepath)
        e_model = EncapsulatedModel()

        model = MyUNet(TEXT_EMB_DIM, device).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer = optim.Adam(model.parameters(), lr=LR)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        lossitem = checkpoint['loss_item']

        e_model.device = device
        e_model.model = model
        e_model.optimizer = optimizer
        e_model.epoch = epoch
        e_model.last_loss_item = lossitem

        return e_model

    def save_my_model(self, model, model_dir, model_file):
        # Сохраняем только state_dict модели
        model_filepath = model_dir + model_file
        model.cpu()
        torch.save(model.state_dict(), model_filepath)

    def load_my_model(self, model_dir, model_file, device):
        # Загружаем модель
        model_filepath = model_dir + model_file
        model = MyUNet(TEXT_EMB_DIM, device).to(device)  # Нужно заново создать архитектуру модели
        model.load_state_dict(torch.load(model_filepath))
        model.eval()  # Устанавливаем модель в режим оценки (для тестирования)
        return model


def show_image(tensor_img):
    """ Визуализация тензора изображения """
    img = tensor_img.cpu().detach().numpy().transpose(1, 2, 0)  # Приводим к (H, W, C)
    img = (img - img.min()) / (
            img.max() - img.min())  # Нормализация к [0,1] (matplotlib ждёт данные в формате [0, 1], другие не примет)
    plt.imshow(img)
    plt.axis("off")  # Убираем оси
    plt.show()
