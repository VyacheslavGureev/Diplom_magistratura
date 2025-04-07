import random
import torch
import torch.nn.functional as F  # Стандартный импорт
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPTokenizer, CLIPTextModel
from sklearn.decomposition import PCA, TruncatedSVD
import os
from PIL import Image

from torchvision import transforms, datasets
import models.hyperparams as hyperparams
import models.utils as utils


def collate_fn(batch):
    if len(batch) % hyperparams.BATCH_SIZE != 0:
        additional_batch = random.choices(batch, k=hyperparams.BATCH_SIZE - (len(batch) % hyperparams.BATCH_SIZE))
        batch = batch + additional_batch
    images, text_embs, masks = zip(*batch)  # Разбираем батч по частям
    images = torch.stack(images)  # Объединяем картинки (B, C, H, W)
    text_embs = torch.stack(text_embs)  # Объединяем текстовые эмбеддинги (B, max_length, txt_emb_dim)
    masks = torch.stack(masks)  # Объединяем маски внимания (B, max_length)
    return images, text_embs, masks


class MNISTTextDataset(Dataset):
    def __init__(self, root, train, transform, tokenizer, text_encoder, max_len_tokens):
        self.mnist = datasets.MNIST(root=root, train=train, download=True, transform=transform)
        # Текстовые описания для цифр
        self.TEXT_DESCRIPTIONS = {
            0: ["Это цифра ноль", "0", "ноль", "нуль"],
            1: ["Изображена единица", "1", "единица", "один"],
            2: ["Нарисована цифра два", "2", "два", "двойка"],
            3: ["На картинке цифра три", "3", "три", "тройка"],
            4: ["Четыре, написанное от руки", "4", "четыре", "четвёрка"],
            5: ["Это пятерка", "5", "пять", "пятёрка"],
            6: ["Цифра шесть, нарисованная от руки", "6", "шесть", "шестёрка"],
            7: ["На изображении семерка", "7", "семь", "семёрка"],
            8: ["Нарисована цифра восемь", "8", "восемь", "восьмёрка"],
            9: ["Рукописная девятка", "9", "девять", "девятка"]
        }
        # self.TEXT_DESCRIPTIONS = {
        #     0: "0",
        #     1: "1",
        #     2: "2",
        #     3: "3",
        #     4: "4",
        #     5: "5",
        #     6: "6",
        #     7: "7",
        #     8: "8",
        #     9: "9"
        # }
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.max_len_tokens = max_len_tokens
        self.drop_conditioning_p = 0.15  # Вероятность обнуления conditioning
        # 36 - max len

    def add_label_smoothing(self, text, prob=0.1):
        words = text.split()
        for i in range(len(words)):
            if random.random() < prob:
                words[i] = "[MASK]"  # Заменяем случайное слово на MASK
        return " ".join(words)

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, idx):
        image, label = self.mnist[idx]  # Получаем картинку и её метку (0-9)
        # caption = self.TEXT_DESCRIPTIONS[label][random.randint(0, 3)]  # Берем описание цифры
        caption = self.TEXT_DESCRIPTIONS[label][1]  # Берем описание цифры
        # caption = self.add_label_smoothing(caption, prob=0.20)  # 20% слов заменяются
        tokens = self.tokenizer(
            caption,
            return_tensors="pt",
            padding="max_length",  # Делаем паддинг до max_length
            truncation=True,  # Обрезаем слишком длинные тексты
            max_length=self.max_len_tokens  # Устанавливаем максимальную длину
        )
        attention_mask = tokens['attention_mask'].squeeze(0)  # (max_length,)
        with torch.no_grad():
            text_emb_reduced = self.text_encoder(**tokens).last_hidden_state.squeeze(0)  # (max_length, txt_emb_dim)
            # if random.random() < self.drop_conditioning_p:
            #     text_emb_reduced = torch.zeros_like(text_emb_reduced)  # Обнуляем текстовый эмбеддинг
            #     attention_mask = torch.zeros_like(attention_mask)
        # text_emb_reduced = F.normalize(text_emb_reduced, p=2, dim=-1)
        return image, text_emb_reduced, attention_mask


class ImageTextDataset(Dataset):

    def __init__(self, image_folder, captions_file, transform, tokenizer, text_encoder, max_len_tokens):
        with open(captions_file, "r") as f:
            lines = f.readlines()  # Загружаем подписи
        self.captions = {}
        lines = lines[1:]  # Первая строчка бесполезна
        for line in lines:
            img, caption = line.split(",", 1)
            if img not in self.captions:
                self.captions[img] = []
            self.captions[img].append(caption)
        self.image_folder = image_folder
        self.transform = transform
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.max_len_tokens = max_len_tokens
        self.image_filenames = list(self.captions.keys())  # Файлы картинок
        # 43 - max len

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        # print(idx)
        # Загружаем картинку
        img_path = os.path.join(self.image_folder, self.image_filenames[idx])
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)  # (C, H, W)
        # Загружаем подпись и получаем текстовый эмбеддинг
        # caption = self.captions[self.image_filenames[idx]][random.randint(0, 4)]  # Берём рандомную подпись
        caption = self.captions[self.image_filenames[idx]][0]  # Берём первую подпись

        tokens = self.tokenizer(
            caption,
            return_tensors="pt",
            padding="max_length",  # Делаем паддинг до max_length
            truncation=True,  # Обрезаем слишком длинные тексты
            max_length=self.max_len_tokens  # Устанавливаем максимальную длину
        )
        attention_mask = tokens['attention_mask'].squeeze(0)  # (max_length,)
        with torch.no_grad():
            text_emb_reduced = self.text_encoder(**tokens).last_hidden_state.squeeze(0)  # (max_length, txt_emb_dim)
        return image, text_emb_reduced, attention_mask


# --- Создание датасета для обучения ---
def create_dataset(image_folder, captions_file):
    transform = transforms.Compose([
        transforms.Resize((hyperparams.IMG_SIZE, hyperparams.IMG_SIZE)),  # Приводим к IMG_SIZExIMG_SIZE
        transforms.ToTensor(),  # Переводим в тензор (C, H, W)
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Нормализация
    ])
    tokenizer = utils.load_data_from_file('datas/embedders/tokenizer.pkl')
    text_encoder = utils.load_data_from_file('datas/embedders/text_encoder.pkl')
    dataset = ImageTextDataset(
        image_folder=image_folder,
        captions_file=captions_file,
        transform=transform,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        max_len_tokens=hyperparams.MAX_LEN_TOKENS
    )
    return dataset


# --- Создание датасета для обучения ---
def create_dataset_mnist(folder, train_flag):
    transform = transforms.Compose([
        transforms.Resize((hyperparams.IMG_SIZE, hyperparams.IMG_SIZE)),  # Приводим к IMG_SIZExIMG_SIZE
        transforms.ToTensor(),  # Переводим в тензор (C, H, W)
        transforms.Normalize(mean=[0.5], std=[0.5])  # Нормализация (один канал)
    ])
    tokenizer = utils.load_data_from_file('datas/embedders/tokenizer.pkl')
    text_encoder = utils.load_data_from_file('datas/embedders/text_encoder.pkl')
    dataset = MNISTTextDataset(root=folder,
                               train=train_flag,
                               transform=transform,
                               tokenizer=tokenizer,
                               text_encoder=text_encoder,
                               max_len_tokens=hyperparams.MAX_LEN_TOKENS)
    return dataset


def get_text_emb(text):
    tokenizer = utils.load_data_from_file('datas/embedders/tokenizer.pkl')
    text_encoder = utils.load_data_from_file('datas/embedders/text_encoder.pkl')
    tokens = tokenizer(
        text,
        return_tensors="pt",
        padding="max_length",  # Делаем паддинг до max_length
        truncation=True,  # Обрезаем слишком длинные тексты
        max_length=hyperparams.MAX_LEN_TOKENS  # Устанавливаем максимальную длину
    )
    attention_mask = tokens['attention_mask'].squeeze(0)  # (max_length,)
    with torch.no_grad():
        text_emb_reduced = text_encoder(**tokens).last_hidden_state.squeeze(0)  # (max_length, txt_emb_dim)
    return text_emb_reduced, attention_mask
