import json
import random

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPTokenizer, CLIPTextModel, CLIPProcessor
import json
import os
from PIL import Image

from torchvision import transforms

# IMG_SIZE = 256  # Размер изображений
IMG_SIZE = 128  # Размер изображений


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

        # self.maxxxx = 0
        # for i in self.image_filenames:
        #     for c in self.captions[i]:
        #         tokens = self.tokenizer(c, return_tensors="pt", padding=True, truncation=True)
        #         if self.maxxxx < tokens.data['input_ids'].shape[1]:
        #             self.maxxxx = tokens.data['input_ids'].shape[1]
        #         print(tokens.data['input_ids'].shape[1])

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        # Загружаем картинку
        img_path = os.path.join(self.image_folder, self.image_filenames[idx])
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)  # (C, H, W)
        # Загружаем подпись и получаем текстовый эмбеддинг
        caption = self.captions[self.image_filenames[idx]][random.randint(0, 4)]  # Берём первую подпись
        tokens = self.tokenizer(
            caption,
            return_tensors="pt",
            padding="max_length",  # Делаем паддинг до max_length
            truncation=True,  # Обрезаем слишком длинные тексты
            max_length=self.max_len_tokens  # Устанавливаем максимальную длину
        )
        attention_mask = tokens['attention_mask'].squeeze(0)  # (max_length,)
        with torch.no_grad():
            text_emb = self.text_encoder(**tokens).last_hidden_state.squeeze(0)  # (max_length, txt_emb_dim)
        return image, text_emb, attention_mask


# --- Создание датасета для обучения ---
def create_dataset(image_folder, captions_file):
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),  # Приводим к IMG_SIZExIMG_SIZE
        transforms.ToTensor(),  # Переводим в тензор (C, H, W)
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Нормализация
    ])

    # Загружаем CLIP
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
    max_len_tokens = 64

    # Создаём датасет
    dataset = ImageTextDataset(
        image_folder=image_folder,
        captions_file=captions_file,
        transform=transform,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        max_len_tokens=max_len_tokens
    )

    # Загружаем в `DataLoader`
    # dataloader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
    return dataset

    # Пример батча
    # images, text_embs, masks = next(iter(dataloader))
    # print('[eq', dataset.maxxxx)

    # return images, text_embs, masks
