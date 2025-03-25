import random

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPTokenizer, CLIPTextModel
from sklearn.decomposition import PCA, TruncatedSVD
import os
from PIL import Image

from torchvision import transforms, datasets
import models.hyperparams as hyperparams




# Текстовые описания для цифр
TEXT_DESCRIPTIONS = {
    0: "Это цифра ноль",
    1: "Изображена единица",
    2: "Нарисована цифра два",
    3: "На картинке цифра три",
    4: "Четыре, написанное от руки",
    5: "Это пятерка",
    6: "Цифра шесть, нарисованная от руки",
    7: "На изображении семерка",
    8: "Нарисована цифра восемь",
    9: "Рукописная девятка"
}


class MNISTTextDataset(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.mnist = datasets.MNIST(root=root, train=train, download=True, transform=transform)

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, idx):
        image, label = self.mnist[idx]  # Получаем картинку и её метку (0-9)
        text = TEXT_DESCRIPTIONS[label]  # Берем описание цифры
        return image, text







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

        # self.text_reducer = TextEmbeddingReducer(512, hyperparams.TEXT_EMB_DIM_REDUCED)
        # self.text_reducer = TextEmbeddingReducer(512, hyperparams.TEXT_EMB_DIM_REDUCED)

        # self.maxxxx = 0
        # for i in self.image_filenames:
        #     for c in self.captions[i]:
        #         tokens = self.tokenizer(c, return_tensors="pt", padding=True, truncation=True)
        #         if self.maxxxx < tokens.data['input_ids'].shape[1]:
        #             self.maxxxx = tokens.data['input_ids'].shape[1]
        # print(self.maxxxx)
        # print('maxxx')

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


def get_text_emb(text):
    # text_reducer = TextEmbeddingReducer(512, hyperparams.TEXT_EMB_DIM_REDUCED)

    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
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


# --- Создание датасета для обучения ---
def create_dataset(image_folder, captions_file):
    transform = transforms.Compose([
        transforms.Resize((hyperparams.IMG_SIZE, hyperparams.IMG_SIZE)),  # Приводим к IMG_SIZExIMG_SIZE
        transforms.ToTensor(),  # Переводим в тензор (C, H, W)
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Нормализация
    ])

    # Загружаем CLIP
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")

    # Создаём датасет
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
def create_dataset_mnist():
    transform = transforms.Compose([
        transforms.Resize((hyperparams.IMG_SIZE, hyperparams.IMG_SIZE)),  # Приводим к IMG_SIZExIMG_SIZE
        transforms.ToTensor(),  # Переводим в тензор (C, H, W)
        transforms.Normalize(mean=[0.5], std=[0.5])  # Нормализация
    ])

    # Загружаем CLIP
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")

    # # Создаём датасет
    # dataset = MNISTTextDataset(
    #     transform=transform,
    #     tokenizer=tokenizer,
    #     text_encoder=text_encoder,
    #     max_len_tokens=hyperparams.MAX_LEN_TOKENS
    # )

    dataset = MNISTTextDataset(root="./data", train=True, transform=transform)

    # Проверим
    image, text = dataset[0]
    print(text)  # Например: "Это цифра ноль"



    return dataset




# def reduce_embedding_linear(text_emb, reduced_dim):
#     text_reducer = TextEmbeddingReducer(512, reduced_dim)
#     text_emb_reduced = text_reducer(text_emb)  # (B, tokens, 256)
#     return text_emb_reduced


# def reduce_embedding_svd(text_emb, reduced_dim):
#     """
#     Уменьшает размерность текстового эмбеддинга с помощью SVD.
#
#     Параметры:
#     - text_emb: torch.Tensor, размерность (tokens, original_dim)
#     - reduced_dim: int, желаемая размерность (reduced_dim)
#
#     Возвращает:
#     - reduced_emb: torch.Tensor, размерность (tokens, reduced_dim)
#     """
#     # SVD-разложение
#     U, S, Vt = torch.linalg.svd(text_emb, full_matrices=False)
#
#     # Оставляем только первые reduced_dim компонент
#     reduced_emb = U[:, :reduced_dim] @ torch.diag(S[:reduced_dim])
#
#     return reduced_emb
#
#
# def reduce_embedding_pca(text_emb, reduced_dim):
#     tokens, original_dim = text_emb.shape
#     pca = PCA(n_components=reduced_dim)
#     text_emb_reshaped = text_emb.view(-1, original_dim).cpu().numpy()
#     reduced_emb = pca.fit_transform(text_emb_reshaped)
#     new_tensor = torch.tensor(reduced_emb, dtype=text_emb.dtype, device=text_emb.device).view(tokens, reduced_dim)
#     return new_tensor
