# --- Изменение этого могут затронуть модель, менеджер и даталодер!!! ---
IMG_SIZE = 32  # Размер изображений
BATCH_SIZE = 16
LR = 0.00005

T = 1000  # Количество шагов в диффузии

# --- Почти никогда не меняется ---
TEXT_EMB_DIM = 512
TIME_EMB_DIM = 256
MAX_LEN_TOKENS = 50
# MAX_LEN_TOKENS = 32
# TEXT_EMB_DIM_REDUCED = 512 # (если применяем pca или svd, то должно быть равно MAX_LEN_TOKENS)

# --- Можно менять почти безболезненно ---
EPOCHS = 1
CURRENT_MODEL_NAME = '32p_mnist.pth'
CURRENT_MODEL_DIR = 'trained/'

# --- Гиперпараметры для дебага ---
OGRANICHITEL = True
# OGRANICHITEL = False
N_OGRANICHITEL = 1000