# --- Гиперпараметры для изображений и текстовых эмбеддингов ---
IMG_SIZE = 132  # Размер изображений
# MAX_LEN_TOKENS = 64
MAX_LEN_TOKENS = 50

TEXT_EMB_DIM = 512
# TEXT_EMB_DIM_REDUCED = 256
# TEXT_EMB_DIM_REDUCED = 512 # (если применяем pca или svd, то должно быть равно MAX_LEN_TOKENS)

CURRENT_MODEL_NAME = '132p_new_model.pth'
CURRENT_MODEL_DIR = 'trained/'

# --- Гиперпараметры для нейросети ---
T = 1000  # Количество шагов в диффузии
BATCH_SIZE = 8
# LR = 0.0001
LR = 0.0001
EPOCHS = 20
TIME_EMB_DIM = 256

# --- Гиперпараметры для дебага ---
# OGRANICHITEL = True
OGRANICHITEL = False
N_OGRANICHITEL = 3
