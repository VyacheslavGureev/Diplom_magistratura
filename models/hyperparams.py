# --- Гиперпараметры для изображений и текстовых эмбеддингов ---
# IMG_SIZE = 256  # Размер изображений
IMG_SIZE = 128  # Размер изображений
# MAX_LEN_TOKENS = 64
MAX_LEN_TOKENS = 64
# TEXT_EMB_DIM_REDUCED = 256
TEXT_EMB_DIM_REDUCED = 64
CURRENT_MODEL_NAME = 'tst.pth'
CURRENT_MODEL_DIR = 'trained/'

# --- Гиперпараметры для нейросети ---
# IMG_SIZE = 32  # Размер изображений
T = 1000  # Количество шагов в диффузии
BATCH_SIZE = 8
# LR = 0.0001
LR = 0.0001
# TEXT_EMB_DIM = 512
EPOCHS = 10