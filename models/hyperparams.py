# --- Изменение этого могут затронуть модель, менеджер и даталодер!!! ---
IMG_SIZE = 32  # Размер изображений
BATCH_SIZE = 16
CHANNELS = 1
LR = 0.0001

T = 1000  # Количество шагов в диффузии

# --- Почти никогда не меняется ---
TEXT_EMB_DIM = 512
TIME_EMB_DIM = 256
MAX_LEN_TOKENS = 50
# TEXT_EMB_DIM_REDUCED = 512 # (если применяем pca или svd, то должно быть равно MAX_LEN_TOKENS)

# --- Можно менять почти безболезненно ---
EPOCHS = 1

MODELS_DIR = 'trained/models/'
MODEL_NAME_DDPM = '32p_mnist.pth'
MODEL_NAME_ADAPT = '32p_mnist_adapt.pth'

CONFIGS_DIR = 'trained/configs/'
MODEL_CONFIG_DDPM = '32p_mnist_config_ddpm.json'
MODEL_CONFIG_ADAPT = '32p_mnist_config_adapt.json'

E_LOADERS_DIR = 'trained/e_loaders/'
E_LOADER_DDPM = 'e_loader.pkl'
E_LOADER_ADAPT = 'e_loader_adapt.pkl'

# --- Гиперпараметры для дебага ---
# OGRANICHITEL = True
OGRANICHITEL = False
N_OGRANICHITEL = 500

# VIZ_STEP = True