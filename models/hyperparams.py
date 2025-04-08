# --- Изменение этого могут затронуть модель, менеджер и даталодер!!! ---
IMG_SIZE = 32  # Размер изображений
BATCH_SIZE = 16
LR = 0.0001

# UNET_CONFIG = {'DOWN':
#               [{'in_C': 64, 'out_C': 128, 'SA': False},
#                {'in_C': 128, 'out_C': 256, 'SA': True},
#                {'in_C': 256, 'out_C': 512, 'SA': False}],
#           'BOTTLENECK': [{'in_C': 512, 'out_C': 512}],
#           'UP': [{'in_C': 512, 'out_C': 256, 'sc_C': 256, 'SA': False, 'CA': False},
#                  {'in_C': 256 + 256, 'out_C': 256, 'sc_C': 128, 'SA': True, 'CA': True}]}

# UNET_CONFIG = {'DOWN':
#               [{'in_C': 4, 'out_C': 8, 'SA': False},
#                {'in_C': 8, 'out_C': 16, 'SA': True},
#                {'in_C': 16, 'out_C': 32, 'SA': False}],
#           'BOTTLENECK': [{'in_C': 32, 'out_C': 32}],
#           'UP': [{'in_C': 32, 'out_C': 16, 'sc_C': 16, 'SA': False, 'CA': False},
#                  {'in_C': 16 + 16, 'out_C': 16, 'sc_C': 8, 'SA': True, 'CA': True}]}

UNET_CONFIG = {'DOWN':
                   [{'in_C': 1, 'out_C': 3, 'SA': False},
                    {'in_C': 3, 'out_C': 4, 'SA': False},
                    {'in_C': 4, 'out_C': 8, 'SA': True},
                    {'in_C': 8, 'out_C': 16, 'SA': False}],
               'BOTTLENECK': [{'in_C': 16, 'out_C': 16}],
               'UP': [{'in_C': 16, 'out_C': 8, 'sc_C': 8, 'SA': False, 'CA': False},
                      {'in_C': 8 + 8, 'out_C': 8, 'sc_C': 4, 'SA': True, 'CA': True},
                      {'in_C': 8 + 4, 'out_C': 6, 'sc_C': 3, 'SA': False, 'CA': False}
                      ]}

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
# OGRANICHITEL = True
OGRANICHITEL = False
N_OGRANICHITEL = 1550
