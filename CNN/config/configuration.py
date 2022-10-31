import torch.cuda
from typing_extensions import Final

INIT_LR: Final = 1e-3
BATCH_SIZE: Final = 64
EPOCHS: Final = 10
LOG_INTERVAL: Final = 100
TRAIN_SPLIT: Final = 0.75
VAL_SPLIT: Final = 1 - TRAIN_SPLIT
OUTPUT_DIR: str = "./output"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")