from pathlib import Path
from basicsr.train import train_pipeline

if __name__ == '__main__':
    train_pipeline(str(Path(__file__).resolve().parent))
