from pathlib import Path
from basicsr.test import test_pipeline

if __name__ == '__main__':
    test_pipeline(str(Path(__file__).resolve().parent))
