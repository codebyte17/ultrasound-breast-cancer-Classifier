import sys
from pathlib import Path

# Add project root to python path
ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))

# Now you can import anything from src
from src.train import main

if __name__ == "__main__":
    main()
