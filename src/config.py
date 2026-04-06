# src/config.py

import os
from pathlib import Path
from dotenv import load_dotenv

# ===== Detect environment =====
if "COLAB_GPU" in os.environ or Path("/content").exists():
    ENV = "colab"
elif "AWS_EXECUTION_ENV" in os.environ or Path("/home/ubuntu").exists():
    ENV = "aws"
else:
    home = Path.home()
    if "yunfan" in str(home).lower():
        ENV = "local_yunfan"
    else:
        ENV = "other"

# ===== Resolve project root =====
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ===== Select env file =====
if ENV == "colab":
    env_path = PROJECT_ROOT / ".env.colab"
elif ENV == "aws":
    env_path = PROJECT_ROOT / ".env.aws"
else:
    env_path = PROJECT_ROOT / ".env.local.yunfan"

# ===== Load env =====
if env_path.exists():
    load_dotenv(env_path)

# ===== Data =====
RAW_DIR = Path(os.getenv("RAW_DIR", PROJECT_ROOT / "data" / "raw"))
PROCESSED_DIR = Path(os.getenv("PROCESSED_DIR", PROJECT_ROOT / "data" / "processed"))

# ===== Outputs =====
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", PROJECT_ROOT / "outputs"))