import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).parent.parent.parent

DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

GENERATED_DIR = ROOT_DIR / "generated"
INDEX_DIR = GENERATED_DIR / "movie_index"

AZURE_CREDENTIALS = {
    'AD_DEPLOYMENT_ID': os.getenv('AD_DEPLOYMENT_ID'),
    'AD_ENGINE': os.getenv('AD_ENGINE'),
    'AD_OPENAI_API_KEY': os.getenv('AD_OPENAI_API_KEY'),
    'AD_OPENAI_API_VERSION': os.getenv('AD_OPENAI_API_VERSION'),
    'AD_OPENAI_API_BASE': os.getenv('AD_OPENAI_API_BASE')
}

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIMENSION = 384
DEVICE = "cuda" if os.getenv("USE_GPU", "false").lower() == "true" else "cpu"

TOP_K_RECOMMENDATIONS = 3
MIN_BUDGET_FILTER = 1_000_000