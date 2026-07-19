import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./recipes.db")
MODEL_DIR = os.getenv("MODEL_DIR", "./models")
DATA_DIR = os.getenv("DATA_DIR", "./recipe_re_dataset")
NER_MODEL_NAME = os.getenv("NER_MODEL_NAME", "bert-base-chinese")
RE_MODEL_NAME = os.getenv("RE_MODEL_NAME", "bert-base-chinese")
MAX_SEQ_LENGTH = int(os.getenv("MAX_SEQ_LENGTH", 128))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 16))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", 2e-5))
NUM_EPOCHS = int(os.getenv("NUM_EPOCHS", 10))

ENTITY_LABELS = ["O", "B-食材", "I-食材", "B-调料", "I-调料", "B-份量", "I-份量", "B-处理方式", "I-处理方式", "B-动作", "I-动作", "B-时间", "I-时间"]

RELATION_LABELS = ["无关系", "食材-份量", "食材-处理方式", "调料-份量", "动作-对象", "动作-时间"]

ENTITY_TYPE_MAP = {
    "食材": "食材",
    "调料": "调料",
    "份量": "份量",
    "处理方式": "处理方式",
    "动作": "动作",
    "时间": "时间"
}

RELATION_TYPE_MAP = {
    "食材-份量": "食材-份量",
    "食材-处理方式": "食材-处理方式",
    "调料-份量": "调料-份量",
    "动作-对象": "动作-对象",
    "动作-时间": "动作-时间"
}