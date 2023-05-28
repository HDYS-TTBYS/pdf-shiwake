from pydantic import BaseModel
from typing import List
import yaml
import logging
import sys

class Preprocessing(BaseModel):
    image_debug: bool
    dpi: int
    min_line_length: int


class Read(BaseModel):
    reading_position: List[int]
    lang: str
    accuracy: int
    available_chars: str


class SortingRules(BaseModel):
    word: str
    dest_dir: str


class Config(BaseModel):
    preprocessing: Preprocessing
    read: Read
    sorting_rules: List[SortingRules]

def get_config():
    try:
        with open("pdf-sorting.yml", "r", encoding="utf-8") as f:
            c = yaml.safe_load(f)
            config = Config(**c)
            return config
    except:
        logging.error("設定ファイルが見つかりません。")
        sys.exit(1)