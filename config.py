from pydantic import BaseModel
from typing import List
import yaml
import logging
import sys


class General(BaseModel):
    multiprocessing: bool


class Preprocessing(BaseModel):
    image_debug: bool
    dpi: int
    min_line_length: int


class Read(BaseModel):
    reading_position: List[int]
    rotate: List[int]
    lang: str
    accuracy: int
    dist_dir: str


class SortingRules(BaseModel):
    word: str
    dist_dir: str


class Config(BaseModel):
    general: General
    preprocessing: Preprocessing
    read: Read
    sorting_rules: List[SortingRules]


def get_config():
    try:
        with open("config.yaml", "r", encoding="utf-8") as f:
            c = yaml.safe_load(f)
            config = Config(**c)
            return config
    except:
        logging.critical("設定ファイルが見つかりません。")
        sys.exit(1)
