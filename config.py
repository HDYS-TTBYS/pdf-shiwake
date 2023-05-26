from pydantic import BaseModel
from typing import List


class Preprocessing(BaseModel):
    image_debug: bool
    dpi: int


class Read(BaseModel):
    top_x: int
    top_y: int
    buttom_x: int
    buttom_y: int
    lang: str
    available_chars: str


class SortingRules(BaseModel):
    word: str
    dest_dir: str


class Config(BaseModel):
    preprocessing: Preprocessing
    read: Read
    sorting_rules: List[SortingRules]
