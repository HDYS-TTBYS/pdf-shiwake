from pydantic import BaseModel
from typing import List


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
