import pyocr
import sys
import logging
import yaml
from config import Config
from pypdf import PdfReader
from convert import read_and_convert_pdf_to_image, pil2cv
import cv2
from utils import list_pdfs, tilt_correction, rotation, is_include_if_move
from convert import cv2pil
import os
from pdfminer.high_level import extract_text
import unicodedata
import time


def main(pdf: str, config: Config, tool) -> None:
    reader = PdfReader(pdf)
    if len(reader.pages) > 1:
        logging.error("1ページ以上のファイルは処理できません。")
        return

    # PDF のソースコードからページのテキストを直接抽出
    text = extract_text(pdf)
    text2 = text.replace(" ", "").replace("　", "").replace("\n", "")
    normalized = unicodedata.normalize("NFKC", text2)
    logging.info(f"{pdf}のソースコードからの抽出結果\n:{normalized}")

    # sorting_ruleのwordが含まれていたら移動
    is_include = is_include_if_move(pdf=pdf, normalized=normalized, config=config)
    # 処理を抜ける
    if is_include:
        return

    image = read_and_convert_pdf_to_image(pdf_path=pdf, dpi=config.preprocessing.dpi)
    img = pil2cv(image)

    # 傾き補正
    tilt_correction_img = tilt_correction(img)
    logging.info(f"{pdf}の傾きの傾き補正が終了")
    if config.preprocessing.image_debug:
        cv2.imwrite(pdf + ".tilt_correction_img" + ".jpg", tilt_correction_img)

    # モノクロ・グレースケール画像へ変換（2値化前の画像処理）
    im_gray = cv2.cvtColor(tilt_correction_img, cv2.COLOR_BGR2GRAY)
    # 二値化
    ret, img_thresh = cv2.threshold(im_gray, 0, 255, cv2.THRESH_OTSU)

    # ＯＣＲ実行
    builder = pyocr.builders.TextBuilder(tesseract_layout=config.read.accuracy)

    # 切り取り範囲
    if config.read.reading_position == [0, 0, 0, 0]:
        block_img = img_thresh  # 全体
    else:
        block_img = img_thresh[
            config.read.reading_position[1] : config.read.reading_position[3],
            config.read.reading_position[0] : config.read.reading_position[2],
        ]  # 範囲指定
        if config.preprocessing.image_debug:
            cv2.imwrite(pdf + ".block_img" + ".jpg", block_img)

    result: str = tool.image_to_string(
        cv2pil(block_img), lang=config.read.lang, builder=builder
    )
    result2 = result.replace(" ", "").replace("　", "").replace("\n", "")
    normalized = unicodedata.normalize("NFKC", result2)
    logging.info(f"{pdf}のocr正方向読み取り結果:\n{normalized}")
    # sorting_ruleのwordが含まれていたら移動
    is_include = is_include_if_move(pdf=pdf, normalized=normalized, config=config)
    # 処理を抜ける
    if is_include:
        return

    # 時計回りに90度回転
    result: str = tool.image_to_string(
        cv2pil(rotation(block_img)), lang=config.read.lang, builder=builder
    )
    result2 = result.replace(" ", "").replace("　", "").replace("\n", "")
    normalized = unicodedata.normalize("NFKC", result2)
    logging.info(f"{pdf}のocr時計回りに90度回転読み取り結果:\n{normalized}")
    # sorting_ruleのwordが含まれていたら移動
    is_include = is_include_if_move(pdf=pdf, normalized=normalized, config=config)
    # 処理を抜ける
    if is_include:
        return

    print(f"{pdf}は仕分け条件に該当しませんでした。")


if __name__ == "__main__":
    # ロギング
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s:%(name)s - %(message)s",
        filename="pdf-sorting.log",
        encoding="utf-8",
    )

    # tesseractを使用
    tools = pyocr.get_available_tools()
    if len(tools) == 0:
        logging.error("OCRツールが見つかりません。")
        sys.exit(1)
    tool = tools[0]

    # 設定
    try:
        with open("pdf-sorting.yml", "r", encoding="utf-8") as f:
            c = yaml.safe_load(f)
            config = Config(**c)
    except:
        logging.error("設定ファイルが見つかりません。")
        sys.exit(1)

    # フォルダを作成する
    for dir in config.sorting_rules:
        if not os.path.exists(dir.dest_dir):
            os.mkdir(dir.dest_dir)
            logging.info(f"{dir.dest_dir}フォルダを作成しました。")

    pdfs = list_pdfs()
    logging.info(f"{len(pdfs)}件のファイルが見つかりました。")
    print(f"{time.strftime('%Y/%m/%d %H:%M:%S')}:処理を開始します。")
    for pdf in pdfs:
        main(pdf=pdf, config=config, tool=tool)
    logging.info("処理が完了しました。")
    print(f"{time.strftime('%Y/%m/%d %H:%M:%S')}:処理が完了しました。")
