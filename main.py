import pyocr
import sys
import logging
import yaml
from config import Config, get_config
from pypdf import PdfReader
from convert import read_and_convert_pdf_to_image, pil2cv
import cv2
from utils import (
    list_pdfs,
    tilt_correction,
    rotation,
    is_include_word,
    create_folder,
    pdf_move,
)
from convert import cv2pil
from pdfminer.high_level import extract_text
import unicodedata
import os
import multiprocessing
import psutil
from myloggin import setup_logger_process, setup_worker_logger


def main(pdf: str, q: multiprocessing.Queue) -> None:
    setup_worker_logger(q)
    # tesseractを使用
    tools = pyocr.get_available_tools()
    if len(tools) == 0:
        logging.critical("OCRツールが見つかりません。")
        sys.exit(1)
    tool = tools[0]

    # 設定
    config = get_config()

    reader = PdfReader(pdf)
    if len(reader.pages) > 1:
        logging.error("1ページ以上のファイルは処理できません。")
        return

    # PDF のソースコードからページのテキストを直接抽出
    text = extract_text(pdf).replace(" ", "").replace("　", "").replace("\n", "")
    normalized = unicodedata.normalize("NFKC", text)
    logging.info(f"{pdf}のソースコードからの抽出結果\n:{normalized}")

    # sorting_ruleのwordが含まれていたら
    is_include = is_include_word(pdf=pdf, normalized=normalized, config=config)
    if is_include:
        # 移動
        pdf_move(pdf, os.path.join(config.read.dest_dir, is_include))
        # 処理を抜ける
        return

    # pdf pathから画像へ
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

    # 切り取り範囲
    if config.read.reading_position == [0, 0, 0, 0]:
        block_img = img_thresh  # 全体
        if config.preprocessing.image_debug:
            cv2.imwrite(pdf + ".block_img" + ".jpg", block_img)
    else:
        block_img = img_thresh[
            config.read.reading_position[1] : config.read.reading_position[3],
            config.read.reading_position[0] : config.read.reading_position[2],
        ]  # 範囲指定
        if config.preprocessing.image_debug:
            cv2.imwrite(pdf + ".block_img" + ".jpg", block_img)

    # 回転角分読み取り
    for r in config.read.rotate:
        result: str = (
            tool.image_to_string(
                cv2pil(rotation(block_img, r)),
                lang=config.read.lang,
                builder=pyocr.builders.TextBuilder(
                    tesseract_layout=config.read.accuracy
                ),
            )
            .replace(" ", "")
            .replace("　", "")
            .replace("\n", "")
        )
        normalized = unicodedata.normalize("NFKC", result)
        logging.info(f"{pdf}のocr角{r}度読み取り結果:\n{normalized}")
        # sorting_ruleのwordが含まれていたら
        is_include = is_include_word(pdf=pdf, normalized=normalized, config=config)
        if is_include:
            # 移動
            pdf_move(pdf, os.path.join(config.read.dest_dir, is_include))
            # 処理を抜ける
            return

    logging.info(f"{pdf}は仕分け条件に該当しませんでした。")


if __name__ == "__main__":
    try:
        multiprocessing.freeze_support()
        # ロギング
        if not os.path.exists("logs"):
            os.makedirs("logs")
        log_q, listener = setup_logger_process()
        listener.start()
        setup_worker_logger(log_q)

        # 設定
        config = get_config()

        # フォルダを作成する
        if not os.path.exists(config.read.dest_dir):
            os.makedirs(config.read.dest_dir)
        create_folder(config)

        pdfs = list_pdfs()
        logging.info(f"{len(pdfs)}件のファイルが見つかりました。")

        args = []
        for pdf in pdfs:
            args.append((pdf, log_q))
        if config.general.multiprocessing:  # マルチプロセス
            with multiprocessing.Pool(psutil.cpu_count(logical=False)) as pool:
                pool.starmap(main, args)
        else:  # 逐次処理
            for pdf in pdfs:
                main(pdf, log_q)
        listener.stop()

        logging.info("処理が完了しました。")

    except Exception as e:
        print(e)
