import pyocr
import sys
import logging
from config import get_config
from pypdf import PdfReader
from convert import read_and_convert_pdf_to_image, pil2cv
import cv2
from utils import (
    list_pdfs,
    tilt_correction,
    rotation,
    create_folder,
    pdf_move,
    is_include_word_diff,
)
from convert import cv2pil
from pdfminer.high_level import extract_text
import unicodedata
import os
import multiprocessing
import psutil
from myloggin import setup_logger_process, setup_worker_logger
import time


def main(pdf: str, q: multiprocessing.Queue, lock) -> None:
    setup_worker_logger(q)
    # tesseractを使用
    tools = pyocr.get_available_tools()
    if len(tools) == 0:
        logging.critical("OCRツールが見つかりません。")
        sys.exit(1)
    tool = tools[0]

    # 設定
    config = get_config(lock)

    reader = PdfReader(pdf)
    if len(reader.pages) > 1:
        logging.error("1ページ以上のファイルは処理できません。")
        return

    # PDF のソースコードからページのテキストを直接抽出
    text = extract_text(pdf).replace(" ", "").replace("　", "").replace("\n", "")
    normalized = unicodedata.normalize("NFKC", text)
    n = normalized
    try:
        if not config.general.full_log:
            n = normalized[0:70] + "..."
    except:
        n = normalized
    logging.info(f"{pdf}のソースコードからの抽出結果\n{n}\n")

    # sorting_ruleのwordが含まれていたら
    dist_dir, match_rate, match_str = is_include_word_diff(
        normalized=normalized, config=config, threshold=config.general.threshold
    )
    if dist_dir:
        logging.info(f"{pdf}の類似度、一致率:{match_rate}、一致結果:{dist_dir}|{match_str}")
        # 移動
        pdf_move(pdf, os.path.join(config.general.dist_dir, dist_dir))
        # 処理を抜ける
        return

    # pdf pathから画像へ
    image = read_and_convert_pdf_to_image(pdf_path=pdf, dpi=config.preprocessing.dpi)
    img = pil2cv(image)

    # 傾き補正
    img = tilt_correction(img)
    logging.info(f"{pdf}の傾きの傾き補正が終了")
    if config.preprocessing.image_debug:
        cv2.imwrite(pdf + ".tilt_correction" + ".jpg", img)

    # 切り取り範囲
    if config.read.reading_position != [0, 0, 0, 0]:
        img = img[
            config.read.reading_position[1] : config.read.reading_position[3],
            config.read.reading_position[0] : config.read.reading_position[2],
        ]
        if config.preprocessing.image_debug:
            cv2.imwrite(pdf + ".block" + ".jpg", img)

    # 回転角分読み取り
    builder = pyocr.builders.TextBuilder(tesseract_layout=config.read.accuracy)
    for r in config.read.rotate:
        result: str = (
            tool.image_to_string(
                cv2pil(rotation(img, r)), lang=config.read.lang, builder=builder
            )
            .replace(" ", "")
            .replace("　", "")
            .replace("\n", "")
        )
        normalized = unicodedata.normalize("NFKC", result)
        n = normalized
        try:
            if not config.general.full_log:
                n = normalized[0:70] + "..."
        except:
            n = normalized
        logging.info(f"{pdf}のocr角{r}度読み取り結果\n{n}\n")

        # sorting_ruleのwordが含まれていたら
        dist_dir, match_rate, match_str = is_include_word_diff(
            normalized=normalized, config=config, threshold=config.general.threshold
        )
        if dist_dir:
            logging.info(f"{pdf}の類似度、一致率:{match_rate}、一致結果:{dist_dir}|{match_str}")
            # 移動
            pdf_move(pdf, os.path.join(config.general.dist_dir, dist_dir))
            # 処理を抜ける
            return

    # 該当しなかったらdist_dir直下へ移動
    pdf_move(pdf, os.path.join(config.general.dist_dir))
    logging.info(f"{pdf}は仕分け条件に該当しませんでした。")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    # ロギング
    if not os.path.exists("logs"):
        os.makedirs("logs")
    log_q, listener = setup_logger_process()
    listener.start()
    setup_worker_logger(log_q)

    # 設定
    lock = multiprocessing.Manager().Lock()
    config = get_config(lock)

    # フォルダを作成する
    if not os.path.exists(config.general.dist_dir):
        os.makedirs(config.general.dist_dir)
    create_folder(config)

    def do():
        pdfs = list_pdfs()
        logging.info(f"{len(pdfs)}件のファイルが見つかりました。")
        args = []
        for pdf in pdfs:
            args.append((pdf, log_q, lock))
        if config.general.multiprocessing:  # マルチプロセス
            with multiprocessing.Pool(psutil.cpu_count(logical=False)) as pool:
                pool.starmap(main, args)
        else:  # 逐次処理
            for pdf in pdfs:
                main(pdf, log_q, lock)
        if config.general.watch:
            time.sleep(3)
            do()

    try:
        do()
    except KeyboardInterrupt:
        logging.info("処理が中断されました。")

    listener.stop()

    logging.info("処理が完了しました。")
