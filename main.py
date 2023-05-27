import pyocr
import sys
import logging
import yaml
from config import Config
from pypdf import PdfReader
from convert import read_and_convert_pdf_to_image, pil2cv
import cv2
from utils import list_pdfs, tilt_correction, delete_straight_line
from convert import cv2pil
import numpy as np

# format="%(asctime)s - %(levelname)s:%(name)s - %(message)s"を　追加
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s:%(name)s - %(message)s",
    filename="pdf-sorting.log",
    encoding="utf-8",
)

# logging.debug("debug")
# logging.info("info")
# logging.warning("warning")
# logging.error("error")
# logging.critical("critical")


def main(pdfs: str, config: Config, tool) -> None:
    for pdf in pdfs:
        reader = PdfReader(pdf)
        if len(reader.pages) > 1:
            logging.error("1ページ以上のファイルは処理できません。")
            return

        image = read_and_convert_pdf_to_image(
            pdf_path=pdf, dpi=config.preprocessing.dpi
        )
        img = pil2cv(image)
        height, width = img.shape[:2]

        # 回転
        tilt_correction_img = tilt_correction(img)
        logging.info(f"{pdf}の傾きの傾き補正が終了")

        # 直線削除
        del_line_img = delete_straight_line(
            img=tilt_correction_img,
            width=width,
            min_line_length=width * (config.preprocessing.min_line_length / 100),
        )
        logging.info(f"{pdf}の直線除去処理が終了")
        if config.preprocessing.image_debug:
            cv2.imwrite(pdf + ".red_line_img" + ".jpg", del_line_img)

        # モノクロ・グレースケール画像へ変換（2値化前の画像処理）
        im_gray = cv2.cvtColor(del_line_img, cv2.COLOR_BGR2GRAY)
        # 二値化
        ret, img_thresh = cv2.threshold(im_gray, 0, 255, cv2.THRESH_OTSU)

        # ＯＣＲ実行
        builder = pyocr.builders.TextBuilder(tesseract_layout=config.read.accuracy)
        if config.read.reading_position == [0, 0, 0, 0]:  # 全体
            result: str = tool.image_to_string(
                cv2pil(img_thresh), lang=config.read.lang, builder=builder
            )
            r = result.replace(" ", "").replace("　", "")
            logging.info(f"{pdf}の読み取り結果:\n{r}")


if __name__ == "__main__":
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
            # print(config)
    except:
        logging.error("設定ファイルが見つかりません。")
        sys.exit(1)

    pdfs = list_pdfs()
    logging.info(" ".join(pdfs) + " のファイルが見つかりました。")
    main(pdfs=pdfs, config=config, tool=tool)
