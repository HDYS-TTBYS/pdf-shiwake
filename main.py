import pyocr
import sys
import logging
import yaml
from config import Config
from pypdf import PdfReader
from convert import read_and_convert_pdf_to_image, pil2cv
import cv2
import numpy as np
from utils import list_pdfs, scale_to_width, tilt_correction, delete_straight_line

# format="%(asctime)s - %(levelname)s:%(name)s - %(message)s"を　追加
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s:%(name)s - %(message)s"
)

# logging.debug("debug")
# logging.info("info")
# logging.warning("warning")
# logging.error("error")
# logging.critical("critical")


def main(pdfs, config) -> None:
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
            
        # 膨張
        kernel = np.ones((5, 5), np.uint8)
        erosion_img = cv2.erode(tilt_correction_img, kernel, iterations=1)
        if config.preprocessing.image_debug:
            cv2.imwrite(pdf + ".erosion_img" + ".jpg", erosion_img)
            
        # 直線削除
        red_line_img = delete_straight_line(erosion_img)
        if config.preprocessing.image_debug:
            cv2.imwrite(pdf + ".red_line_img" + ".jpg", red_line_img)



if __name__ == "__main__":
    # tesseractを使用
    tools = pyocr.get_available_tools()
    if len(tools) == 0:
        logging.error("OCRツールが見つかりません。")
        sys.exit(1)

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
    main(pdfs=pdfs, config=config)
