import glob
import cv2
import numpy as np
from config import Config
import shutil
import os
import logging
import difflib


def list_pdfs() -> str:
    """
    カレントディレクトリのPDFファイルをリストで返す
    """
    files = glob.glob("./*.pdf")
    return files


def scale_to_width(img, h, w, expected_height):
    """高さが指定した値になるように、アスペクト比を固定して、リサイズする。"""
    width = round(w * (expected_height / h))
    dst = cv2.resize(img, dsize=(width, expected_height))

    return dst


def tilt_correction(img):
    """
    スキャン傾き補正
    """
    height, width = img.shape[:2]
    center = (width / 2, height / 2)  # 中心座標設定

    # モノクロ・グレースケール画像へ変換
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 二値化
    ret, img = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)

    contours, hierarchy = cv2.findContours(
        img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )  # 領域検出

    r = []
    for i in contours:
        area = cv2.contourArea(i)  # 各領域の面積
        r.append(area)
    max_value = max(r)
    max_index = r.index(max_value)
    rect = cv2.minAreaRect(contours[max_index])  # 回転外接矩形の算出
    r = 0.0
    if rect[2] < 80:
        r = 90 + rect[2]
    else:
        r = rect[2]
    angle = 180 + 90 + r  # 回転角を設定

    trans = cv2.getRotationMatrix2D(center, angle, scale=1)  # 変換行列の算出

    img = cv2.warpAffine(img, trans, (width, height), flags=cv2.INTER_CUBIC)  # 元画像を回転

    return img


def rotation(img, angle: int):
    """
    画像を回転する
    """
    h, w = img.shape[:2]
    center = (w / 2, h / 2)  # 中心座標設定
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale=1)  # 変換行列の算出
    # 平行移動を加える (rotation + translation)
    affine_matrix = rotation_matrix.copy()
    affine_matrix[0][2] = affine_matrix[0][2] - w / 2 + h / 2
    affine_matrix[1][2] = affine_matrix[1][2] - h / 2 + w / 2
    rotate_img = cv2.warpAffine(
        img, affine_matrix, (h, w), flags=cv2.INTER_CUBIC
    )  # 元画像を回転

    return rotate_img


def delete_straight_line(img, width: int, min_line_length=500):
    """
    直線削除
    """
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    except:
        gray = img
    gray_reverse = cv2.bitwise_not(gray)
    lines = cv2.HoughLinesP(
        gray_reverse,
        rho=1,
        theta=np.pi / 360,
        threshold=80,
        minLineLength=min_line_length,
        maxLineGap=5,
    )

    for line in lines:
        x1, y1, x2, y2 = line[0]
        # 赤線を引く
        # red_line_img = cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), int(width*0.002) )
        # 線を消す(白で線を引く)
        no_lines_img = cv2.line(
            img, (x1, y1), (x2, y2), (255, 255, 255), int(width * 0.002)
        )

    return no_lines_img


def is_include_word(normalized: str, config: Config):
    """
    sorting_ruleのwordが含まれているか
    """
    for rule in config.sorting_rules:
        if rule.word in normalized:
            return rule.dist_dir
    return False


def is_include_word_diff(normalized: str, config: Config, threshold=70):
    """
    sorting_ruleのwordが含まれているか(類似度が一定以上)
    """
    try:
        for rule in config.sorting_rules:
            p = 0
            while p <= len(normalized) - len(rule.word):
                s = difflib.SequenceMatcher(
                    None, rule.word, normalized[p : p + len(rule.word)]
                ).ratio()
                if float(s) >= (threshold / 100):
                    return (
                        rule.dist_dir,
                        float(s),
                        rule.word,
                        normalized[p : p + len(rule.word)],
                    )
                p += 1
    except:
        return False, False, False, False
    return False, False, False, False


def pdf_move(pdf: str, dist_dir: str):
    shutil.move(pdf, dist_dir)
    logging.info(f"{pdf}を{dist_dir}に移動しました。")


def create_folder(config: Config):
    # フォルダを作成する
    for dir in config.sorting_rules:
        if not os.path.exists(os.path.join(config.general.dist_dir, dir.dist_dir)):
            os.mkdir(os.path.join(config.general.dist_dir, dir.dist_dir))
            logging.info(f"{dir.dist_dir}フォルダを作成しました。")
