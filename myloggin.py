import logging
import logging.handlers
import multiprocessing
import sys
from pythonjsonlogger import jsonlogger


def setup_logger_process():
    log_queue = multiprocessing.Manager().Queue()
    # ハンドラの生成と設定
    #   ファイル mptest4.log と標準エラー stderr の2つへ出力する
    file_handler = logging.handlers.RotatingFileHandler(
        filename="./logs/pdf-sorting.log", encoding="utf-8", maxBytes=256000, backupCount=99
    )
    file_formatter = logging.Formatter(
        "%(asctime)s %(processName)-5s %(name)-5s %(levelname)-4s %(message)s"
    )
    file_handler.setFormatter(file_formatter)

    console_handler = logging.StreamHandler(sys.stderr)
    console_formatter = logging.Formatter(
        "%(asctime)s %(processName)-5s %(name)-5s %(levelname)-4s %(message)s"
    )
    console_handler.setFormatter(console_formatter)

    # QueueListenerを、QueueとHandlerを指定して生成
    listener = logging.handlers.QueueListener(log_queue, console_handler, file_handler)
    # 待受の開始
    return log_queue, listener


def setup_worker_logger(log_queue: multiprocessing.Queue):
    # ログの取得・Queueの登録
    logger = logging.getLogger()  # loggerの取得
    handler = logging.handlers.QueueHandler(log_queue)  # QueueHandlerの生成
    logger.addHandler(handler)  # loggerへhandlerを登録
    logger.setLevel(logging.INFO)  # loggerへレベル設定（任意）
