import cv2
import time
import config
import imutils
import sqlite3
import logging
import traceback
import numpy as np
import pandas as pd
from pathlib import Path
from logging.handlers import RotatingFileHandler


def get_logger(log_path, max_bytes=5000000, backup_count=5):
    """
    Default logger will retains the maximum log file size of 5MB and number of stored log is 5 files
    """
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s: %(levelname)s: %(message)s",
        datefmt="%d/%m/%Y %I:%M:%S %p",
        handlers=[
            logging.StreamHandler(),
            RotatingFileHandler(filename=Path(log_path),
                                maxBytes=max_bytes, backupCount=backup_count)
        ])
    return logging.getLogger()


def rank5_accuracy(preds, labels):
    """
    Generate the result of rank1 and rank5 accuracy
    """
    rank1 = 0
    rank5 = 0

    for (pred, ground_true) in zip(preds, labels):
        pred = np.argsort(pred)[::-1]

        if ground_true in pred[:5]:
            rank5 += 1

        if ground_true == pred[0]:
            rank1 += 1

    rank1 = rank1/float(len(preds))*100
    rank5 = rank5/float(len(preds))*100

    return (round(rank1, 2), round(rank5, 2))


def read_list_file(list_file_path):
    data = pd.read_csv(list_file_path, sep="\t", header=None)
    data.columns = ["lst_idx", "label_id", "img_path"]
    return data


def get_img_path(lst_idx, lst_data):
    return lst_data[lst_data["lst_idx"] == lst_idx]["img_path"].values[0]


def get_label_name(label_idx):
    query = f"""
    SELECT make || ' : ' || model AS make_model
    FROM make_model
    WHERE make_model_id = {label_idx}
    """
    label_df = pd.DataFrame()
    with sqlite3.connect(config.SQLITE3_DB_FILE) as conn:
        label_df = pd.read_sql_query(query, conn)

    return label_df["make_model"].values[0]


def img_show_rankN(preds, labels, list_file_path, N=5):
    lst_data = read_list_file(list_file_path)
    lst_idx = 0
    for (pred, act_label_idx) in zip(preds, labels):
        pred_sorted_idx = np.argsort(pred)[::-1]

        # Show highest score image
        pred_label_idx = pred_sorted_idx[0]
        img_path = get_img_path(lst_idx, lst_data)
        pred_score = round(pred[pred_label_idx]*100, 2)

        pred_label = get_label_name(pred_label_idx)
        act_label = get_label_name(act_label_idx)

        print(
            f"*** Predicted: {pred_label:>30} (conf.={pred_score}%) | Actual: {act_label}")

        # Prepare for display the image
        image = cv2.imread(img_path)
        orig = image.copy()
        orig = imutils.resize(orig, width=min(400, orig.shape[1]))

        img_label = pred_label.replace(": ", " ")
        img_label = f"{img_label} : {pred_score} %"
        cv2.putText(orig, img_label, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Print out the result of top N images' label
        for i in range(N-1):
            iter_label_idx = pred_sorted_idx[i+1]
            iter_pred_score = round(pred[iter_label_idx]*100, 2)
            iter_pred_label = get_label_name(iter_label_idx)
            print(
                f"\tPredicted: {iter_pred_label:>30} | confident score ={iter_pred_score:>15}%")

        lst_idx += 1
        # show the image
        cv2.imshow("Image", orig)
        cv2.waitKey(0)


def timeit(method):
    """
    The util to calculate the time consuming of each function using decorator
    """
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print(f"{method.__name__} took {round(te - ts, 2)} seconds")
        return result
    return timed


def exception_to_string(ex):
    """
    This is used for debugging purpose by logging the error detail into log file
    """
    return ''.join(traceback.format_exception(etype=type(ex), value=ex, tb=ex.__traceback__))


def row_count(file):
    with open(file, 'r') as f:
        return sum(1 for _ in f)
