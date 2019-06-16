import time
import logging
import traceback
import numpy as np
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
