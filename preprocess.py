import os
import utils
import config
import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split


def export_rec_file(list_file_path):
    """
    Description: Input images from provided list_file_path will be pre-processed by simply call im2rec toolkit going
    along with mxnet library.

    Args:
    num-thread = 4  : Number of threads dedicated for the process 
    resize = 256    : The input image is resized to 256px for shorted dimension to reduce the computation. This is also considered
                    as an augmentation step during training
    encoding = jpg  : This format is better than png in term of space occupation while still keeping the recognized feature
    quality = 100   : which is the maximum quality
    """
    os.system(
        f"python {config.IM2REC_PY_PATH} {list_file_path} '' --num-thread 4 --resize 256 --encoding .jpg --quality 100")


def create_car_table():
    car_ims_drop_table = """
    DROP TABLE IF EXISTS cars;
    """

    car_ims_create_table = """
    CREATE TABLE IF NOT EXISTS cars(
    car_id integer primary key AUTOINCREMENT,
    image_name varchar(50) NOT NULL,
    make varchar(50) NOT NULL,
    model varchar(50) NOT NULL,
    type varchar(50) NOT NULL,
    year integer NOT NULL,
    make_model_id integer NULL,
    CONSTRAINT image_name_unique UNIQUE(image_name)
    )
    """
    with sqlite3.connect(config.SQLITE3_DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute(car_ims_drop_table)
        cursor.execute(car_ims_create_table)
        cursor.close()


def load_car_data():
    df_car_ims = pd.read_csv(config.FULL_DATASET_FILE)
    query = """
    INSERT INTO cars(image_name, make, model, type, year)
    VALUES (?, ?, ?, ?, ?)
    """
    with sqlite3.connect(config.SQLITE3_DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.executemany(query, df_car_ims.values.tolist())
        cursor.close()


def create_make_model_table():
    make_model_drop_table = """
    DROP TABLE IF EXISTS make_model;
    """

    make_model_create_table = """
    CREATE TABLE IF NOT EXISTS make_model(
    make_model_id integer primary key AUTOINCREMENT,
    make varchar(50) NOT NULL,
    model varchar(50) NOT NULL,
    CONSTRAINT make_model_unique UNIQUE(make, model)
    )
    """
    with sqlite3.connect(config.SQLITE3_DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute(make_model_drop_table)
        cursor.execute(make_model_create_table)
        cursor.close()


def load_make_model_data():
    make_model_insert_data = """
    INSERT INTO make_model(make, model)
    SELECT DISTINCT make, model 
    FROM cars
    """

    make_model_update_id = """
    UPDATE make_model
    SET make_model_id = make_model_id - 1
    """
    with sqlite3.connect(config.SQLITE3_DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute(make_model_insert_data)
        cursor.execute(make_model_update_id)
        cursor.close()


def update_make_mode_id():
    query = """
    UPDATE cars
    SET [make_model_id] = (
                            SELECT m.[make_model_id]
                            FROM make_model m
                            WHERE m.[make] = cars.[make] AND m.[model] = cars.[model]
                        )
    """

    with sqlite3.connect(config.SQLITE3_DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute(query)
        cursor.close()


def get_car_data():
    query = f"""
    SELECT '{config.TRAIN_IMG_PATH}/' || image_name AS img_path,
            make_model_id AS img_label,
            make || ':' || model as img_description
    FROM cars
    """
    data_collection = pd.DataFrame()
    with sqlite3.connect(config.SQLITE3_DB_FILE) as conn:
        data_collection = pd.read_sql_query(query, conn)

    return data_collection


def run_preprocess(logger):
    # Prepare the sqlite3 database which will include 5 below steps
    logger.debug("Create car table")
    create_car_table()
    logger.debug("Load data from csv file to car table")
    load_car_data()
    logger.debug("Create car maker and car model table")
    create_make_model_table()
    logger.debug(
        "Insert data into make_model table based on fact data from cars table")
    load_make_model_data()
    logger.debug("Update back the id or label_id to the cars table")
    update_make_mode_id()

    # Get the car information including the path to images and their labels
    logger.debug("Prepare data to generate list and rec files")
    data_collection = get_car_data()

    # The dataset is to be splited into 15% for validation, 15% for test and the rest 70% for training
    num_val = int(config.NUM_VAL_IMAGES * data_collection.shape[0])
    num_test = int(config.NUM_TEST_IMAGES * data_collection.shape[0])

    # Initialize the image path and image label
    train_paths = data_collection["img_path"].tolist()
    train_labels = data_collection["img_label"].tolist()

    # Prepare VALIDATION data set with 15% predefined on the total dataset
    # Split the train with stratify specified to labels dataset to prevent bias
    split = train_test_split(train_paths, train_labels,
                             test_size=num_val, stratify=train_labels)
    (train_paths, val_paths, train_labels, val_labels) = split

    # Prepare TEST data set with 15% predefined on the total dataset
    split = train_test_split(train_paths, train_labels,
                             test_size=num_test, stratify=train_labels)
    (train_paths, test_paths, train_labels, test_labels) = split

    # Prepare list and rec files for the MXNET used later
    datasets = [
        ("train", train_paths, train_labels, config.TRAIN_LIST_FILE),
        ("val", val_paths, val_labels, config.VAL_LIST_FILE),
        ("test", test_paths, test_labels, config.TEST_LIST_FILE)]

    logger.debug("Generating lst and rec files ...")
    for (_, paths, labels, output_path) in datasets:
        with open(output_path, "w") as f:
            for (i, (path, label)) in enumerate(zip(paths, labels)):
                row = "\t".join([str(i), str(label), path])
                f.write(f"{row}\n")
        export_rec_file(output_path)


if __name__ == "__main__":
    # Init the logger
    logger = utils.get_logger(f"./logs/preprocessing.log")
    try:
        run_preprocess(logger)
    except Exception as ex:
        error_message = utils.exception_to_string(ex)
        logger.error(error_message)
