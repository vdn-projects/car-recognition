import config
import mxnet as mx
import argparse
import os
import utils
import preprocess


@utils.timeit
def run_test(logger):
    """
    There are two testing modes as described below:
    1. grab_test:
      - The script grab the images and lst file from GRAB_LIST_FILE to build the rec file
      - After that the rec file is feed into the model testing function to evaluate the result(rank1 & rank5 accuracy percentage)
    2. not grab_test:
    - This is simply the self test for validating the model
    """
    img_rec_path = config.TEST_REC_FILE
    batch_size = config.BATCH_SIZE
    if args["mode"] == "grab_test":
        logger.debug("Preprocess test images from GRAB")
        preprocess.export_rec_file(config.GRAB_LIST_FILE)
        img_rec_path = config.GRAB_REC_FILE
        batch_size = min(
            batch_size, preprocess.row_count(config.GRAB_LIST_FILE))

    # Initialize an mxnet ImageRecordIter instance for the test record
    # The RGB mean apply as a normalization method with exact setting of orginal VGG16 dataset
    logger.debug("Load test-images record file")
    test_iter = mx.io.ImageRecordIter(
        path_imgrec=img_rec_path,
        data_shape=config.IMAGE_SIZE,
        batch_size=batch_size,
        mean_r=config.R_MEAN,
        mean_g=config.G_MEAN,
        mean_b=config.B_MEAN)

    logger.debug(
        f"Load the model argument and auxiliary parameters from checkpoint#{epoch}")
    (symbol, arg_params, aux_params) = mx.model.load_checkpoint(
        config.CHECKPOINT_PATH, epoch)

    logger.debug("Initialize the model before evaluation the test dataset")
    model = mx.mod.Module(symbol=symbol, context=config.MODEL_PROCESS_CONTEXT)
    model.bind(data_shapes=test_iter.provide_data,
               label_shapes=test_iter.provide_label)
    model.set_params(arg_params, aux_params)

    predictions = []
    targets = []
    # Iterate the batches and append a list of ouput prediction results
    for (preds, _, batch) in model.iter_predict(test_iter):
        preds = preds[0].asnumpy()
        labels = batch.label[0].asnumpy().astype("int")

        predictions.extend(preds)
        targets.extend(labels)

    # Ouput the result with rank1 and rank5
    (rank1, rank5) = utils.rank5_accuracy(
        predictions, targets[:len(predictions)])
    print(f"Total #test images: {len(predictions)}")
    print(f"rank-1 accuracy: {rank1}%")
    print(f"rank-5 accuracy: {rank5}%")


if __name__ == "__main__":
    # Init argument parser
    ap = argparse.ArgumentParser()
    ap.add_argument("-e", "--epoch", type=int, default=config.DEFAULT_EPOCH_NUMBER,
                    help="model stored with desired epoch number")

    ap.add_argument("-m", "--mode", default="self_test",
                    help="working mode that is to use the standford dataset(default) or the dataset provided by Grab AI for S.E.A")

    args = vars(ap.parse_args())

    # Parsing the epoch number
    epoch = args["epoch"]

    # Init logger
    logger = utils.get_logger(f"./logs/testing_{epoch}.log")

    try:
        run_test(logger)
    except Exception as ex:
        error_message = utils.exception_to_string(ex)
        logger.error(error_message)
