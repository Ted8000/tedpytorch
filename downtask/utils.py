import logging
import pickle
import time


def get_logger():
    # pathname = "./log/{}_{}.txt".format(dataset, time.strftime("%m-%d_%H-%M-%S"))
    # logger = logging.getLogger()
    # logger.setLevel(logging.INFO)
    # formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s",
    #                               datefmt='%Y-%m-%d %H:%M:%S')

    # file_handler = logging.FileHandler(pathname)
    # file_handler.setLevel(logging.DEBUG)
    # file_handler.setFormatter(formatter)

    # stream_handler = logging.StreamHandler()
    # stream_handler.setLevel(logging.DEBUG)
    # stream_handler.setFormatter(formatter)

    # logger.addHandler(file_handler)
    # logger.addHandler(stream_handler)
    logging.basicConfig(format='%(asctime)s - %(message)s', filename='log', filemode='a', level=logging.INFO)
    logger = logging

    return logger


def save_file(path, data):
    with open(path, "wb") as f:
        pickle.dump(data, f)


def load_file(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data