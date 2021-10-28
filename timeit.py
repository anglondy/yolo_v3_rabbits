from constants import *
from data import DataEncoderAndDecoder
import time


if __name__ == '__main__':
    data_preparation = DataEncoderAndDecoder()
    x_train, _ = data_preparation.get_norm_data(TRAIN_PATH)

    yolo_v3 = load_yolo()

    start = time.time()
    yolo_v3.predict(x_train)

    print('FPS ~', int(len(x_train) // (time.time() - start)))
    # on GPU: 50-100 FPS
