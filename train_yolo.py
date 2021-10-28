from constants import *
from data import DataEncoderAndDecoder
from yolo_model import YoloV3
from yolo_loss import *
from kmeans import get_centers_prep


if __name__ == '__main__':
    data_preparation = DataEncoderAndDecoder()

    x_train, y_train = data_preparation.get_norm_data(TRAIN_PATH)
    x_val, y_val = data_preparation.get_norm_data(VAL_PATH)
    x_test, y_test = data_preparation.get_norm_data(TEST_PATH)

    cluster_centers = get_centers_prep()

    y_train_proper_full = data_preparation.get_proper_y_set(y_train, cluster_centers, grid_size=GRID_SIZE)
    y_val_proper_full = data_preparation.get_proper_y_set(y_val, cluster_centers, grid_size=GRID_SIZE)

    yolo_model = YoloV3(num_classes=NUM_CLASSES, anchors=ANCHORS, train_vgg=TRAIN_VGG)

    yolo_model.compile(loss=YoloV3Loss(cluster_centers, with_iou=True), lr=LEARNING_RATE)
    yolo_model.train(x_train, y_train_proper_full, validation_data=(x_val, y_val_proper_full), epochs=3)

    yolo_model.compile(loss=YoloV3Loss(cluster_centers, with_iou=False), lr=LEARNING_RATE / 10)
    yolo_model.train(x_train, y_train_proper_full, validation_data=(x_val, y_val_proper_full), epochs=5)

    yolo_model.compile(loss=YoloV3Loss(cluster_centers, with_iou=True), lr=LEARNING_RATE / 10)
    yolo_model.train(x_train, y_train_proper_full, validation_data=(x_val, y_val_proper_full), epochs=5)

    yolo_model.save(MODEL_PATH)
