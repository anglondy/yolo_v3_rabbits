from constants import *
from data import DataEncoderAndDecoder
from yolo_model import YoloV3
from yolo_loss import YoloV3Loss
from kmeans import get_centers_prep


if __name__ == '__main__':
    data_preparation = DataEncoderAndDecoder()

    x_test, y_test = data_preparation.get_norm_data(TEST_PATH)

    cluster_centers = get_centers_prep()

    yolo_model = load_yolo()
    y_pred = yolo_model.predict(x_test)

    predictions, scores = data_preparation.decode_predictions(y_pred, cluster_centers)
    final_predictions, _ = data_preparation.final_boxes(predictions, scores)

    plot_predictions(x_test, y_test, final_predictions)
