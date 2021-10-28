from proj_yolov3.constants import tf, np, plt, cv2, MODEL_PATH
from proj_yolov3.yolo_loss import *


# Some useful functions
def iou_boxes(box1, box2):
    x_min1, y_min1, x_max1, y_max1 = box1
    x_min2, y_min2, x_max2, y_max2 = box2

    x_min = max(x_min1, x_min2)
    x_max = min(x_max1, x_max2)
    y_min = max(y_min1, y_min2)
    y_max = min(y_max1, y_max2)

    intersection = (y_max - y_min) * (x_max - x_min)
    union = (x_max1 - x_min1) * (y_max1 - y_min1) + (x_max2 - x_min2) * (y_max2 - y_min2) - intersection

    return max(0.0, intersection / union)


def non_max(boxes, scores, threshold=0.5):
    mask = np.ones((len(boxes),))
    for i in range(len(boxes)):
        if not mask[i]:
            continue

        for j in range(i + 1, len(boxes)):
            if not mask[j]:
                continue

            if iou_boxes(boxes[i], boxes[j]) < threshold:
                continue

            if scores[i] > scores[j]:
                mask[j] = 0
            else:
                mask[i] = 0

    mask = np.where(mask > 0.5, True, False)
    return np.array(boxes)[mask], np.array(scores)[mask]


def plot_predictions(images, true_boxes, predictions, max_images=-1):
    for i in range(len(images)):
        if i == max_images:
            break

        image = images[i].copy()
        image = tf.image.resize(image, size=(416, 416)).numpy()
        boxes = predictions[i]

        for box in boxes:
            x_min, y_min, x_max, y_max = box
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (1, 0, 0), 2, cv2.LINE_AA)

        for box in true_boxes[i]:
            x_min, y_min, x_max, y_max = box
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 1, 0), 2, cv2.LINE_AA)
        plt.imshow(image)
        plt.show()


def load_yolo():
    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={
            'YoloV3Loss': YoloV3Loss
        }
    )
    return model


def find_precision_and_recall(y_true, y_pred):
    if len(y_pred) == 0:
        return 0, 0

    results = np.zeros((len(y_true), len(y_pred)))

    for i, box_pred in enumerate(y_pred):
        for j, box_true in enumerate(y_true):
            box_true = [box_true[0], box_true[1], box_true[2] - box_true[0], box_true[3] - box_true[1]]
            if iou_boxes(box_pred, box_true) > 0.5:
                results[j, i] = 1

    true_pos = np.sum(results)
    false_negative = len(y_true) - np.sum(np.max(results, axis=1))
    false_positive = len(y_pred) - np.sum(np.max(results, axis=0))

    precision = true_pos / (true_pos + false_positive)
    recall = true_pos / (true_pos + false_negative)
    return precision, recall
