from constants import tf, np


def iou_for_loss(boxes_true, boxes_pred):
    # this iou functions used only for training, use iou_boxes instead
    scores = tf.TensorArray(tf.float32, size=len(boxes_true))

    for i in range(len(boxes_true)):
        x_mid1, y_mid1, width1, height1 = boxes_true[i, 0], boxes_true[i, 1], boxes_true[i, 2], boxes_true[i, 3]
        x_mid2, y_mid2, width2, height2 = boxes_pred[i, 0], boxes_pred[i, 1], boxes_pred[i, 2], boxes_pred[i, 3]

        x_min1, y_min1 = x_mid1 - (width1 / 2), y_mid1 - (height1 / 2)
        x_max1, y_max1 = x_mid1 + (width1 / 2), y_mid1 + (height1 / 2)

        x_min2, y_min2 = x_mid2 - (width2 / 2), y_mid2 - (height2 / 2)
        x_max2, y_max2 = x_mid2 + (width2 / 2), y_mid2 + (height2 / 2)

        x_min = tf.math.maximum(x_min1, x_min2)
        x_max = tf.math.minimum(x_max1, x_max2)
        y_min = tf.math.maximum(y_min1, y_min2)
        y_max = tf.math.minimum(y_max1, y_max2)

        intersection = (y_max - y_min) * (x_max - x_min)
        union = (x_max1 - x_min1) * (y_max1 - y_min1) + (x_max2 - x_min2) * (y_max2 - y_min2) - intersection
        scores = scores.write(i, tf.math.maximum(0.0, intersection / union))

    return scores.stack()


class YoloV3Loss(tf.keras.losses.Loss):
    def __init__(self,
                 cluster_centers: np.array,
                 with_iou: bool = True,
                 object_cost: float = 5,
                 no_obj_cost: float = 0.5,
                 box_cost: float = 5,
                 class_ratio: float = 10,
                 **kwargs):
        super().__init__(**kwargs)
        self.object_cost = object_cost
        self.no_obj_cost = no_obj_cost
        self.box_cost = box_cost
        self.with_iou = with_iou
        self.cluster_centers = cluster_centers
        self.class_ratio = class_ratio

    def call(self, y_true, y_pred):
        loss = 0
        for i in range(3):  # for each scale
            cur_y_true = y_true[i]
            cur_y_pred = y_pred[i]
            cur_centroid = np.array(self.cluster_centers[i]).reshape(1, 2)

            obj = cur_y_true[..., 0] == 1
            no_obj = cur_y_true[..., 0] == 0

            box_coord = tf.math.sigmoid(cur_y_pred[..., 1:3])

            # No object loss
            no_obj_loss = tf.keras.losses.BinaryCrossentropy()(cur_y_true[..., 0:1][no_obj],
                                                               tf.math.sigmoid(cur_y_pred[..., 0:1][no_obj]))

            # Object loss
            if self.with_iou:
                box_pred = tf.concat([box_coord, tf.math.multiply(tf.math.exp(cur_y_pred[..., 3:5]), cur_centroid)],
                                     axis=-1)
                box_true = tf.concat(
                    [cur_y_true[..., 1:3], tf.math.multiply(tf.math.exp(cur_y_true[..., 3:5]), cur_centroid)], axis=-1)
                iou_s = iou_for_loss(box_true[obj], box_pred[obj])

                object_loss = tf.keras.losses.BinaryCrossentropy()(cur_y_true[..., 0:1][obj], tf.math.sigmoid(
                    tf.math.multiply(iou_s, cur_y_pred[..., 0:1][obj])))

            else:
                object_loss = tf.keras.losses.BinaryCrossentropy()(cur_y_true[..., 0:1][obj],
                                                                   tf.math.sigmoid(cur_y_pred[..., 0:1][obj]))

            # Box loss
            box_pred_ = tf.concat([box_coord, cur_y_pred[..., 3:5]], axis=-1)
            box_loss = tf.keras.losses.MeanSquaredError()(cur_y_true[..., 1:5][obj], box_pred_[obj])

            # Class loss
            class_loss_obj = tf.keras.losses.CategoricalCrossentropy()(cur_y_true[..., 5:][obj],
                                                                       tf.math.sigmoid(cur_y_pred[..., 5:][obj]))
            class_loss_no_obj = tf.keras.losses.CategoricalCrossentropy()(cur_y_true[..., 5:][no_obj],
                                                                          tf.math.sigmoid(cur_y_pred[..., 5:][no_obj]))

            loss += (
                object_loss * self.object_cost
                + no_obj_loss * self.no_obj_cost
                + box_loss * self.box_cost
                + class_loss_obj * self.class_ratio
                + class_loss_no_obj
            )

        return loss
