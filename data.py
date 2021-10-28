from constants import np, tf, pd, os, GRID_SIZE
from utils.utils import non_max


# Class for data preparation for training and encoding after predicting
class DataEncoderAndDecoder:
    @staticmethod
    def get_image_jpg(path: str) -> np.array:  # get single image array
        image = tf.io.read_file(path)
        return tf.image.decode_jpeg(image, channels=3)

    def get_norm_data(self, path: str, target_shape=(224, 224)) -> (np.array, list):
        # Returns (np.array with shape (num_images, target_shape[0], target_shape[1], 3) and list with box coordinates)
        df = pd.read_csv(os.path.join(path, '_annotations.csv'))
        labels = []
        images = []
        names = []

        for i in range(df.shape[0]):
            line = df.iloc[i].to_numpy()
            if line[0] in names:
                labels[-1].append(line[-4:])
                continue

            labels.append([line[-4:]])
            images.append(self.get_image_jpg(os.path.join(path, line[0])))
            names.append(line[0])

        images = np.array(images, dtype=np.float32)
        images = tf.image.resize(images, size=target_shape).numpy()
        images = images / 255
        return images, labels

    @staticmethod
    def is_closest_centroid(box, centroids, grid_size, input_shape=(416, 416)) -> bool:
        box_to_w_and_h = np.array([(box[2] - box[0]) / input_shape[1], (box[3] - box[1]) / input_shape[0]])

        dist_1 = np.sum((box_to_w_and_h - centroids[0]) ** 2)
        dist_2 = np.sum((box_to_w_and_h - centroids[1]) ** 2)
        dist_3 = np.sum((box_to_w_and_h - centroids[2]) ** 2)

        if grid_size == GRID_SIZE:  # First scale prediction
            if dist_1 < min(dist_2, dist_3):
                return True
            return False

        elif grid_size == GRID_SIZE * 2:  # Second scale prediction
            if dist_2 < min(dist_1, dist_3):
                return True
            return False

        if dist_3 < min(dist_1, dist_2):  # Last scale prediction
            return True
        return False

    def get_proper_y_set_single_grid(self, y_set, cluster_centroids, grid_size=7, input_shape=(416, 416)) -> np.array:
        proper_y = np.zeros((len(y_set), grid_size, grid_size, 6))
        for i, boxes in enumerate(y_set):
            for box in boxes:
                if not self.is_closest_centroid(box, cluster_centroids, grid_size=grid_size, input_shape=input_shape):
                    continue

                if grid_size == 7:
                    centroid = cluster_centroids[0]
                elif grid_size == 14:
                    centroid = cluster_centroids[1]
                else:
                    centroid = cluster_centroids[2]

                x_min, y_min, x_max, y_max = box

                x_mid = int((x_min + x_max) / 2)
                y_mid = int((y_min + y_max) / 2)

                det_x = input_shape[1] // grid_size
                det_y = input_shape[0] // grid_size

                grid_x = y_mid // det_y
                grid_y = x_mid // det_x

                if grid_x >= grid_size:
                    grid_x = grid_size - 1  # just to dodge some bugs
                if grid_y >= grid_size:
                    grid_y = grid_size - 1

                x_mid = (x_mid % det_x) / det_x
                y_mid = (y_mid % det_y) / det_y

                w = np.log((x_max - x_min) / input_shape[1] / centroid[1] + 1e-10)
                h = np.log((y_max - y_min) / input_shape[0] / centroid[0] + 1e-10)

                proper_y[i, grid_x, grid_y] = [1, y_mid, x_mid, h, w, 1]

        return np.array(proper_y)

    def get_proper_y_set(self, y_set, cluster_centroids, grid_size=7, input_shape=(416, 416)) -> list[np.array]:
        y_train_proper_1 = self.get_proper_y_set_single_grid(y_set, cluster_centroids,
                                                             input_shape=input_shape, grid_size=grid_size)
        y_train_proper_2 = self.get_proper_y_set_single_grid(y_set, cluster_centroids,
                                                             input_shape=input_shape, grid_size=grid_size * 2)
        y_train_proper_3 = self.get_proper_y_set_single_grid(y_set, cluster_centroids,
                                                             input_shape=input_shape, grid_size=grid_size * 4)
        y_train_proper_full = [y_train_proper_1, y_train_proper_2, y_train_proper_3]
        return y_train_proper_full

    @staticmethod
    def decode_boxes(predictions, cluster_centroids,
                     grid_size=GRID_SIZE, target_shape=(416, 416), threshold_1=0.5, threshold_2=0.5) -> (list, list):
        all_boxes = []
        all_scores = []
        for num, grid in enumerate(predictions):
            all_boxes.append([])
            all_scores.append([])
            for i in range(predictions.shape[1]):
                for j in range(predictions.shape[2]):
                    is_object = tf.math.sigmoid(grid[i, j, 0]).numpy()
                    score = tf.math.sigmoid(grid[i, j, 5]).numpy()
                    if is_object > threshold_1 and score > threshold_2:
                        if grid_size == 7:
                            centroid = cluster_centroids[0]
                        elif grid_size == 14:
                            centroid = cluster_centroids[1]
                        else:
                            centroid = cluster_centroids[2]

                        x, y, w, h = grid[i, j, 1:5]
                        x_mid, y_mid = int((x + i) * (target_shape[0] // grid_size)), int(
                            (y + j) * (target_shape[1] // grid_size))
                        w, h = int(min(np.exp(w) * target_shape[0] * centroid[0], target_shape[1])), int(
                            min(np.exp(h) * target_shape[1] * centroid[1], target_shape[0]))

                        box_decoded: list[int] = [
                            int(y_mid - h / 2),
                            int(x_mid - w / 2),
                            int(y_mid + h / 2),
                            int(x_mid + w / 2)
                        ]

                        box_decoded[0] = max(0, box_decoded[0])
                        box_decoded[1] = max(0, box_decoded[1])
                        box_decoded[2] = min(target_shape[1], box_decoded[2])
                        box_decoded[3] = min(target_shape[0], box_decoded[3])

                        all_boxes[num].append(np.array(box_decoded))
                        all_scores[num].append(score)
        return all_boxes, all_scores

    def decode_predictions(self, predictions: list, clusters: np.array) -> (list, list):
        decoded_test_pred_1, test_scores_1 = self.decode_boxes(predictions[0], clusters)
        decoded_test_pred_2, test_scores_2 = self.decode_boxes(predictions[1], clusters, grid_size=GRID_SIZE * 2)
        decoded_test_pred_3, test_scores_3 = self.decode_boxes(predictions[2], clusters, grid_size=GRID_SIZE * 4)

        pred, scores = [], []
        for i in range(len(decoded_test_pred_1)):
            pred.append([])
            scores.append([])

            for box in decoded_test_pred_1[i]:
                pred[i].append(box)

            for box in decoded_test_pred_2[i]:
                pred[i].append(box)

            for box in decoded_test_pred_3[i]:
                pred[i].append(box)

            for score in test_scores_1[i]:
                scores[i].append(score)

            for score in test_scores_2[i]:
                scores[i].append(score)

            for score in test_scores_3[i]:
                scores[i].append(score)

        return pred, scores

    @staticmethod
    def final_boxes(predictions, scores):
        fin_boxes, fin_scores = [], []
        for i in range(len(predictions)):
            boxes = predictions[i]
            score = scores[i]

            boxes, score = non_max(boxes, score)

            fin_boxes.append(boxes)
            fin_scores.append(scores)

        return fin_boxes, fin_scores
