from constants import np, KMeans


# Finding clusters centroids
def get_centers(y_set: list, anchors: int = 1) -> np.array:
    all_boxes_for_k_means = []

    for boxes in y_set:
        for box in boxes:
            x_min, y_min, x_max, y_max = box
            all_boxes_for_k_means.append([x_max - x_min, y_max - y_min])

    all_boxes_for_k_means = np.array(all_boxes_for_k_means)

    k_means = KMeans(anchors * 3)

    k_means.fit(all_boxes_for_k_means)
    return k_means.cluster_centers_


def get_centers_prep() -> np.array:
    cluster_centers = np.array([[284.06060606, 349.24242424], [176.77777778, 203.], [97.22222222, 130.83333333]])
    cluster_centers = cluster_centers / 416
    return cluster_centers
