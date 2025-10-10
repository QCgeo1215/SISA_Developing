# manhattan_distance.py

def is_prediction_in_threshold_range(pred, true, threshold):
    """
    Check if the predicted value is within a specified Manhattan distance threshold from the true value.

    :param pred: Predicted value (2D coordinates, e.g., [x, y])
    :param true: True value (2D coordinates, e.g., [x, y])
    :param threshold: Threshold distance to determine if the prediction is within range
    :return: Boolean indicating if the prediction is within the Manhattan distance threshold
    """
    x_true, y_true = true
    x_pred, y_pred = pred
    manhattan_distance = abs(x_true - x_pred) + abs(y_true - y_pred)
    return manhattan_distance <= threshold