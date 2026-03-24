import numpy as np


def get_y_from_matrix(config, matrix, weighted=False):
    """
    Get the predicted values of target variable with prediction length L-1

    Args:
        config: Configuration object with attributes:
            - EMBEDDING_LEN (int): Length of embedding
            - TRAIN_LEN (int): Length of training data
        matrix: numpy array of shape [L, m] containing the values
        weighted: bool, whether to calculate weighted mean

    Returns:
        numpy array of predicted values with length (EMBEDDING_LEN - 1)
    """
    # Input validation
    if not isinstance(matrix, np.ndarray) or matrix.ndim != 2:
        raise ValueError("matrix must be a 2D numpy array")

    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)

    L, m = matrix.shape
    if L < config.EMBEDDING_LEN or m < config.TRAIN_LEN:
        raise ValueError("matrix dimensions are smaller than required by config")

    if m == 1:
        matrix = matrix.flatten()
        predict_y = []

        for i in range(config.EMBEDDING_LEN - 1):

            start_idx = config.TRAIN_LEN - config.EMBEDDING_LEN + i + 1

            diag_values = []
            for j in range(config.EMBEDDING_LEN - 1 - i):
                row_idx = config.EMBEDDING_LEN - 1 - j
                col_idx = start_idx + j

                if row_idx < L and col_idx < L:
                    diag_values.append(matrix[row_idx])

            if weighted and diag_values:
                y_count = len(diag_values)
                weights = np.arange(1, y_count + 1, dtype=np.float32)
                weights /= weights.sum()
                predict_y.append(np.dot(weights, diag_values))
            elif diag_values:
                predict_y.append(np.mean(diag_values))
            else:
                predict_y.append(0.0)

        return np.array(predict_y)

    predict_y = []
    total_predictions = config.EMBEDDING_LEN - 1

    for i in range(total_predictions):
        y = []
        start_col = config.TRAIN_LEN - config.EMBEDDING_LEN + i + 1

        # Ensure we don't go out of matrix bounds
        max_j = min(config.EMBEDDING_LEN - 1 - i, m - start_col)

        for j in range(max_j):
            row = config.EMBEDDING_LEN - 1 - j
            col = start_col + j
            if row < L and col < m:  # Double check bounds
                y.append(matrix[row, col])

        if len(y) == 0:
            predict_y.append(0.0)  # or np.nan depending on your needs
            continue

        if weighted:
            y_count = len(y)
            weights = np.arange(1, y_count + 1, dtype=np.float32)
            weights /= weights.sum()  # Normalize
            predict_y.append(np.dot(weights, y))
        else:
            predict_y.append(np.mean(y))

    return np.array(predict_y[:total_predictions])  # Ensure correct length