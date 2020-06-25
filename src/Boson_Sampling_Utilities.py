# TR TODO: Add proper type hints.


def per(mtx, column, selected, prod):
    """
    Row expansion for the permanent of matrix mtx.
    The counter column is the current column,
    selected is a list of indices of selected rows,
    and prod accumulates the current product.
    """
    if column == mtx.shape[1]:
        return prod
    else:
        result = 0
        for row in range(mtx.shape[0]):
            if row not in selected:
                result = result \
                         + per(mtx, column + 1, selected + [row], prod * mtx[row, column])
        return result


def calculate_permanent(mat):
    """
    Returns the permanent of the matrix mat.
    """
    return per(mat, 0, [], 1)
