


def split_matrix(input_matrix, amount_of_workers):
    rows_split = []
    division_value = len(input_matrix) // amount_of_workers
    leftover = len(input_matrix) % amount_of_workers
    matrix_division_start = 0
    matrix_division_end = division_value + min(1, leftover)
    for i in range(amount_of_workers):
        rows_split.append(input_matrix[matrix_division_start:matrix_division_end])
        leftover = max(0, leftover - 1)
        matrix_division_start = matrix_division_end
        matrix_division_end += division_value + min(1, leftover)
    return rows_split