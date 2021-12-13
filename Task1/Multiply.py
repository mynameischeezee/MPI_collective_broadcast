from Task1.MatrixMultiplication import matrix_multiplication


def line_and_second_matrix_multiplication(line, second_matrix):
    matrix_size = len(line)
    result_line = []
    for element in line:
        line_sum = matrix_multiplication(element, second_matrix)
        result_line.append(line_sum)
    return result_line