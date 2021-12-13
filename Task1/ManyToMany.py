import sys
from mpi4py import MPI
from numpy import array, zeros_like, arange
from numpy import empty

from MatrixGenerator import get_matrix
from MatrixMultiplication import matrix_multiplication
from SplitMatrix import split_matrix

comm = MPI.COMM_WORLD
my_rank = comm.Get_rank()
num_of_processes = comm.Get_size()
MatrixDimension = int(sys.argv[1])
time_start = 0
second_matrix = second_matrix = get_matrix(MatrixDimension)

if my_rank == 0:
    first_matrix = second_matrix = get_matrix(MatrixDimension)
    first_matrix_row = split_matrix(first_matrix, num_of_processes)

else:
    first_matrix_row = None

if my_rank == 0:
    result_matrix = [[[] for j in range(0, num_of_processes)] for i in range(0, num_of_processes)]
    time_start = MPI.Wtime()

first_matrix_row = comm.scatter(first_matrix_row, root=0)

_sendData = array(matrix_multiplication(first_matrix_row, second_matrix))
_recvData = empty(MatrixDimension*500, dtype=float)
comm.Alltoall(_sendData, _recvData)

data = comm.gather(_recvData, root=0)

# if my_rank == 0:
#     for i, row in enumerate(data):
#         for j in range(len(row)):
#             result_matrix[j][i] = row[j]
#     #print(result_matrix)

if my_rank == 0:
    spent_time = MPI.Wtime() - time_start
    print("Finished in: ", spent_time)

MPI.Finalize()
