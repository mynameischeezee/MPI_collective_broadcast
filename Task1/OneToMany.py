import sys
from MatrixMultiplication import matrix_multiplication
from mpi4py import MPI
from MatrixGenerator import get_matrix
from SplitMatrix import split_matrix

comm = MPI.COMM_WORLD
my_rank = comm.Get_rank()
num_of_processes = comm.Get_size()
MatrixDimension = int(sys.argv[1])

time_start = 0
second_matrix = get_matrix(MatrixDimension)

if my_rank == 0:
    first_matrix = get_matrix(MatrixDimension)
    first_matrix_row = split_matrix(first_matrix, num_of_processes)
    time_start = MPI.Wtime()
else:
    first_matrix_row = None

first_matrix_row = comm.scatter(first_matrix_row, root=0)

line = matrix_multiplication(first_matrix_row, second_matrix)
data = comm.gather(line, root=0)

if my_rank == 0:
    spent_time = MPI.Wtime() - time_start
    print("Finished in: ", spent_time)

MPI.Finalize()
