from mpi4py import MPI
from random import randint
import sys

from MatrixMultiplication import matrix_multiplication
from SplitMatrix import split_matrix

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
workers = comm.Get_size() - 1
CONST_MASTER_RANK = 0
MatrixDimension = int(sys.argv[1])
MatrixA = [[randint(0, 9) for i in range(MatrixDimension)] for j in range(MatrixDimension)]
MatrixB = [[randint(0, 9) for i in range(MatrixDimension)] for j in range(MatrixDimension)]
ResultMatrix = []


def distribute_and_send_data():
    rows = split_matrix(MatrixA, workers)
    pid = 1
    for element in rows:
        comm.send(element, dest=pid, tag=1)
        comm.send(MatrixB, dest=pid, tag=2)
        pid = pid + 1


def get_matrix_data():
    global ResultMatrix
    pid = 1
    for n in range(workers):
        row = comm.recv(source=pid, tag=pid)
        ResultMatrix += row
        pid = pid + 1


def master_operation():
    time_start = MPI.Wtime()
    distribute_and_send_data()
    get_matrix_data()
    spent_time = MPI.Wtime() - time_start
    print("[!] master process with #%d finished in: %5.10fs." % (rank, spent_time))


def slave_operation():
    time_start = MPI.Wtime()
    receive_first_operation = comm.recv(source=CONST_MASTER_RANK, tag=1)
    receive_second_operation = comm.recv(source=CONST_MASTER_RANK, tag=2)
    matrix_multiplication_result = matrix_multiplication(receive_first_operation, receive_second_operation)
    comm.send(matrix_multiplication_result, dest=CONST_MASTER_RANK, tag=rank)
    spent_time = MPI.Wtime() - time_start
    print("[!] slave process with #%d finished in: %5.10fs." % (rank, spent_time))


if __name__ == '__main__':
    if rank == CONST_MASTER_RANK:
        master_operation()
    else:
        slave_operation()
