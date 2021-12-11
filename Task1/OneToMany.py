from mpi4py import MPI
from random import randint
import sys

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
workers = comm.Get_size() - 1
CONST_MASTER_RANK = 0
MatrixDimension = int(sys.argv[1])
MatrixA = [[randint(0, 9) for i in range(MatrixDimension)] for j in range(MatrixDimension)]
MatrixB = [[randint(0, 9) for i in range(MatrixDimension)] for j in range(MatrixDimension)]
ResultMatrix = []

