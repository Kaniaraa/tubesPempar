from mpi4py import MPI
import pandas as pd
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Rank 0 membaca data terfilter saja
if rank == 0:
    df = pd.read_csv('filtered_movies.csv', sep=',')  # file hasil filter
    data_split = np.array_split(df, size)
else:
    data_split = None

local_data = comm.scatter(data_split, root=0)

# Reduce: hitung public_vote
local_sum = local_data['public_vote'].sum()
local_count = len(local_data)

total_sum = comm.reduce(local_sum, op=MPI.SUM, root=0)
total_count = comm.reduce(local_count, op=MPI.SUM, root=0)

if rank == 0:
    if total_count > 0:
        avg_vote = total_sum / total_count
        print(f"Rata-rata public_vote dari file terfilter: {avg_vote:.2f}")
    else:
        print("File terfilter kosong.")
