from mpi4py import MPI
import pandas as pd
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Rank 0 membaca dan membagi data
if rank == 0:
    df = pd.read_csv('dataset/filmtv_movies.csv', sep=',')
    data_split = np.array_split(df, size)
else:
    data_split = None

local_data = comm.scatter(data_split, root=0)

# Filter lokal
filtered = local_data[(local_data['public_vote'] > 7) & (local_data['year'] > 2010)]

# Tampilkan hasil filter per proses
print(f"[Rank {rank}] Jumlah film setelah filter: {len(filtered)}")
print(filtered[['title', 'year', 'public_vote']].head())
