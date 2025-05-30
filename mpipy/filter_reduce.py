from mpi4py import MPI
import pandas as pd
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Hanya rank 0 yang membaca data, lalu membaginya
if rank == 0:
    df = pd.read_csv('dataset/filmtv_movies.csv', sep=',')
    # Bagi data secara merata ke semua proses
    data_split = np.array_split(df, size)
else:
    data_split = None

# Scatter: bagi data ke tiap proses
local_data = comm.scatter(data_split, root=0)

# Filter lokal: ambil film dengan vote publik > 7 dan tahun > 2010
filtered = local_data[(local_data['public_vote'] > 7) & (local_data['year'] > 2010)]

# Reduce: hitung rata-rata 'public_vote' dari data yang sudah difilter
local_sum = filtered['public_vote'].sum()
local_count = len(filtered)

# Reduce total sum dan count ke rank 0
total_sum = comm.reduce(local_sum, op=MPI.SUM, root=0)
total_count = comm.reduce(local_count, op=MPI.SUM, root=0)

# Rank 0 hitung rata-rata final
if rank == 0:
    if total_count > 0:
        avg_vote = total_sum / total_count
        print(f"Rata-rata public_vote untuk film setelah 2010 dengan public_vote > 7 adalah: {avg_vote:.2f}")
    else:
        print("Tidak ada data yang memenuhi filter.")
