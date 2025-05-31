from mpi4py import MPI
import pandas as pd
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Rank 0 membaca dan membagi data
if rank == 0:
    try:
        df = pd.read_csv('filtered_movies.csv')  # file hasil filter genre Action
        if 'public_vote' not in df.columns:
            raise ValueError("Kolom 'public_vote' tidak ditemukan.")
        
        data_split = np.array_split(df, size)
        print(f"[Rank 0] Jumlah total data: {len(df)} baris")
    except Exception as e:
        print(f"[Rank 0] Gagal membaca file: {e}")
        data_split = [pd.DataFrame()] * size
else:
    data_split = None

# Scatter data ke semua proses
local_data = comm.scatter(data_split, root=0)

# Cek validitas data
if 'public_vote' in local_data.columns:
    local_sum = local_data['public_vote'].to_numpy().sum()
    local_count = len(local_data)
else:
    local_sum = 0.0
    local_count = 0

# Kumpulkan data ke rank 0
total_sum = comm.reduce(local_sum, op=MPI.SUM, root=0)
total_count = comm.reduce(local_count, op=MPI.SUM, root=0)

# (Opsional) Kirim semua hasil lokal untuk ditampilkan kontribusinya
all_sums = comm.gather(local_sum, root=0)
all_counts = comm.gather(local_count, root=0)

# Output akhir hanya di rank 0
if rank == 0:
    print("\n --- HASIL AKHIR MPI ---")
    if total_count > 0:
        avg_vote = total_sum / total_count
        print(f" Jumlah film: {total_count}")
        print(f" Total public_vote: {total_sum:.2f}")
        print(f" Rata-rata public_vote: {avg_vote:.2f}")
    else:
        print(" File terfilter kosong.")

    print("\n Kontribusi per proses:")
    for i in range(size):
        percent = (all_sums[i] / total_sum * 100) if total_sum else 0
        print(f" - Rank {i}: {all_counts[i]} data, sum = {all_sums[i]:.2f} ({percent:.2f}%)")
