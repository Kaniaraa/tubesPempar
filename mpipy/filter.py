from mpi4py import MPI
import pandas as pd

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def split_dataframe(df, n):
    # Calculate the size of each chunk
    chunk_size = len(df) // n
    chunks = []
    for i in range(n):
        start = i * chunk_size
        # For the last chunk, take all remaining data
        if i == n - 1:
            end = len(df)
        else:
            end = (i + 1) * chunk_size
        chunks.append(df.iloc[start:end])
    return chunks

# Rank 0 read dan distribute data
if rank == 0:
    df = pd.read_csv('dataset/filmtv_movies.csv', sep=',')
    data_split = split_dataframe(df, size)
else:
    data_split = None

local_data = comm.scatter(data_split, root=0)

# Filter local
filtered = local_data[(local_data['public_vote'] > 7) & (local_data['year'] > 2010)]

# Enhanced visual output per rank
print(f"\n╔════════════════════════════╗")
print(f"║          RANK: {rank:<2}          ║")
print(f"╚════════════════════════════╝")
print(f"Number of films before filter: {len(local_data)}")
print(f"Number of films after filter : {len(filtered)}")
print(filtered[['title', 'year', 'public_vote']].head())


# Gather filtered results from all processes to root
all_filtered = comm.gather(filtered, root=0)

# Rank 0 merges and saves the results
if rank == 0:
    combined_df = pd.concat(all_filtered, ignore_index=True)
    combined_df.sort_values(by='year', ascending=False, inplace=True)
    combined_df.to_csv('filtered_movies.csv', index=False)
    print(f"\nSaved: filtered_movies.csv ({len(combined_df)} rows)")