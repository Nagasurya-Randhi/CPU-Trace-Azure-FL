import os
import json
import numpy as np
from numpy.random import dirichlet
from config import DATA_DIR, NUM_CLIENTS, ALPHA

def distribute_files():
    all_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
    num_files = len(all_files)
    proportions = dirichlet(np.ones(NUM_CLIENTS) * ALPHA, size=1).flatten()
    split_indices = (proportions * num_files).astype(int)
    current_sum = split_indices.sum()
    diff = num_files - current_sum
    
    for i in range(diff):
        split_indices[np.argmin(split_indices)] += 1
    for i in range(-diff):
        split_indices[np.argmax(split_indices)] -= 1

    client_files = {}
    start_index = 0
    for client_id in range(NUM_CLIENTS):
        end_index = start_index + split_indices[client_id]
        client_files[client_id] = all_files[start_index:end_index]
        start_index = end_index
        print(f"Client {client_id} has {len(client_files[client_id])} files.")

    with open('file_allocation.json', 'w') as f:
        json.dump(client_files, f)

if __name__ == "__main__":
    distribute_files()
