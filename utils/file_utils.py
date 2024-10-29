import os
import pandas as pd
import torch

def save_to_csv(file_directory, file_name, data):
    if not os.path.exists(file_directory):
        os.makedirs(file_directory)

    file_path = os.path.join(file_directory, file_name)

    if not os.path.exists(file_path):
        data.to_csv(file_path, index=False)

def save_to_pth(file_directory, file_name, data):
    if not os.path.exists(file_directory):
        os.makedirs(file_directory)

    file_path = file_directory + file_name

    if not os.path.exists(file_path):
        torch.save(data, file_path)

def load_csv_as_tensor(file_path):
    csv_data = pd.read_csv(file_path)
    tensor_data = torch.tensor(csv_data.values)
    return tensor_data

def load_csv_as_list(file_path):
    csv_data = pd.read_csv(file_path)
