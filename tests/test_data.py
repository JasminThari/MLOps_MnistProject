from mypaths import PATH_DATA
import torch
from torch.utils.data import DataLoader
import pytest
import os 

data_folder = os.path.join(PATH_DATA, "processed")  

@pytest.mark.skipif(not os.path.exists(data_folder), reason="Data files not found")
def test_data():
    dataset = torch.load(f"{PATH_DATA}/processed/train_dataset.pt")
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    first_batch = True
    for batch_data, batch_labels in dataloader:
        if first_batch:
            first_batch = False
        else:
            break  # only check first batch
        
        assert len(batch_data) == 64, "Dataset size is not correct"
        assert batch_data.shape[0] == 64, "Dataset shape is not correct"
        assert batch_data.shape[1] == 28, "Dataset shape is not correct" 
        assert batch_data.shape[2] == 28, "Dataset shape is not correct" 
        assert batch_labels.unique().shape[0] == 10, "Dataset labels are not correct"