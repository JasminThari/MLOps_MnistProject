from mypaths import PROJECT_ROOT, PATH_DATA
import torch
from torch.utils.data import DataLoader
import sys

sys.path.append(PROJECT_ROOT)
#from MLOps_MnistProject.train_model import train_step
#from MLOps_MnistProject.models.model import MyNeuralNet


# def test_train_step():
#     model = MyNeuralNet()
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#     criterion = torch.nn.CrossEntropyLoss()
#     dataset = torch.load(f"{PATH_DATA}/processed/train_dataset.pt")
#     dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
#     for batch_data, batch_labels in dataloader:
#         if first_batch:
#             first_batch = False
#         else:
#             break  # only check first batch
    
#     train_step(model, optimizer, criterion, batch_data, batch_labels)
#     assert model.conv1.weight.grad is not None
