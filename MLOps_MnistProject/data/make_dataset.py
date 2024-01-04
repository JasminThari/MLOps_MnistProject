import torch
from torchvision import transforms
from torch.utils.data import TensorDataset
import os

def make_preprocessing():
    raw_data_folder = "data/raw"
    processed_data_folder = "data/processed"
    
    train_images_data = []
    train_target_data = []
    test_images_data = []
    test_target_data = []

    for root, _, files in os.walk(raw_data_folder):
        for file in files:
            if "train_images" in file:
                file_path = os.path.join(root, file)
                tensor_data = torch.load(file_path)

                normalize = transforms.Normalize(mean=tensor_data.mean(), std=tensor_data.std())
                normalized_data = normalize(tensor_data)

                train_images_data.append(normalized_data)

            elif "train_target" in file:
                file_path = os.path.join(root, file)
                tensor_data = torch.load(file_path)
                train_target_data.append(tensor_data)

            elif "test_images" in file:
                file_path = os.path.join(root, file)
                tensor_data = torch.load(file_path)
                normalize = transforms.Normalize(mean=tensor_data.mean(), std=tensor_data.std())
                normalized_data = normalize(tensor_data)
                test_images_data.append(normalized_data)

            elif "test_target" in file:
                file_path = os.path.join(root, file)
                tensor_data = torch.load(file_path)
                test_target_data.append(tensor_data)


    data_dict = {
        "train_images_data": torch.cat(train_images_data, dim=0),
        "train_target_data": torch.cat(train_target_data, dim=0),
        "test_images_data": torch.cat(test_images_data, dim=0),
        "test_target_data": torch.cat(test_target_data, dim=0)}

    for key, tensor in data_dict.items():
        output_path = os.path.join(processed_data_folder, f"{key}.pt")
        torch.save(tensor, output_path)

def make_dataset():
    processed_data_folder = "data/processed"
    folder = os.path.join(processed_data_folder)
    
    train_images = torch.load(f"{folder}/train_images_data.pt")
    train_target = torch.load(f"{folder}/train_target_data.pt")
    train_dataset = TensorDataset(train_images, train_target)
    
    torch.save(train_dataset, f"{folder}/train_dataset.pt")
    
    test_images = torch.load(f"{folder}/test_images_data.pt")
    test_target = torch.load(f"{folder}/test_target_data.pt")
    test_dataset = TensorDataset(test_images, test_target)
    
    torch.save(test_dataset, f"{folder}/test_dataset.pt")


if __name__ == '__main__':
    # Get the data and process it
    make_preprocessing()
    make_dataset()