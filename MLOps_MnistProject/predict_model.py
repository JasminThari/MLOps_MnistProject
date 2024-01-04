#%%
from pathlib import Path
import torch
from torch.utils.data import DataLoader, TensorDataset
import os
import click

#%%
@click.group()
def cli():
    """Command line interface."""
    pass


@click.command()
@click.option("--model_path", default="models/model_mnist_latest.pt", help="model to use for prediction")
@click.option("--data_path", default="data/processed/test_dataset.pt", help="test dataloader to use for prediction")
def predict(
    model_path: str,
    data_path: str,    
) -> list:
    """Run prediction for a given model and dataloader.

    Args:
        model: model to use for prediction
        dataloader: dataloader with batches

    Returns
        Tensor of shape [N, d] where N is the number of samples and d is the output dimension of the model

    """
    dataloader = torch.load(data_path)
    test_loader = DataLoader(dataloader, batch_size=64, shuffle=False)

    model = torch.load(model_path)
    predictions = []

    with torch.no_grad():
        for images, _ in test_loader:
            images = images.unsqueeze(1)
            y_pred = model(images)
            predictions.append(y_pred)
    model_filename = Path(model_path).stem
    torch.save(predictions, f"reports/predictions/predictions_{model_filename}.pt")
    return predictions


cli.add_command(predict)

if __name__ == "__main__":
    cli()

