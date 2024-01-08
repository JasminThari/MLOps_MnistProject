import sys

import pytest
import torch
from mypaths import PROJECT_ROOT

sys.path.append(PROJECT_ROOT)
from MLOps_MnistProject.models.model import MyNeuralNet


@pytest.mark.parametrize("batch_size", [64, 128])
def test_model(batch_size):
    dummy_data = torch.randn(batch_size, 1, 28, 28)
    model = MyNeuralNet()
    y_pred = model(dummy_data)
    assert y_pred.shape == (batch_size, 10)


# tests/test_model.py
def test_error_on_wrong_shape():
    model = MyNeuralNet()
    with pytest.raises(ValueError, match='Expected input to a 4D tensor'):
        model(torch.randn(1,2,3))
