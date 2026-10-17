# app/services/train.py
""""""
import logging
from typing import Any

import torch

from models.train_model import trainModel

logger = logging.getLogger(__name__)


def test_model(torch_model, training_parameters, test_dataloader, torch_device: torch.device) -> dict[str, Any]:
    try:
        torch_model.eval()
        correct = 0
        total = 0
        test_loss = 0.0

        loss_function_name = training_parameters.loss_function
        loss_function = getattr(torch.nn, loss_function_name, None)
        if loss_function is None:
            raise ValueError(f"Invalid loss function name: {loss_function_name}")
        criterion = loss_function()

        with torch.no_grad():
            for inputs, targets in test_dataloader:
                inputs = inputs.to(torch_device)
                targets = targets.to(torch_device)

                outputs = torch_model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
                test_loss += criterion(outputs, targets).item()

        accuracy = 100 * correct / total if total > 0 else 0
        test_loss /= len(test_dataloader)

        logger.info("Test accuracy: %.2f%%, Test loss: %.4f", accuracy, test_loss)

        return {"accuracy": accuracy, "test_loss": test_loss}

    except Exception as e:
        logger.exception("Error during testing")
        raise ValueError(f"Error during testing: {e}") from e
