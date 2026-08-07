# app/services/train.py
""""""
import logging
from typing import Any

import torch

import nn_core
from nn_core import ArchitectureSpec

from core.settings import settings
from models.train_model import trainModel

from .dataloader import get_cifar_dataloaders, get_mnist_dataloaders, _get_dataset_loader

logger = logging.getLogger(__name__)

def _parse_training_parameters(model_spec: dict[str, Any], training_parameters: dict[str, Any]) -> dict[str, Any]:
    # Derive batch_size from spec's input_shape if present, fallback to parameters
    input_shape = model_spec.get("input_shape") if isinstance(model_spec, dict) else None
    spec_batch_size = input_shape[0] if (input_shape and len(input_shape) >= 1) else None
    batch_size = spec_batch_size or training_parameters.batch_size or 32

    lr = training_parameters.lr
    momentum = training_parameters.momentum
    weight_decay = training_parameters.weight_decay
    nesterov = training_parameters.nesterov
    ending_lr = training_parameters.ending_lr
    lr_schedule = training_parameters.lr_schedule
    epochs = training_parameters.epochs

    dataset_name = training_parameters.dataset_name
    dataset_loader = _get_dataset_loader(dataset_name)

    loss_function_name = training_parameters.loss_function
    loss_function = getattr(torch.nn, loss_function_name, None)
    if loss_function is None:
        raise ValueError(f"Invalid loss function name: {loss_function_name}")

    optimizer_name = training_parameters.optimizer
    optimizer = getattr(torch.optim, optimizer_name, None)
    if optimizer is None:
        raise ValueError(f"Invalid optimizer name: {optimizer_name}")

    return {
        "dataset_loader": dataset_loader,
        "dataset_name": dataset_name,
        "batch_size": int(batch_size),
        "loss_function": loss_function,
        "optimizer": optimizer,
        "lr": float(lr),
        "momentum": float(momentum),
        "weight_decay": float(weight_decay),
        "nesterov": bool(nesterov),
        "ending_lr": float(ending_lr) if ending_lr is not None else None,
        "lr_schedule": str(lr_schedule).lower() if lr_schedule is not None else None,
        "epochs": int(epochs),
    }


def train_model(model_spec_dict: dict[str, Any], training_parameters: Any, torch_device: torch.device) -> trainModel:
    try:
        parsed = _parse_training_parameters(model_spec_dict, training_parameters)

        model = nn_core.build_model_from_dict_via_pydantic(model_spec_dict).to(torch_device)
        logger.info("Model built and moved to device: %s", torch_device)

        if parsed["dataset_loader"] == "cifar":
            train_dataloader, test_dataloader = get_cifar_dataloaders(
                parsed["batch_size"],
                parsed["batch_size"],
                parsed["dataset_name"],
                0,
                resize=None,
                datadir="_dataset",
            )
        elif parsed["dataset_loader"] == "mnist":
            train_dataloader, test_dataloader = get_mnist_dataloaders(
                parsed["batch_size"],
                parsed["batch_size"],
                0,
            )
        else:
            raise ValueError(f"Unsupported dataset loader: {parsed['dataset_loader']}")

        criterion = parsed["loss_function"]()
        optimizer_cls = parsed["optimizer"]

        optimizer_kwargs = {"lr": parsed["lr"]}
        if "momentum" in optimizer_cls.__init__.__code__.co_varnames:
            optimizer_kwargs["momentum"] = parsed["momentum"]
        if "weight_decay" in optimizer_cls.__init__.__code__.co_varnames:
            optimizer_kwargs["weight_decay"] = parsed["weight_decay"]
        if "nesterov" in optimizer_cls.__init__.__code__.co_varnames:
            optimizer_kwargs["nesterov"] = parsed["nesterov"]

        optimizer = optimizer_cls(model.parameters(), **optimizer_kwargs)
        scheduler = None
        if parsed["lr_schedule"] == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=parsed["epochs"],
                eta_min=parsed["ending_lr"] if parsed["ending_lr"] is not None else 0.0,
            )

        train_losses: list[float] = []

        logger.info("Starting training for %d epochs", parsed["epochs"])
        for epoch in range(parsed["epochs"]):
            model.train()
            running_loss = 0.0
            total_samples = 0

            for inputs, targets in train_dataloader:
                inputs = inputs.to(torch_device)
                targets = targets.to(torch_device)

                optimizer.zero_grad(set_to_none=True)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                bs = inputs.size(0)
                running_loss += float(loss.item()) * bs
                total_samples += bs

            if scheduler is not None:
                scheduler.step()

            epoch_loss = running_loss / max(total_samples, 1)
            train_losses.append(epoch_loss)
            logger.info("Epoch %d/%d - train_loss=%.6f", epoch + 1, parsed["epochs"], epoch_loss)

        return trainModel(
            training_spec=training_parameters,
            train_loss=train_losses,
            torch_model=model,
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader,
            torch_device=torch_device,
        )

    except Exception as e:
        logger.exception("Error during training")
        raise ValueError(f"Error during training: {e}") from e
