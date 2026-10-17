""""""
import logging
from typing import Any

import torch

import nn_core
from nn_core import ArchitectureSpec

from foresight.pruners.predictive import find_measures
from foresight.pruners import measures as zc_measures
from foresight.dataset import get_cifar_dataloaders, get_mnist_dataloaders

from core.settings import settings

logger = logging.getLogger(__name__)


def _compute_measures_with_fallback(model: torch.nn.Module, train_dataloader, dataload_info: list[Any], torch_device: torch.device) -> dict[str, Any]:
    try:
        return find_measures(
            model,
            dataloader=train_dataloader,
            dataload_info=dataload_info,
            device=torch_device,
        )
    except Exception as exc:
        logger.warning(
            "Full measure computation failed; falling back to per-measure execution: %s",
            exc,
        )

    fallback_measures: dict[str, Any] = {}
    for measure_name in zc_measures.available_measures:
        try:
            one_measure = find_measures(
                model,
                dataloader=train_dataloader,
                dataload_info=dataload_info,
                device=torch_device,
                measure_names=[measure_name],
            )
            fallback_measures[measure_name] = one_measure.get(measure_name, "not computable")
        except Exception as measure_exc:
            logger.warning("Measure %s is not computable: %s", measure_name, measure_exc)
            fallback_measures[measure_name] = "not computable"

    return fallback_measures


def predict_model_performance(model_spec_dict: dict[str, Any], batch_size: int, dataset_name: str, num_imgs_or_batches: int, output_nodes: int, torch_device: torch.device):
    dataset_loader = _get_dataset_loader(dataset_name)

    model = nn_core.build_model_from_dict_via_pydantic(model_spec_dict)
    model = model.to(torch_device)
    logger.info("Model built successfully and moved to device: %s", torch_device)

    if dataset_loader == "cifar":
        train_dataloader, test_dataloader = get_cifar_dataloaders(batch_size, batch_size, dataset_name, 0, resize=None, datadir='_dataset')

    elif dataset_loader == "mnist":
        train_dataloader, test_dataloader = get_mnist_dataloaders(batch_size, batch_size, 0)

    dataload = "random"
    num_classes = output_nodes

    dataload_info = [dataload, num_imgs_or_batches, num_classes]

    logger.info(
        "Dataloader %s prepared. Starting measure computation (batch_size=%d, dataload=%s, num_imgs_or_batches=%d, output_nodes=%d)",
        dataset_loader,
        batch_size,
        dataload,
        num_imgs_or_batches,
        num_classes,
    )
    logger.debug("Dataloader info: %s, Dataset: %s", dataload_info, dataset_name)
    measures = _compute_measures_with_fallback(
        model=model,
        train_dataloader=train_dataloader,
        dataload_info=dataload_info,
        torch_device=torch_device,
    )
    logger.info("Measure computation completed for dataset: %s", dataset_name)
    logger.debug("Measures result: %s", measures)
    return measures


def _get_dataset_loader(dataset_name: str) -> str:
    dataset = next(
        (
            dataset
            for dataset in settings.dataset_options["dataset_options"]
            if dataset["name"] == dataset_name
        ),
        None,
    )

    if dataset is None:
        available = ", ".join(
            d["name"] for d in settings.dataset_options["dataset_options"]
        )
        raise ValueError(
            f"Unknown dataset_name '{dataset_name}'. Allowed values: {available}"
        )

    return str(dataset["dataloader"])
