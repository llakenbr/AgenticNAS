from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.encoders import jsonable_encoder

from predictor import model_predictor_test_cifar10

import torch

from pydantic import BaseModel
import json
import toon

import nn_core
from nn_core import ArchitectureSpec

configure_logging(
    level=settings.log_level if hasattr(settings, "log_level") else "INFO",
    service_name=settings.app_name,
    log_dir=settings.log_dir,
)

logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup
    logger.info(
        "Started %s v%s with device: %s",
        settings.app_name,
        settings.app_version,
        device,
    )
    yield
    # shutdown


app = FastAPI(
    title=settings.app_name,
    description=settings.app_description,
    version=settings.app_version,
    lifespan=lifespan,
)


def _sanitize_for_json(obj):
    """
    Recursively sanitize an object to make it JSON-serializable.
    Replaces NaN with null, inf with null, and handles nested structures.

    Args:
        obj: Object to sanitize (dict, list, or scalar)

    Returns:
        Sanitized version of the object
    """
    if isinstance(obj, dict):
        return {key: _sanitize_for_json(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(item) for item in obj]
    elif isinstance(obj, float):
        if math.isnan(obj):
            logger.warning("Sanitizing NaN to null in response")
            return None
        elif math.isinf(obj):
            logger.warning(f"Sanitizing inf ({'+inf' if obj > 0 else '-inf'}) to null in response")
            return None
        else:
            return obj
    else:
        return obj


@app.get("/")
def root() -> JSONResponse:
    return JSONResponse(
        {
            "service": settings.app_name,
            "version": settings.app_version,
            "dataset_options": settings.dataset_options,
        }
    )


@app.post("/zero_estimate")
async def zero_estimate(request: Request):
    try:
        model_spec = template.model_spec
        if isinstance(model_spec, str):
            try:
                model_spec = json.loads(model_spec)
            except json.JSONDecodeError as e:
                raise ValueError(f"model_spec is a string but not valid JSON: {e}")

        # Derive batch_size from spec's input_shape if present, fallback to parameters
        input_shape = model_spec.get("input_shape") if isinstance(model_spec, dict) else None
        spec_batch_size = input_shape[0] if (input_shape and len(input_shape) >= 1) else None

        parameters = template.parameters
        batch_size = spec_batch_size or parameters.batch_size or 32
        dataset_name = parameters.dataset_name
        output_nodes = parameters.output_nodes
        num_imgs_or_batches = parameters.num_imgs_or_batches or 2

        logger.info("Received zero_estimate request with dataset: %s, batch_size: %i, output_nodes: %i", dataset_name, batch_size, output_nodes)

        res = predict_model_performance(model_spec, batch_size=batch_size, dataset_name=dataset_name, num_imgs_or_batches=num_imgs_or_batches, output_nodes=output_nodes, torch_device=device)
        logger.info("Completed zero_estimate prediction")

        # Sanitize response to remove NaN/inf values before JSON serialization
        sanitized_res = _sanitize_for_json(res)
        logger.debug("Sanitized response: %s", sanitized_res)

        return JSONResponse(content=jsonable_encoder(sanitized_res))

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))