from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.encoders import jsonable_encoder

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

MAX_FORWARD_PASS_ELEMENTS = 16_777_216
MAX_FORWARD_PASS_DIM = 8_192


def _validate_forward_pass_input_shape(input_shape):
    if not isinstance(input_shape, (list, tuple)):
        return "input_shape must be a list or tuple of positive integers"

    if not input_shape:
        return "input_shape must contain at least one dimension"

    total_elements = 1
    for index, dimension in enumerate(input_shape):
        if not isinstance(dimension, int):
            return f"input_shape[{index}] must be an integer"
        if dimension <= 0:
            return f"input_shape[{index}] must be greater than zero"
        if dimension > MAX_FORWARD_PASS_DIM:
            return (
                f"input_shape[{index}]={dimension} exceeds the safe limit of "
                f"{MAX_FORWARD_PASS_DIM}"
            )

        total_elements *= dimension
        if total_elements > MAX_FORWARD_PASS_ELEMENTS:
            return (
                f"input_shape has {total_elements} elements, which exceeds the safe "
                f"limit of {MAX_FORWARD_PASS_ELEMENTS}"
            )

    return None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup
    logger.info(
        "Started %s v%s",
        settings.app_name,
        settings.app_version,
    )
    yield
    # shutdown


app = FastAPI(
    title=settings.app_name,
    description=settings.app_description,
    version=settings.app_version,
    lifespan=lifespan,
)


@app.get("/")
def root() -> JSONResponse:
    return JSONResponse(
        {
            "service": settings.app_name,
            "version": settings.app_version,
        }
    )


@app.get("/get_spec_options")
async def get_spec_options():
    options = nn_core.get_llm_schema()
    return JSONResponse(content=jsonable_encoder(options))

@app.get("/get_spec_options_toon", response_class=PlainTextResponse)
async def get_spec_options_toon():
    options = toon.encode(nn_core.get_llm_schema())
    return options

@app.get("/get_json_schema", response_model=None)
async def get_json_schema():
    schema = ArchitectureSpec.model_json_schema()
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["$id"] = "https://nncore.dev/architecture.schema.json"

    return Response(
        content=json.dumps(schema, separators=(",", ":"), ensure_ascii=False),
        media_type="application/schema+json"
    )

@app.post("/validate")
async def validate(request: Request):
    try:
        request = await request.json()
        if "spec" in request:
            spec = request.get("spec", {})
        else:
            spec = request

        print("INFO: Received validation request for spec: ", spec)

        res = nn_core.validate_dict(spec)
        logger.info("Schema validation passed")

        input_shape = list(spec.get("input_shape", [2, 3, 32, 32]))
        shape_error = _validate_forward_pass_input_shape(input_shape)
        if shape_error is not None:
            logger.warning("Rejected unsafe forward-pass input shape: %s", shape_error)
            raise HTTPException(status_code=422, detail=shape_error)

        # Forward pass validation
        model = nn_core.build_model_from_dict_via_pydantic(spec)
        if input_shape[0] < 2:
            input_shape[0] = 2  # BN requires batch >= 2
        forward_error = nn_core.validate_forward_pass(model, input_shape)
        if forward_error is not None:
            raise HTTPException(status_code=400, detail=forward_error)
        logger.info("Forward pass validation passed")

        return JSONResponse(content=jsonable_encoder(res))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/validate_toon")
async def validate_toon(toon_spec: str):
    res = nn_core.validate_architecture_toon(toon_spec)
    return res

@app.post("/create_base_spec")
async def create_init_model_spec(request: Request):
    try:
        spec = await request.json()

        input_shape = spec.get("input_shape",)
        output_shape = spec.get("output_shape")
        output_activation = spec.get("output_activation")
        res = nn_core.generate_init_model(input_shape=input_shape, output_shape=output_shape, output_activation=output_activation)

        return json.loads(res)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
