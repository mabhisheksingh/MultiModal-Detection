from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/v1/anpr_object")


def _deprecated() -> None:
    raise HTTPException(
        status_code=410,
        detail="v1 ANPR API is deprecated. Use POST /v2/anpr_object/process instead.",
    )


@router.post("/process-image")
async def process_image():
    _deprecated()


@router.post("/process-video")
async def process_video():
    _deprecated()
