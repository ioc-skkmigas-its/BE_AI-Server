from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/")
def root(request: Request) -> dict[str, str]:
    return {
        "message": "FastAPI + AutoGluon starter is running",
        "version": request.app.state.settings.app_version,
    }
