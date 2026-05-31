import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from triton_server.app import anpr_v1 as anpr_v1_routes
from triton_server.app import app


def test_v1_anpr_routes_are_not_registered():
    client = TestClient(app)

    response = client.post("/v1/anpr_object/process-image", json={})

    assert response.status_code == 404


def test_v2_anpr_route_is_registered():
    paths = {route.path for route in app.routes}

    assert "/v2/anpr_object/process" in paths
    assert "/v1/anpr_object/process-image" not in paths
    assert "/v1/anpr_object/process-video" not in paths


def test_v1_deprecated_helper_raises_http_410():
    with pytest.raises(HTTPException) as exc:
        anpr_v1_routes._deprecated()

    assert exc.value.status_code == 410


@pytest.mark.anyio
async def test_v1_deprecated_stub_endpoints_raise_http_410():
    with pytest.raises(HTTPException) as img_exc:
        await anpr_v1_routes.process_image()
    with pytest.raises(HTTPException) as vid_exc:
        await anpr_v1_routes.process_video()

    assert img_exc.value.status_code == 410
    assert vid_exc.value.status_code == 410


@pytest.mark.anyio
async def test_health_route_still_available_after_v1_anpr_removal():
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
