import pytest
from fastapi.testclient import TestClient

from triton_server.app import app


def test_v1_anpr_routes_are_not_registered():
    client = TestClient(app)

    response = client.post("/v1/anpr_object/process-image", json={})

    assert response.status_code == 404


def test_v2_anpr_route_is_registered():
    client = TestClient(app)
    response = client.get("/openapi.json")

    assert response.status_code == 200
    openapi = response.json()
    assert "/anpr/v2/anpr_object/process" in openapi["paths"]
    assert "/v1/anpr_object/process-image" not in openapi["paths"]
    assert "/v1/anpr_object/process-video" not in openapi["paths"]


@pytest.mark.anyio
async def test_health_route_still_available():
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
