# -*- coding: utf-8 -*-
import pytest
from fastapi.testclient import TestClient
from app.main import app

@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c

def test_root_endpoint(client):
    response = client.get("/", headers={"Accept": "application/json"})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "version" in data

def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_metrics_endpoint(client):
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "puijai_incoming_webhooks_total" in response.text

def test_liff_endpoint(client):
    response = client.get("/liff")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "canvas-container" in response.text

def test_pdpa_delete_endpoint(client):
    response = client.delete("/api/users/U_TEST_USER_PDPA/data")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "details" in data

