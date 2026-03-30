import pytest
from fastapi.testclient import TestClient
from backend.api_server import app
import backend.api_server as api_server
import unittest.mock as mock

# Reuse the FastAPI app through TestClient
client = TestClient(app)

# Initialize a mock predictor to avoid 503 errors
api_server.predictor = mock.Mock()

def test_health_endpoint():
    """Verify that the health check returns expected status and timestamp."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["healthy", "degraded"]
    assert "components" in data

def test_predict_endpoint_no_data():
    """Verify 404 behavior when a match_id does not exist in Redis."""
    # We mock the predictor to return an error dictionary
    api_server.predictor.predict_live_match.return_value = {"error": "No live data available"}
    
    response = client.get("/predict/invalid_match_id")
    assert response.status_code == 404
    assert response.json()["detail"] == "No live data available"

def test_list_matches_empty():
    """Verify match listing works even when Redis is empty."""
    with mock.patch("redis.Redis.keys") as mock_keys:
        mock_keys.return_value = []
        response = client.get("/matches")
        assert response.status_code == 200
        assert response.json() == []

@pytest.mark.asyncio
async def test_predict_success():
    """Verify 200 OK and JSON structure for a successful prediction flow."""
    mock_result = {
        "match_id": "test_match",
        "win_probability": 0.65,
        "confidence": 0.85,
        "balls_analyzed": 10
    }
    
    api_server.predictor.predict_live_match.return_value = mock_result
    
    response = client.get("/predict/test_match")
    assert response.status_code == 200
    assert response.json()["win_probability"] == 0.65
