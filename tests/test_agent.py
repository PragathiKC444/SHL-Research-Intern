from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_chat_clarifies_vague_request() -> None:
    response = client.post("/chat", json={"messages": [{"role": "user", "content": "I need an assessment"}]})
    assert response.status_code == 200
    payload = response.json()
    assert payload["recommendations"] == []
    assert payload["end_of_conversation"] is False
    assert "role" in payload["reply"].lower() or "measure" in payload["reply"].lower()


def test_chat_recommends_java_assessments() -> None:
    response = client.post(
        "/chat",
        json={
            "messages": [
                {"role": "user", "content": "Hiring a mid-level Java developer with stakeholder interaction. Include personality tests."}
            ]
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert 1 <= len(payload["recommendations"]) <= 10
    assert any("java" in item["name"].lower() for item in payload["recommendations"])


def test_chat_compares_catalog_items() -> None:
    response = client.post(
        "/chat",
        json={
            "messages": [
                {"role": "user", "content": "What is the difference between OPQ and GSA?"}
            ]
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["recommendations"] == []
    assert "vs" in payload["reply"].lower() or "difference" in payload["reply"].lower()


def test_chat_refuses_off_topic_request() -> None:
    response = client.post(
        "/chat",
        json={
            "messages": [
                {"role": "user", "content": "What salary should I offer a Java engineer in London?"}
            ]
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["recommendations"] == []
    assert "shl assessment" in payload["reply"].lower() or "shl" in payload["reply"].lower()
