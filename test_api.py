"""Простой смоук-тест REST API.

Использование:
    python scripts/test_api.py [BASE_URL]

По умолчанию BASE_URL = http://localhost:8000
Проверяет эндпоинты /health, /predict, /predict/batch.
"""

import json
import sys
import urllib.request

BASE_URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"


def _request(method: str, path: str, payload: dict | None = None) -> dict:
    url = f"{BASE_URL}{path}"
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        url, data=data, method=method, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode())


def main() -> int:
    print(f"Тестируем API по адресу {BASE_URL}\n")

    print("GET /health")
    health = _request("GET", "/health")
    print(json.dumps(health, ensure_ascii=False, indent=2))
    if not health.get("model_loaded"):
        print("\nМодель не загружена — дальнейшие тесты пропущены.")
        return 1

    print("\nPOST /predict")
    single = _request(
        "POST", "/predict", {"text": "This movie was absolutely fantastic and touching!"}
    )
    print(json.dumps(single, ensure_ascii=False, indent=2))

    print("\nPOST /predict/batch")
    batch = _request(
        "POST",
        "/predict/batch",
        {
            "texts": [
                "One of the best films I have ever seen.",
                "Terrible plot and boring acting, a waste of time.",
            ]
        },
    )
    print(json.dumps(batch, ensure_ascii=False, indent=2))

    print("\nВсе запросы выполнены успешно.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
