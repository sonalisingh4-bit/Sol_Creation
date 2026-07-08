"""Entry point: run the Solution Creation Tool web app.

    python main.py                  # start the server on http://127.0.0.1:8000
    uvicorn app.main:app --reload   # alternative, with autoreload

On hosts like Render, PORT is provided by the platform and the app must bind
0.0.0.0; locally we stay on 127.0.0.1:8000.
"""
from __future__ import annotations

import os

import uvicorn


def main() -> None:
    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0" if os.getenv("PORT") else "127.0.0.1")
    uvicorn.run("app.main:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
