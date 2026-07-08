"""Entry point: run the Solution Creation Tool web app.

    python main.py                  # start the server on http://127.0.0.1:8000
    uvicorn app.main:app --reload   # alternative, with autoreload
"""
from __future__ import annotations

import uvicorn


def main() -> None:
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()
