"""CLI entrypoint for (re-)indexing Qdrant. The actual logic lives in
app/indexing.py, since app/main.py's startup check needs to call it too
(auto-index on first `docker compose up` if the collections are empty).

Usage: python scripts/index.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.indexing import index_all

if __name__ == "__main__":
    index_all()
    print("Done.")
