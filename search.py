"""
Entry point kept for convenience. The actual logic now lives under src/bubbola_gare/.

Usage examples:
    PYTHONPATH=src python search.py ingest
    PYTHONPATH=src python search.py summarize
    PYTHONPATH=src python search.py embed-summary
    PYTHONPATH=src python search.py search "your query here"
"""

from bubbola_gare.pipeline import main


if __name__ == "__main__":
    main()
