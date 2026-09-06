#!/usr/bin/env python3
"""
backend/run.py – Convenience server launcher
=============================================
    python backend/run.py
    python backend/run.py --host 0.0.0.0 --port 8000 --reload
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is on path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Brain Tumour Segmentation API Server")
    parser.add_argument("--host",   default="0.0.0.0",            help="Bind host")
    parser.add_argument("--port",   default=8000, type=int,        help="Bind port")
    parser.add_argument("--reload", action="store_true",           help="Hot-reload (development)")
    parser.add_argument("--workers",default=1, type=int,           help="Number of workers")
    parser.add_argument("--log-level", default="info",
                        choices=["debug", "info", "warning", "error"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        import uvicorn
    except ImportError:
        print("uvicorn not installed. Run:  pip install uvicorn[standard]")
        sys.exit(1)

    uvicorn.run(
        "backend.app.main:app",
        host      = args.host,
        port      = args.port,
        reload    = args.reload,
        workers   = args.workers if not args.reload else 1,
        log_level = args.log_level,
    )


if __name__ == "__main__":
    main()
