"""Optional authenticated export; SDK output suppressed to avoid signed-link disclosure."""

import argparse
import contextlib
import io
import os
from pathlib import Path

from dotenv import load_dotenv


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", default="s-workspace-b9ro3")
    parser.add_argument("--project", default="bus-route-number-zitdp")
    parser.add_argument("--version", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("data/roboflow-export"))
    args = parser.parse_args()
    key = os.getenv("ROBOFLOW_API_KEY")
    if not key:
        parser.error("Configure ROBOFLOW_API_KEY no ambiente ou .env; nunca no chat")
    if args.output.exists():
        parser.error("Destino já existe; use outro caminho")
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            from roboflow import Roboflow

            project = Roboflow(api_key=key).workspace(args.workspace).project(args.project)
            project.version(args.version).download("yolov8", location=str(args.output))
    except Exception:
        raise SystemExit(
            "Exportação falhou; confira acesso, licença e limites na conta Roboflow"
        ) from None
    print(f"Exportação em {args.output}; audite labels e grupos antes de preparar splits.")


if __name__ == "__main__":
    main()
