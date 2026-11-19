"""Run real BR V1/OCR on an authorized image; no transcript or manual ROI input."""

import argparse
import json
from pathlib import Path

from sonar_vision.desktop.presentation import BR
from sonar_vision.processing.local_video import create_demo_analyzer
from sonar_vision.processing.photograph import analyze_photograph, decode_photograph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=Path("examples/campinas.jpg"))
    parser.add_argument("--weights", type=Path, default=BR)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.weights.is_file():
        parser.error("Download verified runtime models first")
    if args.output.exists():
        parser.error("Output exists; preserve it and choose a new directory")
    image = decode_photograph(args.image)
    analyzer = create_demo_analyzer(args.weights, 640, static=True)
    args.output.mkdir(parents=True)
    report = analyze_photograph(image, args.output, analyzer)
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
