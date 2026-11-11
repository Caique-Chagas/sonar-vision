import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(prog="sonar-vision")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("hardware")
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--manifest", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--sequence-reviewed", action="store_true")
    training = commands.add_parser("train")
    training.add_argument("--data", type=Path, required=True)
    training.add_argument("--config", type=Path, default=Path("configs/training.yaml"))
    training.add_argument("--output", type=Path, required=True)
    training.add_argument("--device")
    training.add_argument("--smoke", action="store_true")
    training.add_argument("--resume", type=Path)
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("--data", type=Path, required=True)
    evaluation.add_argument("--weights", type=Path, required=True)
    evaluation.add_argument("--output", type=Path, required=True)
    evaluation.add_argument("--device", default="cpu")
    inference = commands.add_parser("infer")
    inference.add_argument("--source", required=True)
    inference.add_argument(
        "--local", action="store_true", help="CLI-only explicit local file access"
    )
    inference.add_argument("--route", required=True)
    inference.add_argument("--max-frames", type=int, default=150)
    args = parser.parse_args()
    if args.command == "hardware":
        from sonar_vision.training.runner import hardware

        print(json.dumps(hardware(), indent=2))
    elif args.command == "prepare":
        from sonar_vision.training.dataset import prepare_dataset

        result = prepare_dataset(
            args.manifest,
            args.output,
            split_assurance="sequence_reviewed"
            if args.sequence_reviewed
            else "origin_group_only_sequence_unknown",
        )
        print(json.dumps({k: v for k, v in result.items() if k != "records"}, indent=2))
    elif args.command == "train":
        from sonar_vision.training.runner import train

        print(
            train(
                args.data,
                args.config,
                args.output,
                device=args.device,
                smoke=args.smoke,
                resume=args.resume,
            )
        )
    elif args.command == "evaluate":
        from sonar_vision.training.runner import evaluate_detector

        print(
            json.dumps(
                evaluate_detector(args.weights, args.data, args.output, args.device), indent=2
            )
        )
    else:
        from sonar_vision.core.config import Settings
        from sonar_vision.detection.yolo import YoloSignDetector
        from sonar_vision.ocr.paddle import PaddleEngine
        from sonar_vision.processing.pipeline import Context, FrameProcessor
        from sonar_vision.processing.sources import VideoSource
        from sonar_vision.validation.normalizer import clean_code

        settings = Settings()
        processor = FrameProcessor(
            YoloSignDetector(settings.model_path),
            PaddleEngine(),
            Context(
                clean_code(args.route),
                "cli",
                "cli",
                "local_video" if args.local else "video_stream",
            ),
            settings,
        )
        with VideoSource(
            args.source, settings.allowed_hosts, local=args.local, max_frames=args.max_frames
        ) as source:
            for index, timestamp, frame in source:
                _, events = processor.process(frame, index, timestamp)
                for event in events:
                    print(event.model_dump_json())


if __name__ == "__main__":
    main()
