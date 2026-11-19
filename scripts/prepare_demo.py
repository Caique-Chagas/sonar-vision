"""Prepare a bounded segment of an authorized user MP4, using the existing pipeline."""

import argparse
from pathlib import Path

from prepare_technical_presentation import ROOT, prepare

from sonar_vision.desktop.presentation import (
    BR,
    BUS,
    ClipSpec,
    atomic_json,
    inspect_media,
    prepare_clip,
    verify_prepared,
)
from sonar_vision.training.dataset import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["sao-paulo", "santos-a", "santos-b"], required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--start", type=float, default=0)
    parser.add_argument("--seconds", type=float, default=10)
    parser.add_argument(
        "--authorized", action="store_true", help="You have permission to process this MP4"
    )
    args = parser.parse_args()
    if not args.authorized:
        parser.error("Use --authorized only for media you may process")
    if not 0 < args.seconds <= 20 or args.start < 0:
        parser.error("Use a positive segment up to20seconds")
    if not BR.is_file() or not BUS.is_file():
        parser.error("First run: python scripts/download_runtime_models.py")
    meta = inspect_media(args.video)
    frames = round(args.seconds * meta["fps"])
    if frames > 1200:
        parser.error("Frame budget1200 exceeded")
    previous = Path("artifacts/demo/sonar-video-v3/final-presentation") / args.profile
    spec = ClipSpec(
        args.profile, str(args.video), round(args.start * meta["fps"]), frames, args.profile.upper()
    )
    if (previous / "manifest.json").exists():
        existing = verify_prepared(previous)
        if existing["spec"] != spec.__dict__:
            parser.error("Profile already prepared for another segment; preserve it")
    prepare_clip(spec, previous, progress=lambda message: print(message, flush=True))
    prepare(args.profile, previous)
    directories = {k: ROOT / "presentation" / k for k in ["sao-paulo", "santos-a", "santos-b"]}
    if all((d / "manifest.json").is_file() for d in directories.values()):
        for d in directories.values():
            verify_prepared(d)
        atomic_json(
            ROOT / "active.json",
            dict(
                status="TECHNICALLY_VALIDATED_PRIVATE",
                line_verified=False,
                manifests={k: sha256(d / "manifest.json") for k, d in directories.items()},
            ),
        )
        print("Three prepared profiles are ready: ./scripts/demo_final.sh")
    else:
        print("Profile prepared. Prepare the other two profiles to activate all desktop commands.")
        print(
            "For this single raw profile: ./scripts/demo_final.sh "
            + args.profile
            + " --presentation-version historical"
        )


if __name__ == "__main__":
    main()
