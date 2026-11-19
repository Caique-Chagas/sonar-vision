"""Re-render immutable real inference caches and synthesize justified accessibility cues."""

import copy
import hashlib
import json
import shutil
import subprocess
from collections import Counter
from pathlib import Path

import cv2
import imageio_ffmpeg

from sonar_vision.desktop.evidence import PresentationEvidence
from sonar_vision.desktop.presentation import (
    Encoder,
    atomic_json,
    inference_pixels,
    inspect_media,
    verify_prepared,
)
from sonar_vision.desktop.profiles import Profile
from sonar_vision.desktop.render import render
from sonar_vision.speech.offline import synthesize_code
from sonar_vision.training.dataset import sha256

ROOT = Path("artifacts/demo/sonar-final-technical")
HISTORICAL = Path("artifacts/demo/sonar-video-v3/final-presentation")
SOURCES = {key: HISTORICAL / key for key in ["sao-paulo", "santos-a", "santos-b"]}


def prepare(key, previous):
    directory = ROOT / "presentation" / key
    if (directory / "manifest.json").exists():
        return verify_prepared(directory)
    if directory.exists():
        raise ValueError("Incomplete render exists; inspect before resuming, do not overwrite")
    original = verify_prepared(previous)
    assert original["binding"]["evidence_kind"] == "real_model_inference"
    records = [json.loads(s) for s in (previous / "predictions.jsonl").read_text().splitlines()]
    directory.mkdir(parents=True)
    binding = dict(
        original["binding"],
        parent_manifest=str(previous / "manifest.json"),
        parent_sha256=sha256(previous / "manifest.json"),
        boxes_origin="current_frame_detector",
        inference_reexecuted=False,
        textual_stability_is_not_line_verification=True,
        implementations={
            str(p): sha256(p)
            for p in [
                Path("src/sonar_vision/desktop/evidence.py"),
                Path("src/sonar_vision/desktop/render.py"),
                Path("src/sonar_vision/desktop/label_layout.py"),
            ]
        },
    )
    atomic_json(directory / "binding.json", binding)
    source = Path(original["spec"]["source"])
    assert sha256(source) == binding["source_sha256"]
    fps = original["annotated"]["fps"]
    cap = cv2.VideoCapture(str(source), cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_POS_FRAMES, original["spec"]["start_frame"])
    encoder = Encoder(directory / "annotated.mp4", 1280, 720, fps)
    profile = Profile(
        key,
        original["spec"]["city"],
        "V3 ônibus + BR V1 letreiro",
        "",
        str(source),
        960,
        "Vídeo pessoal do usuário",
        "Declarado",
        "ML autorizado",
        "Linha não verificada",
    )
    evidence = PresentationEvidence()
    derived = []
    events = []
    state_counts = Counter()
    layout_counts = Counter()
    try:
        for record in records:
            ok, raw = cap.read()
            assert ok
            frame = inference_pixels(raw)
            assert hashlib.sha256(frame.tobytes()).hexdigest() == record["input_sha256"]
            result = evidence.observe(record)
            state_counts[result["state"]] += 1
            if result["audio_event"]:
                events.append(result["audio_event"])
            view = dict(
                frame=frame,
                index=record["index"],
                row=record["row"],
                presentation=True,
                source_fps=fps,
                technical=True,
                evidence=result,
            )
            canvas = render(profile, view, "Pré-processado", False)
            layout = view["label_layout"]
            layout_counts.update(
                {k: layout[k] for k in ["skipped", "panel_occlusions", "label_overlaps"]}
            )
            derived.append(dict(copy.deepcopy(record), evidence=result, label_layout=layout))
            encoder.write(canvas)
            if record["index"] in {0, len(records) // 2, len(records) - 1} or result["audio_event"]:
                cv2.imwrite(str(directory / f"inspection-{record['index']:06d}.png"), canvas)
    finally:
        cap.release()
        encoder.close()
    shutil.copy2(previous / "reference.mp4", directory / "reference.mp4")
    with (directory / "predictions.jsonl").open("w") as stream:
        for record in derived:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    cues = []
    for event in events:
        audio = synthesize_code(event["code"], directory / "audio")
        cues.append(
            dict(
                event,
                audio_file="audio/" + audio["file"],
                audio_sha256=audio["sha256"],
                duration_seconds=audio["duration_seconds"],
            )
        )
    atomic_json(directory / "audio-events.json", cues)
    # Offset is source-derived clip time. Preserve duration and copy the video stream.
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-v",
        "error",
        "-nostdin",
        "-i",
        str(directory / "annotated.mp4"),
    ]
    if cues:
        for cue in cues:
            command += ["-i", str(directory / cue["audio_file"])]
        filters = []
        for i, cue in enumerate(cues):
            delay = round(cue["timestamp_seconds"] * 1000)
            filters.append(f"[{i + 1}:a]adelay={delay}:all=1[a{i}]")
        filters.append(
            "".join(f"[a{i}]" for i in range(len(cues)))
            + f"amix=inputs={len(cues)}:normalize=0,apad[mix]"
        )
        command += ["-filter_complex", ";".join(filters), "-map", "0:v:0", "-map", "[mix]"]
    else:
        command += [
            "-f",
            "lavfi",
            "-i",
            "anullsrc=r=22050:cl=mono",
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
        ]
    command += [
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "96k",
        "-t",
        str(original["duration_seconds"]),
        "-movflags",
        "+faststart",
        str(directory / "annotated-audio.mp4"),
    ]
    subprocess.run(command, check=True, timeout=60, capture_output=True)
    assert inspect_media(directory / "annotated.mp4")["frames"] == len(records)
    assert inspect_media(directory / "annotated-audio.mp4")["frames"] == len(records)
    files = [p for p in directory.rglob("*") if p.is_file() and p.suffix != ".log"]
    manifest = dict(
        original,
        binding=binding,
        files=[dict(name=str(p.relative_to(directory)), sha256=sha256(p)) for p in files],
        temporal_states=dict(state_counts),
        label_layout=dict(layout_counts),
        audio_events=len(cues),
        audio_events_file="audio-events.json",
        line_verified=False,
        temporal_confirmation=False,
        annotation_origin="detector_only",
        inference_reexecuted=False,
        all_frames_inferred=True,
    )
    atomic_json(directory / "manifest.json", manifest)
    print(
        key,
        json.dumps(dict(states=state_counts, labels=layout_counts, cues=cues), ensure_ascii=False),
        flush=True,
    )
    return manifest


def main():
    ROOT.mkdir(exist_ok=True)
    result = {key: prepare(key, path) for key, path in SOURCES.items()}
    atomic_json(
        ROOT / "presentation-summary.json",
        {
            k: dict(
                states=v["temporal_states"],
                counts=v["counts"],
                layout=v["label_layout"],
                audio_events=v["audio_events"],
            )
            for k, v in result.items()
        },
    )


if __name__ == "__main__":
    main()
