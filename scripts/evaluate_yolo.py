"""Avaliação do YOLO11-cls por frame e por sequência no conjunto de teste."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pharma_inspect.data import binary_label
from pharma_inspect.evaluation import (
    aggregate,
    apply_threshold,
    cluster_bootstrap_intervals,
    compute_metrics,
    metrics_intervals,
    performance_metrics,
    write_json_report,
)
from pharma_inspect.pipeline import build_context
from pharma_inspect.yolo.inference import predict_probs
from pharma_inspect.yolo.model import load_classifier, resolve_path


def parse_args() -> argparse.Namespace:
    """Define os argumentos da linha de comando.

    Returns:
        argparse.Namespace: Argumentos interpretados.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/default.yaml"))
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--weights", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--limit-sequences", type=int, default=None)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--output", default=None, help="Arquivo JSON de saída")
    return parser.parse_args()


def main() -> None:
    """Executa a avaliação e grava o relatório JSON.

    Returns:
        None: O relatório é gravado em ``evaluation.reports_dir``.
    """
    args = parse_args()
    context = build_context(args.config, args.manifest)
    config = context["config"]
    label_map = context["label_map"]
    samples = list(context["subsets"]["test"])
    if args.limit_sequences is not None:
        samples = samples[: args.limit_sequences]
    if not samples:
        raise SystemExit("Test subset is empty")

    inference_config = config.get("inference", {})
    imgsz = args.imgsz if args.imgsz is not None else int(inference_config.get("imgsz", 1280))
    device = args.device or inference_config.get("device", "cpu")
    max_frames_value = inference_config.get("max_frames")
    max_frames = int(max_frames_value) if max_frames_value is not None else None
    if imgsz <= 0 or (max_frames is not None and max_frames <= 0):
        raise ValueError("imgsz must be positive and max_frames, when set, must be positive")
    if args.limit_sequences is not None and args.limit_sequences <= 0:
        raise ValueError("limit_sequences must be positive")
    strategy = inference_config.get("aggregation", "mean")
    threshold = float(inference_config.get("threshold", 0.5))
    model = load_classifier(config, weights=args.weights, device=device, root=ROOT)

    frame_scores = []
    frame_truths = []
    frame_predictions = []
    frame_groups = []
    sequence_scores = []
    sequence_truths = []
    sequence_predictions = []
    latencies = []

    for index, sample in enumerate(samples):
        truth = binary_label(sample.raw_label, label_map)
        scores = []
        frame_paths = sample.frame_paths if max_frames is None else sample.frame_paths[:max_frames]
        for path in frame_paths:
            with Image.open(path) as image:
                started = time.perf_counter()
                score = predict_probs(model, image.convert("RGB"), imgsz=imgsz, device=device)
                latencies.append((time.perf_counter() - started) * 1000.0)
            scores.append(score)
            frame_truths.append(truth)
            frame_groups.append(index)
            frame_predictions.append(apply_threshold(score, threshold))
            frame_scores.append(score.tolist())
        aggregated = aggregate(scores, strategy)
        sequence_scores.append(aggregated.tolist())
        sequence_truths.append(truth)
        sequence_predictions.append(apply_threshold(aggregated, threshold))

    report = {
        "protocol": {
            "task": config["model"].get("variant", "yolo11n-cls"),
            "split": "test",
            "aggregation": strategy,
            "threshold": threshold,
            "imgsz": imgsz,
            "device": device,
            "max_frames_per_sequence": max_frames,
            "sequences": len(samples),
            "frames": len(frame_scores),
            "weights": str(resolve_path(args.weights or config["model"]["weights"], ROOT)),
            "preprocessing": inference_config.get("preprocessing", "letterbox"),
            "dataset_manifest": str(config["data"]["manifest"]),
            "dataset_manifest_sha256": hashlib.sha256(
                resolve_path(config["data"]["manifest"], ROOT).read_bytes()
            ).hexdigest(),
        },
        "metrics": compute_metrics(sequence_truths, sequence_predictions),
        "metrics_ci95": metrics_intervals(sequence_truths, sequence_predictions),
        "frame_metrics": compute_metrics(frame_truths, frame_predictions),
        "frame_metrics_ci95": cluster_bootstrap_intervals(
            frame_truths,
            frame_predictions,
            frame_groups,
            n_boot=args.n_bootstrap,
            seed=int(config.get("seed", 42)),
        ),
        "performance": performance_metrics(latencies),
        "latencies_ms": [round(value, 4) for value in latencies],
        "scores": {
            "sequences": sequence_scores,
            "frames": frame_scores,
        },
    }
    reports_dir = Path(config.get("evaluation", {}).get("reports_dir", "artifacts/yolo_reports"))
    output = Path(args.output) if args.output else reports_dir / "test.json"
    if not output.is_absolute():
        output = ROOT / output
    path = write_json_report(output, report)
    print(json.dumps({"report": str(path), "summary": report["metrics"]}, indent=2,
                     sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
