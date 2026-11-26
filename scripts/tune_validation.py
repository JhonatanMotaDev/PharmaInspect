"""Varredura de limiar e agregação no conjunto de validação.

Gera os escores de validação, testa todas as combinações de agregação e limiar,
seleciona a melhor por balanced accuracy (desempate: recall de `Clean`,
acurácia e proximidade de 0,5) e grava ``validation.json``. A seleção oficial
só é válida quando executada com um checkpoint treinado no split deduplicado.

Uso::

    .venv/bin/python scripts/tune_validation.py --weights artifacts/yolo/yolo11n-cls-1280.pt

Para um ensaio com o checkpoint antigo, use ``--status dry_run``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pharma_inspect.data import binary_label  # noqa: E402
from pharma_inspect.evaluation import (  # noqa: E402
    STRATEGIES,
    aggregate,
    apply_threshold,
    compute_metrics,
    write_json_report,
)
from pharma_inspect.pipeline import build_context  # noqa: E402
from pharma_inspect.yolo.inference import predict_probs  # noqa: E402
from pharma_inspect.yolo.model import load_classifier, resolve_path  # noqa: E402

DEFAULT_THRESHOLDS = [round(float(value), 4) for value in np.arange(0.30, 0.96, 0.05)]


def sweep(
    truths: list[int],
    aggregated: dict[str, list[np.ndarray]],
    thresholds: list[float],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Avalia todas as combinações de agregação e limiar.

    Args:
        truths (list[int]): Rótulos binários por cluster.
        aggregated (dict): Escores agregados por estratégia.
        thresholds (list[float]): Limiares candidatos.

    Returns:
        tuple[list[dict[str, Any]], dict[str, Any]]: Linhas da varredura e
        a combinação selecionada.

    Raises:
        ValueError: Se os limiares estiverem fora de (0, 1).
    """
    if any(not 0.0 < value < 1.0 for value in thresholds):
        raise ValueError("thresholds must lie strictly between 0 and 1")
    rows: list[dict[str, Any]] = []
    for strategy, scores in aggregated.items():
        for threshold in thresholds:
            predictions = [apply_threshold(score, threshold) for score in scores]
            metrics = compute_metrics(truths, predictions)
            recall_clean = metrics["per_class"]["0"]["recall"]
            recall_contaminated = metrics["per_class"]["1"]["recall"]
            rows.append(
                {
                    "aggregation": strategy,
                    "threshold": threshold,
                    "accuracy": round(float(metrics["accuracy"]), 6),
                    "recall_clean": round(float(recall_clean), 6),
                    "recall_contaminated": round(float(recall_contaminated), 6),
                    "balanced_accuracy": round(float((recall_clean + recall_contaminated) / 2), 6),
                    "confusion_matrix": metrics["confusion_matrix"],
                }
            )
    chosen = max(
        rows,
        key=lambda row: (
            row["balanced_accuracy"],
            row["recall_clean"],
            row["accuracy"],
            -abs(row["threshold"] - 0.5),
        ),
    )
    return rows, dict(chosen)


def parse_args() -> argparse.Namespace:
    """Define os argumentos da linha de comando.

    Returns:
        argparse.Namespace: Argumentos interpretados.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/default.yaml"))
    parser.add_argument("--weights", default=None, help="Checkpoint usado na varredura")
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--status", choices=("official", "dry_run"), default="official")
    parser.add_argument("--output", default=str(ROOT / "artifacts/yolo_reports/validation.json"))
    parser.add_argument("--limit-clusters", type=int, default=None)
    parser.add_argument(
        "--thresholds",
        default=",".join(str(value) for value in DEFAULT_THRESHOLDS),
        help="Limiares candidatos separados por vírgula",
    )
    return parser.parse_args()


def main() -> None:
    """Executa a varredura e grava o relatório de validação.

    Returns:
        None: O resumo é enviado para a saída padrão.
    """
    args = parse_args()
    context = build_context(args.config)
    config = context["config"]
    samples = list(context["subsets"]["val"])
    if args.limit_clusters is not None:
        samples = samples[: args.limit_clusters]
    if not samples:
        raise SystemExit("Validation subset is empty")

    inference_config = config.get("inference", {})
    imgsz = int(args.imgsz or inference_config.get("imgsz", 1280))
    device = args.device or inference_config.get("device", "cpu")
    weights_value = args.weights or config["model"]["weights"]
    weights_path = resolve_path(weights_value, ROOT)
    model = load_classifier(config, weights=weights_value, device=device, root=ROOT)

    truths: list[int] = []
    frame_scores: list[list[float]] = []
    cluster_frame_scores: list[list[list[float]]] = []
    for sample in samples:
        truths.append(binary_label(sample.raw_label, context["label_map"]))
        scores: list[list[float]] = []
        for path in sample.frame_paths:
            with Image.open(path) as image:
                score = predict_probs(model, image.convert("RGB"), imgsz=imgsz, device=device)
            values = score.tolist()
            scores.append(values)
            frame_scores.append(values)
        cluster_frame_scores.append(scores)

    strategies_scores = {
        strategy: [aggregate(scores, strategy) for scores in cluster_frame_scores]
        for strategy in STRATEGIES
    }
    rows, chosen = sweep(truths, strategies_scores, [float(v) for v in args.thresholds.split(",")])

    report = {
        "status": args.status,
        "must_rerun_after_training": args.status != "official",
        "warning": (
            "Checkpoint antigo: os clusters de validação têm vizinhos quase idênticos no "
            "treino antigo. Esta varredura demonstra o protocolo, mas NÃO define a configuração "
            "oficial. Repetir depois do treino no dataset deduplicado."
            if args.status == "dry_run"
            else None
        ),
        "protocol": {
            "split": "val",
            "clusters": len(samples),
            "frames": len(frame_scores),
            "imgsz": imgsz,
            "device": device,
            "preprocessing": inference_config.get("preprocessing", "letterbox"),
            "max_frames_per_cluster": None,
            "weights": str(weights_path),
            "weights_sha256": hashlib.sha256(weights_path.read_bytes()).hexdigest(),
            "dataset_manifest": str(config["data"]["manifest"]),
            "dataset_manifest_sha256": hashlib.sha256(
                resolve_path(config["data"]["manifest"], ROOT).read_bytes()
            ).hexdigest(),
            "threshold_grid": [float(v) for v in args.thresholds.split(",")],
            "aggregations": list(STRATEGIES),
            "selection_rule": "max balanced_accuracy, then recall_clean, accuracy, proximity to 0.5",
        },
        "selection": chosen,
        "sweep": rows,
        "scores": {
            "clusters": {name: [score.tolist() for score in values] for name, values in strategies_scores.items()},
            "frames": frame_scores,
        },
    }
    output = write_json_report(args.output, report)
    print(
        json.dumps(
            {
                "report": str(output),
                "status": args.status,
                "selected": chosen,
                "clusters": len(samples),
                "frames": len(frame_scores),
            },
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
