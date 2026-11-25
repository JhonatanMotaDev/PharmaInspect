"""Treino do YOLO11-cls em GPU externa; nunca executa treino local por padrão."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pharma_inspect.pipeline import build_context, load_config
from pharma_inspect.yolo.model import resolve_path
from pharma_inspect.yolo.training import train_classifier


def parse_args() -> argparse.Namespace:
    """Define os argumentos da linha de comando.

    Returns:
        argparse.Namespace: Argumentos interpretados.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/default.yaml"))
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--data", default=None, help="Diretório raiz ou data.yaml exportado")
    parser.add_argument("--weights", default=None, help="Pesos iniciais do Ultralytics")
    parser.add_argument("--output-weights", default=None, help="Destino do melhor checkpoint")
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--project", default=None, help="Diretório temporário dos runs do Ultralytics")
    return parser.parse_args()


def _weight_source(value: str, root: Path) -> str:
    """Usa um arquivo local quando disponível ou um nome Ultralytics."""
    path = resolve_path(value, root)
    return str(path) if path.is_file() else value


def main() -> None:
    """Executa o treino e registra os parâmetros usados.

    Returns:
        None: O resumo é enviado para a saída padrão.
    """
    args = parse_args()
    config = load_config(args.config)
    # Com --data o dataset já foi exportado; não exigir manifesto nem frames
    # brutos na máquina que faz o treino.
    if args.manifest or not args.data:
        build_context(args.config, args.manifest)
    data_yaml = resolve_path(
        args.data or config["data"].get("yolo_dir", "data/yolo") + "/data.yaml", ROOT
    )
    model_config = config["model"]
    output_weights = resolve_path(
        args.output_weights or model_config["weights"], ROOT
    )
    device = args.device or config["training"].get("device", "cuda:0")
    result = train_classifier(
        config,
        data_yaml=data_yaml,
        pretrained_weights=_weight_source(
            args.weights or model_config["pretrained_weights"], ROOT
        ),
        output_weights=output_weights,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        imgsz=args.imgsz,
        project=resolve_path(
            args.project or config["training"].get("project", "artifacts/yolo/runs"),
            ROOT,
        ),
    )
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
