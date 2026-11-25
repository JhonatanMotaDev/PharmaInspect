"""Exporta o manifesto ResiDual para o formato YOLO-cls (somente leitura na fonte)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pharma_inspect.pipeline import build_context
from pharma_inspect.yolo.dataset import export_dataset


def parse_args() -> argparse.Namespace:
    """Define os argumentos da linha de comando.

    Returns:
        argparse.Namespace: Argumentos interpretados.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/default.yaml"))
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--output", default=None, help="Diretório do dataset YOLO-cls")
    parser.add_argument("--copy", action="store_true", help="Copia em vez de criar links")
    parser.add_argument("--max-frames-per-sequence", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    """Executa a exportação e imprime o resumo em JSON.

    Returns:
        None: O resumo é enviado para a saída padrão.
    """
    args = parse_args()
    context = build_context(args.config, args.manifest)
    output = args.output or context["config"].get("data", {}).get("yolo_dir", "data/yolo")
    output_path = Path(output)
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    summary = export_dataset(
        context,
        output_path,
        use_symlinks=not args.copy,
        max_frames_per_sequence=args.max_frames_per_sequence,
    )
    print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
