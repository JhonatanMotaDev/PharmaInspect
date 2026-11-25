"""Gera o manifesto do ResiDual a partir da estrutura de diretórios."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pharma_inspect.data import CLASS_NAMES, LABEL_MAP


def build_samples(dataset_root: Path) -> list[dict[str, object]]:
    """Cria uma amostra por diretório de sequência.

    Args:
        dataset_root (Path): Raiz do dataset original.

    Returns:
        list[dict[str, object]]: Amostras com rótulo, frames e notas.

    Raises:
        FileNotFoundError: Se alguma classe esperada não existir.
        ValueError: Se uma sequência não tiver frames JPEG.
    """
    samples: list[dict[str, object]] = []
    for class_name in CLASS_NAMES:
        class_dir = dataset_root / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(class_dir)
        for sequence_dir in sorted(path for path in class_dir.iterdir() if path.is_dir()):
            frame_paths = sorted(sequence_dir.glob("*.jpg"))
            if not frame_paths:
                raise ValueError(f"No JPEG frames in {sequence_dir}")
            duplicates = sorted(path.name for path in frame_paths if "(1)" in path.stem)
            notes = []
            if len(frame_paths) != 19:
                notes.append(f"frames={len(frame_paths)}; majority=19")
            if duplicates:
                notes.append(f"exact_duplicates={len(duplicates)}")
            # Caminhos relativos à raiz mantêm o manifesto portátil entre máquinas.
            stored_frames = []
            for path in frame_paths:
                resolved = path.resolve()
                stored_frames.append(
                    str(resolved.relative_to(ROOT)) if resolved.is_relative_to(ROOT) else str(resolved)
                )
            samples.append(
                {
                    "sample_id": sequence_dir.name,
                    "sequence_id": sequence_dir.name,
                    "raw_label": class_name,
                    "modality": "rgb",
                    "quality_label": "accepted",
                    "rationale": "sequence_directory",
                    "frame_paths": stored_frames,
                    "notes": "; ".join(notes),
                }
            )
    return samples


def build_manifest(dataset_root: Path) -> dict[str, object]:
    """Monta o documento JSON com amostras, rótulos e metadados.

    Args:
        dataset_root (Path): Raiz do dataset original.

    Returns:
        dict[str, object]: Manifesto completo.

    Raises:
        FileNotFoundError: Se alguma classe esperada não existir.
        ValueError: Se uma sequência não tiver frames JPEG.
    """
    samples = build_samples(dataset_root)
    counts = {
        class_name: sum(1 for sample in samples if sample["raw_label"] == class_name)
        for class_name in CLASS_NAMES
    }
    source_root = dataset_root.resolve()
    source = (
        str(source_root.relative_to(ROOT))
        if source_root.is_relative_to(ROOT)
        else str(source_root)
    )
    return {
        "metadata": {
            "source": source,
            "unit": "sequence",
            "label_map": dict(LABEL_MAP),
            "sequence_counts": counts,
            "binary_sequence_counts": {
                "Clean": counts["Clean"],
                "Contaminated": sum(counts[name] for name in CLASS_NAMES if name != "Clean"),
            },
            "total_sequences": len(samples),
            "total_frames": sum(len(sample["frame_paths"]) for sample in samples),
            "split_policy": "group_split by sequence, stratified by binary label (config: 70/15/15, seed=42)",
        },
        "label_map": dict(LABEL_MAP),
        "samples": samples,
    }


def main(argv=None) -> int:
    """Ponto de entrada do gerador de manifesto.

    Args:
        argv (Sequence[str] | None): Argumentos de linha de comando; quando
            ``None``, usa ``sys.argv``.

    Returns:
        int: Código de saída zero.

    Raises:
        FileNotFoundError: Se alguma classe esperada não existir.
        ValueError: Se uma sequência não tiver frames JPEG.
        OSError: Se o manifesto não puder ser gravado.
    """
    parser = argparse.ArgumentParser(description="Gera o manifesto do ResiDual para o PharmaInspect")
    parser.add_argument("--dataset-root", default=str(ROOT / "data/raw/ResiDual_Dataset"))
    parser.add_argument("--output", default=str(ROOT / "data/processed/residual_manifest.json"))
    args = parser.parse_args(argv)

    manifest = build_manifest(Path(args.dataset_root))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    metadata = manifest["metadata"]
    print(f"manifest: {output}")
    print(f"sequences: {metadata['total_sequences']}")
    print(f"frames: {metadata['total_frames']}")
    print(f"per class: {metadata['sequence_counts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
