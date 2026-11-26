"""Verifica a integridade do dataset oficial na máquina de treino.

Recomputa os hashes registrados em ``data/dedup/MANIFEST.YAML``, confere os
links simbólicos e as contagens exportadas. Não altera nenhum arquivo.

Uso::

    .venv/bin/python scripts/verify_dataset.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("train", "val", "test")
CLASSES = ("Clean", "Contaminated")


def sha256(path: Path) -> str | None:
    """Calcula o SHA-256 de um arquivo, ou None quando ausente.

    Args:
        path (Path): Arquivo a ser lido.

    Returns:
        str | None: Digest hexadecimal, ou None se o arquivo não existir.
    """
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve(relative_path: str) -> Path:
    """Resolve um caminho do manifesto relativo à raiz do repositório.

    Args:
        relative_path (str): Caminho registrado no manifesto.

    Returns:
        Path: Caminho absoluto.
    """
    path = Path(relative_path)
    return path if path.is_absolute() else ROOT / path


def parse_args() -> argparse.Namespace:
    """Define os argumentos da linha de comando.

    Returns:
        argparse.Namespace: Argumentos interpretados.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(ROOT / "data/dedup/MANIFEST.YAML"))
    return parser.parse_args()


def main() -> None:
    """Executa a verificação e imprime o resultado.

    Returns:
        None: O resultado é enviado para a saída padrão.

    Raises:
        SystemExit: Com código 1 quando alguma checagem falha.
    """
    args = parse_args()
    manifest_path = resolve(args.manifest)
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    dataset_dir = manifest_path.parent

    paths = {
        "generator_script": ROOT / "scripts/build_dedup_dataset.py",
        "config": ROOT / "configs/default.yaml",
        "original_manifest": ROOT / "data/processed/residual_manifest.json",
        "raw_zip": ROOT / "data/raw/ResiDual_Dataset.zip",
        "portable_manifest": dataset_dir / "manifest.json",
        "data_yaml": dataset_dir / "data.yaml",
        "export_manifest": dataset_dir / "export_manifest.json",
    }
    hashes = {}
    for name, path in paths.items():
        actual = sha256(path)
        expected = manifest["hashes"].get(name)
        hashes[name] = {
            "expected": expected,
            "actual": actual,
            "ok": actual == expected,
            "optional": expected is None and actual is None,
        }

    broken = [
        str(link)
        for split in SPLITS
        for cls in CLASSES
        for link in (dataset_dir / split / cls).glob("*")
        if not link.resolve().is_file()
    ]
    counts = {
        f"{split}/{cls}": len(list((dataset_dir / split / cls).glob("*")))
        for split in SPLITS
        for cls in CLASSES
    }
    expected_counts = {
        f"{split}/{cls}": manifest["validation"]["export_counts"][split][cls]
        for split in SPLITS
        for cls in CLASSES
    }
    counts_ok = counts == expected_counts

    splits = {split: set() for split in SPLITS}
    for cluster in manifest["clusters"]:
        splits[cluster["split"]].add(cluster["id"])
    overlaps = [
        f"{left}/{right}"
        for index, left in enumerate(SPLITS)
        for right in SPLITS[index + 1 :]
        if splits[left] & splits[right]
    ]

    checks = {
        "hashes": all(item["ok"] for item in hashes.values()),
        "symlinks": not broken,
        "counts": counts_ok,
        "no_split_overlap": not overlaps,
        "no_cross_split_near_duplicates": manifest["validation"][
            "cross_split_near_duplicate_pairs"
        ]
        == 0,
    }
    result = {
        "ok": all(checks.values()),
        "checks": checks,
        "hashes": hashes,
        "counts": counts,
        "expected_counts": expected_counts,
        "broken_symlinks": broken,
        "split_overlaps": overlaps,
        "clusters": len(manifest["clusters"]),
    }
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    if not result["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
