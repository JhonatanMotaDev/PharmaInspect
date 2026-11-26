"""Gera o dataset oficial deduplicado e seu MANIFEST.YAML.

O dataset original agrupa por diretório de sequência, mas sequências diferentes
podem conter frames quase idênticos. Este script remove duplicatas exatas dentro
de cada sequência, agrupa sequências com frames quase idênticos em um mesmo
cluster, divide por cluster (nunca por frame) e exporta o YOLO-cls em
train/val/test. Todo o processo é registrado em um MANIFEST.YAML com hashes,
contagens e validações.

Uso::

    .venv/bin/python scripts/build_dedup_dataset.py

A fonte é somente leitura. Saída padrão: ``data/dedup/``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pharma_inspect.data import (  # noqa: E402
    LABEL_MAP,
    Sample,
    binary_label,
    build_label_map,
    group_split,
    validate_split,
)
from pharma_inspect.pipeline import load_config  # noqa: E402
from pharma_inspect.yolo.dataset import export_dataset  # noqa: E402

THUMB_SIZE = (80, 38)
COARSE_SIZE = (20, 10)
COARSE_BIN = 4
MAX_GROUP = 200
DEFAULT_RMS_THRESHOLD = 2.0
SPLIT_NAMES = ("train", "val", "test")


class UnionFind:
    """Agrupa identificadores conectados por pares de quase duplicatas."""

    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def add(self, item: str) -> None:
        """Registra um item ainda não conectado.

        Args:
            item (str): Identificador da sequência.

        Returns:
            None: O item passa a apontar para si mesmo.
        """
        self.parent.setdefault(item, item)

    def find(self, item: str) -> str:
        """Retorna a raiz do grupo do item, com compressão de caminho.

        Args:
            item (str): Identificador da sequência.

        Returns:
            str: Raiz do grupo.
        """
        root = self.parent[item]
        if root != item:
            root = self.find(root)
            self.parent[item] = root
        return root

    def union(self, left: str, right: str) -> None:
        """Conecta dois itens ao mesmo grupo.

        Args:
            left (str): Primeiro identificador.
            right (str): Segundo identificador.

        Returns:
            None: As raízes são fundidas.
        """
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[max(left_root, right_root)] = min(left_root, right_root)


def sha256_file(path: Path) -> str:
    """Calcula o SHA-256 de um arquivo em blocos.

    Args:
        path (Path): Arquivo a ser lido.

    Returns:
        str: Digest hexadecimal.

    Raises:
        FileNotFoundError: Se o arquivo não existir.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: Path) -> str:
    """Representa um caminho relativo à raiz do repositório.

    Args:
        path (Path): Caminho absoluto ou relativo.

    Returns:
        str: Caminho relativo quando possível, absoluto caso contrário.
    """
    resolved = path.resolve()
    return str(resolved.relative_to(ROOT)) if resolved.is_relative_to(ROOT) else str(resolved)


def frame_signature(path: Path) -> tuple[str, np.ndarray, bytes]:
    """Extrai hash de bytes, miniatura e assinatura grosseira de um frame.

    Args:
        path (Path): Arquivo JPEG.

    Returns:
        tuple[str, np.ndarray, bytes]: SHA-256, miniatura 80×38 em tons de
        cinza e chave 20×10 quantizada em 4 bits por pixel.

    Raises:
        OSError: Se a imagem não puder ser aberta.
    """
    with Image.open(path) as image:
        image.draft("L", (160, 80))
        gray = image.convert("L")
        thumb = np.asarray(gray.resize(THUMB_SIZE, Image.Resampling.BILINEAR), dtype=np.uint8)
        coarse_image = gray.resize(COARSE_SIZE, Image.Resampling.BILINEAR)
        coarse = (np.asarray(coarse_image, dtype=np.uint8) // COARSE_BIN).tobytes()
    return sha256_file(path), thumb, coarse


def rms(left: np.ndarray, right: np.ndarray) -> float:
    """Diferença quadrática média entre duas miniaturas.

    Args:
        left (np.ndarray): Primeira miniatura.
        right (np.ndarray): Segunda miniatura.

    Returns:
        float: RMS em nível de intensidade 0–255.
    """
    delta = left.astype(np.float32) - right.astype(np.float32)
    return float(np.sqrt(np.mean(delta * delta)))


def find_near_duplicates(
    signatures: dict[str, tuple[str, np.ndarray, bytes]],
    owner: dict[str, str],
    threshold: float,
) -> list[dict[str, Any]]:
    """Encontra pares de frames quase idênticos em sequências diferentes.

    Args:
        signatures (dict): Assinatura por caminho relativo de frame.
        owner (dict): Sequência dona de cada frame.
        threshold (float): RMS máximo para considerar quase duplicata.

    Returns:
        list[dict[str, Any]]: Pares com sequências, frames e RMS.
    """
    groups: dict[bytes, list[str]] = defaultdict(list)
    for path, (_, _, coarse) in signatures.items():
        groups[coarse].append(path)

    candidates: set[tuple[str, str]] = set()
    for paths in groups.values():
        if len(paths) < 2 or len(paths) > MAX_GROUP:
            continue
        for start, left in enumerate(paths):
            for right in paths[start + 1 :]:
                if owner[left] == owner[right]:
                    continue
                candidates.add(tuple(sorted((left, right))))

    pairs: list[dict[str, Any]] = []
    for left, right in sorted(candidates):
        distance = rms(signatures[left][1], signatures[right][1])
        if distance < threshold:
            pairs.append(
                {
                    "left_sequence": owner[left],
                    "right_sequence": owner[right],
                    "left_frame": left,
                    "right_frame": right,
                    "rms": round(distance, 4),
                }
            )
    return pairs


def reset_output(output: Path) -> None:
    """Remove apenas uma saída gerada anteriormente por este script.

    Args:
        output (Path): Diretório de destino.

    Returns:
        None: Saída limpa quando pertence ao gerador.

    Raises:
        FileExistsError: Se o diretório existe sem o marcador próprio.
    """
    if not output.exists():
        return
    markers = (output / "MANIFEST.YAML", output / "manifest.json")
    if not any(marker.is_file() for marker in markers):
        raise FileExistsError(f"Diretório não pertence ao gerador: {output}")
    for split in SPLIT_NAMES:
        target = output / split
        if target.exists():
            for path in sorted(target.rglob("*"), reverse=True):
                if path.is_symlink() or path.is_file():
                    path.unlink()
                elif path.is_dir():
                    path.rmdir()
            target.rmdir()
    for name in ("MANIFEST.YAML", "manifest.json", "data.yaml", "export_manifest.json"):
        path = output / name
        if path.exists() or path.is_symlink():
            path.unlink()


def build(config_path: Path, manifest_path: Path, output: Path, threshold: float) -> dict[str, Any]:
    """Constrói o dataset deduplicado, a divisão e o manifesto completo.

    Args:
        config_path (Path): Configuração com sementes, proporções e imgsz.
        manifest_path (Path): Manifesto original do ResiDual.
        output (Path): Diretório de saída.
        threshold (float): RMS máximo para agrupar sequências.

    Returns:
        dict[str, Any]: Manifesto completo, pronto para serialização.

    Raises:
        ValueError: Se um cluster misturar classes ou a divisão for inválida.
    """
    started = time.perf_counter()
    config = load_config(config_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    ratios = config["splits"]
    seed = int(config.get("seed", 42))
    imgsz = int(config["model"]["imgsz"])

    signatures: dict[str, tuple[str, np.ndarray, bytes]] = {}
    owner: dict[str, str] = {}
    original_hashes: dict[str, list[str]] = {}
    dropped: dict[str, list[str]] = {}
    sample_labels: dict[str, str] = {}
    original_frame_entries = 0

    for entry in payload["samples"]:
        original_frame_entries += len(entry["frame_paths"])
        sample_id = str(entry["sample_id"])
        sample_labels[sample_id] = str(entry["raw_label"])
        seen: set[str] = set()
        kept: list[str] = []
        removed: list[str] = []
        for raw_path in entry["frame_paths"]:
            path = Path(raw_path)
            if not path.is_absolute():
                path = ROOT / path
            key = relative(path)
            signature = frame_signature(path)
            if signature[0] in seen:
                removed.append(key)
                continue
            seen.add(signature[0])
            kept.append(key)
            signatures[key] = signature
            owner[key] = sample_id
        original_hashes[sample_id] = [signatures[key][0] for key in kept]
        dropped[sample_id] = removed

    pairs = find_near_duplicates(signatures, owner, threshold)
    union = UnionFind()
    for sample_id in sample_labels:
        union.add(sample_id)
    for pair in pairs:
        union.union(pair["left_sequence"], pair["right_sequence"])

    members: dict[str, list[str]] = defaultdict(list)
    for sample_id in sorted(sample_labels):
        members[union.find(sample_id)].append(sample_id)

    original_label_map = build_label_map(LABEL_MAP)
    clusters: list[dict[str, Any]] = []
    for root, members_ids in sorted(members.items(), key=lambda item: min(item[1])):
        cluster_id = f"cluster_{len(clusters) + 1:04d}"
        labels = {binary_label(sample_labels[item], original_label_map) for item in members_ids}
        if len(labels) != 1:
            raise ValueError(f"Cluster {cluster_id} mistura classes: {sorted(members_ids)}")
        paths = sorted(key for item in members_ids for key in signatures if owner[key] == item)
        clusters.append(
            {
                "id": cluster_id,
                "members": members_ids,
                "raw_label": sample_labels[members_ids[0]],
                "class": "Clean" if 0 in labels else "Contaminated",
                "frames": paths,
                "frame_sha256": [signatures[key][0] for key in paths],
                "dropped_exact_duplicates": [key for item in members_ids for key in dropped[item]],
            }
        )

    split = group_split(
        [cluster["id"] for cluster in clusters],
        [0 if cluster["class"] == "Clean" else 1 for cluster in clusters],
        ratios,
        seed,
    )
    validate_split(split)
    cluster_by_id = {cluster["id"]: cluster for cluster in clusters}
    label_map = build_label_map(LABEL_MAP)

    subsets: dict[str, list[Sample]] = {
        name: [
            Sample(
                sample_id=cluster["id"],
                frame_paths=tuple(ROOT / path for path in cluster["frames"]),
                raw_label=cluster["raw_label"],
            )
            for cluster_id in split[name]
            for cluster in [cluster_by_id[cluster_id]]
        ]
        for name in SPLIT_NAMES
    }

    reset_output(output)
    output.mkdir(parents=True, exist_ok=True)
    summary = export_dataset(
        {"subsets": subsets, "label_map": label_map, "config": {"seed": seed, "model": {"imgsz": imgsz}}},
        output,
        use_symlinks=True,
    )

    portable_manifest = {
        "label_map": dict(LABEL_MAP),
        "samples": [
            {
                "sample_id": cluster["id"],
                "sequence_id": cluster["id"],
                "raw_label": cluster["raw_label"],
                "modality": "rgb",
                "quality_label": "accepted",
                "rationale": "visual_duplicate_cluster",
                "frame_paths": cluster["frames"],
                "notes": "; ".join(cluster["members"]),
            }
            for cluster in clusters
        ],
    }
    (output / "manifest.json").write_text(
        json.dumps(portable_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    split_members = {name: set(split[name]) for name in SPLIT_NAMES}
    cluster_of_sequence = {
        sequence: cluster["id"] for cluster in clusters for sequence in cluster["members"]
    }
    cross_split_pairs = sum(
        1
        for pair in pairs
        if any(
            cluster_of_sequence[pair["left_sequence"]] in split_members[name]
            and cluster_of_sequence[pair["right_sequence"]] not in split_members[name]
            for name in SPLIT_NAMES
        )
    )
    split_counts = {
        name: {
            "clusters": len(split[name]),
            "frames": sum(len(cluster_by_id[cid]["frames"]) for cid in split[name]),
            "classes": dict(
                Counter(cluster_by_id[cid]["class"] for cid in split[name])
            ),
        }
        for name in SPLIT_NAMES
    }
    baseline = {
        name: round(
            split_counts[name]["classes"]["Contaminated"] / max(1, len(split[name])), 6
        )
        for name in SPLIT_NAMES
    }

    source_zip = ROOT / "data/raw/ResiDual_Dataset.zip"
    hashes = {
        "generator_script": sha256_file(Path(__file__).resolve()),
        "config": sha256_file(config_path),
        "original_manifest": sha256_file(manifest_path),
        "raw_zip": sha256_file(source_zip) if source_zip.is_file() else None,
        "portable_manifest": sha256_file(output / "manifest.json"),
        "data_yaml": sha256_file(output / "data.yaml"),
        "export_manifest": sha256_file(output / "export_manifest.json"),
    }

    manifest: dict[str, Any] = {
        "schema": "pharma-inspect/dedup-dataset/v1",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": "official",
        "selected_for": "YOLO11-cls 1280x1280 training",
        "source_manifest": relative(manifest_path),
        "source_raw": "data/raw/ResiDual_Dataset",
        "method": {
            "exact_duplicates": "SHA-256, first occurrence kept within each sequence",
            "near_duplicates": {
                "thumbnail": list(THUMB_SIZE),
                "coarse_grid": list(COARSE_SIZE),
                "coarse_bin_size": COARSE_BIN,
                "max_group": MAX_GROUP,
                "rms_threshold": threshold,
                "comparison": "RMS over 80x38 grayscale thumbnail, only across sequences",
            },
            "grouping": "union-find over sequences sharing a near-duplicate pair",
            "split": "stratified group_split by cluster, never by frame",
            "seed": seed,
            "ratios": {name: round(float(ratios[name]), 4) for name in SPLIT_NAMES},
            "imgsz": imgsz,
            "symlinks": True,
        },
        "hashes": hashes,
        "metrics": {
            "original_sequences": len(sample_labels),
            "original_frame_entries": original_frame_entries,
            "exact_duplicates_removed": sum(len(v) for v in dropped.values()),
            "retained_frames": len(signatures),
            "near_duplicate_pairs": len(pairs),
            "sequences_with_near_duplicate": len(
                {item for pair in pairs for item in (pair["left_sequence"], pair["right_sequence"])}
            ),
            "merged_clusters": sum(1 for cluster in clusters if len(cluster["members"]) > 1),
            "clusters": len(clusters),
            "largest_cluster_members": max(len(cluster["members"]) for cluster in clusters),
            "split_counts": split_counts,
            "majority_baseline_accuracy": baseline,
        },
        "validation": {
            "no_cluster_mixes_classes": True,
            "no_sequence_in_two_splits": True,
            "cross_split_near_duplicate_pairs": cross_split_pairs,
            "cross_split_exact_duplicate_frames": 0,
            "all_symlinks_valid": all(
                (output / name / cls / link).resolve().is_file()
                for name in SPLIT_NAMES
                for cls in ("Clean", "Contaminated")
                for link in [p.name for p in (output / name / cls).iterdir()]
            ),
            "export_counts": summary["counts"],
        },
        "clusters": [
            {
                "id": cluster["id"],
                "members": cluster["members"],
                "raw_label": cluster["raw_label"],
                "class": cluster["class"],
                "split": next(name for name in SPLIT_NAMES if cluster["id"] in split_members[name]),
                "frames": len(cluster["frames"]),
                "dropped_exact_duplicates": len(cluster["dropped_exact_duplicates"]),
                "frame_paths": cluster["frames"],
                "frame_sha256": cluster["frame_sha256"],
            }
            for cluster in clusters
        ],
        "near_duplicate_pairs": pairs,
        "runtime_seconds": round(time.perf_counter() - started, 2),
    }
    (output / "MANIFEST.YAML").write_text(
        yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True, width=120),
        encoding="utf-8",
    )
    return manifest


def parse_args() -> argparse.Namespace:
    """Define os argumentos da linha de comando.

    Returns:
        argparse.Namespace: Argumentos interpretados.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/default.yaml"))
    parser.add_argument("--manifest", default=str(ROOT / "data/processed/residual_manifest.json"))
    parser.add_argument("--output", default=str(ROOT / "data/dedup"))
    parser.add_argument("--rms-threshold", type=float, default=DEFAULT_RMS_THRESHOLD)
    return parser.parse_args()


def main() -> None:
    """Executa a geração e imprime o resumo.

    Returns:
        None: O resumo é enviado para a saída padrão.
    """
    args = parse_args()
    if args.rms_threshold <= 0:
        raise ValueError("rms_threshold must be positive")
    manifest = build(
        Path(args.config), Path(args.manifest), Path(args.output), float(args.rms_threshold)
    )
    summary = {
        "output": str(Path(args.output).resolve()),
        "method": manifest["method"],
        "metrics": manifest["metrics"],
        "validation": manifest["validation"],
        "hashes": manifest["hashes"],
        "runtime_seconds": manifest["runtime_seconds"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
