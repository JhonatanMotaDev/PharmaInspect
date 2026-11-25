"""Exportação do manifesto ResiDual para o formato de dataset do YOLO-cls."""

from __future__ import annotations

import json
import os
import re
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..data import binary_label, label_name

SPLIT_NAMES = ("train", "val", "test")
CLASS_NAMES = ("Clean", "Contaminated")


@dataclass(frozen=True)
class Letterbox:
    """Geometria do letterbox quadrado usado na entrada do YOLO.

    Attributes:
        scale (float): Fator de escala aplicado à imagem original.
        fitted_width (int): Largura da imagem após escala, antes do padding.
        fitted_height (int): Altura da imagem após escala, antes do padding.
        offset_x (int): Deslocamento horizontal da imagem no canvas.
        offset_y (int): Deslocamento vertical da imagem no canvas.
        size (int): Lado do canvas quadrado.
    """

    scale: float
    fitted_width: int
    fitted_height: int
    offset_x: int
    offset_y: int
    size: int


def letterbox_geometry(width: int, height: int, imgsz: int) -> Letterbox:
    """Calcula a geometria do letterbox aplicado antes da inferência.

    Args:
        width (int): Largura original da imagem.
        height (int): Altura original da imagem.
        imgsz (int): Lado do canvas quadrado de entrada do modelo.

    Returns:
        Letterbox: Escala, dimensões encaixadas e deslocamentos.

    Raises:
        ValueError: Se as dimensões não forem positivas.
    """
    if width <= 0 or height <= 0 or imgsz <= 0:
        raise ValueError("width, height and imgsz must be positive")
    scale = min(imgsz / height, imgsz / width)
    fitted_width = max(1, round(width * scale))
    fitted_height = max(1, round(height * scale))
    return Letterbox(
        scale=scale,
        fitted_width=fitted_width,
        fitted_height=fitted_height,
        offset_x=(imgsz - fitted_width) // 2,
        offset_y=(imgsz - fitted_height) // 2,
        size=imgsz,
    )


def _safe_name(value: str) -> str:
    """Remove caracteres que não devem aparecer em nomes de arquivo.

    Args:
        value (str): Identificador original.

    Returns:
        str: Identificador seguro para nome de arquivo.
    """
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return cleaned.strip("._") or "sample"


def export_dataset(
    context: Mapping[str, Any],
    output_root: str | Path,
    *,
    use_symlinks: bool = True,
    max_frames_per_sequence: int | None = None,
) -> dict[str, Any]:
    """Escreve o dataset YOLO-cls a partir de um contexto já dividido.

    Os arquivos originais não são modificados. Por padrão, cada imagem é
    referenciada por um link simbólico absoluto; ``use_symlinks=False`` copia
    os arquivos, útil em ambientes sem suporte a links.

    Args:
        context (Mapping[str, Any]): Contexto com ``subsets`` e ``label_map``.
        output_root (str | Path): Diretório destino do dataset.
        use_symlinks (bool): Se deve criar links em vez de cópias.
        max_frames_per_sequence (int | None): Limite opcional de frames por
            sequência, útil para ensaios rápidos.

    Returns:
        dict[str, Any]: Contagens por split/classe, ordem das classes e
            parâmetros da exportação.

    Raises:
        ValueError: Se o contexto não tiver os subconjuntos esperados ou se
            ``max_frames_per_sequence`` não for positivo.
        FileExistsError: Se um nome de destino já existir apontando para outro
            arquivo, evitando sobrescrever evidências silently.
    """
    if max_frames_per_sequence is not None and max_frames_per_sequence <= 0:
        raise ValueError("max_frames_per_sequence must be positive")
    subsets = context["subsets"]
    if set(subsets) != set(SPLIT_NAMES):
        raise ValueError(f"context subsets must define exactly {SPLIT_NAMES}")
    label_map = context["label_map"]

    root = Path(output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    counts: dict[str, dict[str, int]] = {}
    link_total = 0

    for split in SPLIT_NAMES:
        for class_name in CLASS_NAMES:
            (root / split / class_name).mkdir(parents=True, exist_ok=True)

        split_counts = {class_name: 0 for class_name in CLASS_NAMES}
        for sample in subsets[split]:
            code = binary_label(sample.raw_label, label_map)
            class_name = label_name(code)
            frame_paths: Sequence[Path] = sample.frame_paths
            if max_frames_per_sequence is not None:
                frame_paths = frame_paths[:max_frames_per_sequence]
            for index, source in enumerate(frame_paths):
                source_path = Path(source).resolve()
                if not source_path.is_file():
                    raise FileNotFoundError(source_path)
                destination = (
                    root
                    / split
                    / class_name
                    / f"{_safe_name(sample.sample_id)}__{index:04d}{source_path.suffix.lower()}"
                )
                if destination.exists() or destination.is_symlink():
                    if use_symlinks and destination.is_symlink():
                        if destination.resolve() == source_path:
                            continue
                    raise FileExistsError(f"Destination already exists: {destination}")
                if use_symlinks:
                    os.symlink(source_path, destination)
                else:
                    shutil.copy2(source_path, destination)
                split_counts[class_name] += 1
                link_total += 1
        counts[split] = split_counts

    data_yaml = {
        "path": str(root),
        "train": "train",
        "val": "val",
        "test": "test",
        "nc": len(CLASS_NAMES),
        "names": list(CLASS_NAMES),
    }
    yaml_path = root / "data.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {data_yaml['path']}",
                f"train: {data_yaml['train']}",
                f"val: {data_yaml['val']}",
                f"test: {data_yaml['test']}",
                f"nc: {data_yaml['nc']}",
                "names:",
                *[f"  - {name}" for name in data_yaml["names"]],
                "",
            ]
        ),
        encoding="utf-8",
    )
    provenance = {
        "source": "ResiDual manifest exported by scripts/export_yolo_dataset.py",
        "data_yaml": str(yaml_path),
        "class_order": list(CLASS_NAMES),
        "counts": counts,
        "links": link_total,
        "use_symlinks": bool(use_symlinks),
        "max_frames_per_sequence": max_frames_per_sequence,
        "seed": context.get("config", {}).get("seed"),
        "imgsz": context.get("config", {}).get("model", {}).get("imgsz"),
    }
    (root / "export_manifest.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return provenance
