"""Carregamento, rótulos, divisão e amostragem de frames do ResiDual."""

from __future__ import annotations

import json
import random
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CLEAN = "Clean"
CONTAMINATED = "Contaminated"
BINARY_CLASSES = (CLEAN, CONTAMINATED)
CLASS_NAMES = (CLEAN, "Glass", "Plastic", "Rubber", "Sand")
LABEL_MAP = {name: CLEAN if name == CLEAN else CONTAMINATED for name in CLASS_NAMES}
SPLIT_NAMES = ("train", "val", "test")
REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Sample:
    """Uma sequência rotulada e seus frames.

    Attributes:
        sample_id (str): Identificador da sequência e da amostra.
        frame_paths (tuple[Path, ...]): Caminhos dos frames, absolutos ou
            relativos à raiz do projeto.
        raw_label (str): Rótulo original da classe.
    """

    sample_id: str
    frame_paths: tuple[Path, ...]
    raw_label: str


@dataclass(frozen=True)
class DatasetIndex:
    """Índice validado de amostras.

    Attributes:
        samples (tuple[Sample, ...]): Amostras carregadas do manifesto.
        label_map (dict[str, int]): Mapa de rótulo original para código binário.
    """

    samples: tuple[Sample, ...]
    label_map: dict[str, int]


def build_label_map(raw_to_binary: Mapping[str, str]) -> dict[str, int]:
    """Converte rótulos originais para os códigos binários 0 e 1.

    Args:
        raw_to_binary (Mapping[str, str]): Mapa de rótulo original para
            ``Clean`` ou ``Contaminated``.

    Returns:
        dict[str, int]: Mapa de rótulo original para 0 ou 1.

    Raises:
        ValueError: Se houver rótulo binário inválido ou se alguma classe
            binária estiver ausente.
    """
    invalid = {raw: value for raw, value in raw_to_binary.items() if value not in BINARY_CLASSES}
    if invalid:
        raise ValueError(f"Unsupported binary labels: {invalid}")
    present = set(raw_to_binary.values())
    if present != set(BINARY_CLASSES):
        raise ValueError(f"Label map must contain both binary classes, got {sorted(present)}")
    return {raw: int(value == CONTAMINATED) for raw, value in raw_to_binary.items()}


def binary_label(raw_label: str, label_map: Mapping[str, int]) -> int:
    """Retorna o código binário de um rótulo original.

    Args:
        raw_label (str): Rótulo original presente no manifesto.
        label_map (Mapping[str, int]): Mapa validado de rótulo para código.

    Returns:
        int: 0 para ``Clean`` e 1 para ``Contaminated``.

    Raises:
        KeyError: Se ``raw_label`` não estiver em ``label_map``.
    """
    try:
        return label_map[raw_label]
    except KeyError as exc:
        raise KeyError(f"Raw label {raw_label!r} is absent from the validated label map") from exc


def label_name(code: int) -> str:
    """Retorna o nome textual de um código binário.

    Args:
        code (int): Código 0 ou 1.

    Returns:
        str: ``Clean`` para 0 e ``Contaminated`` para 1.

    Raises:
        ValueError: Se ``code`` não for 0 ou 1.
    """
    if code not in (0, 1):
        raise ValueError(f"Binary code must be 0 or 1, got {code}")
    return BINARY_CLASSES[code]


def load_index(manifest_path: str | Path) -> DatasetIndex:
    """Lê um manifesto JSON e devolve o índice de amostras.

    Args:
        manifest_path (str | Path): Caminho do manifesto gerado.

    Returns:
        DatasetIndex: Índice com amostras e mapa de rótulos.

    Raises:
        ValueError: Se uma amostra não tiver frames.
        OSError: Se o arquivo não puder ser lido.
        json.JSONDecodeError: Se o conteúdo não for JSON válido.

    Note:
        Caminhos relativos no manifesto são resolvidos a partir da raiz do
        repositório, mantendo o arquivo portátil entre máquinas.
    """
    payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    samples = []
    for entry in payload["samples"]:
        frame_paths = tuple(
            path if path.is_absolute() else REPO_ROOT / path
            for path in (Path(item) for item in entry["frame_paths"])
        )
        if not frame_paths:
            raise ValueError(f"Sample {entry['sample_id']!r} has no frames")
        samples.append(
            Sample(
                sample_id=str(entry["sample_id"]),
                frame_paths=frame_paths,
                raw_label=str(entry["raw_label"]),
            )
        )
    return DatasetIndex(tuple(samples), dict(payload["label_map"]))


def sample_group(sample: Any) -> str:
    """Retorna o identificador de grupo de uma amostra.

    Args:
        sample (Any): Amostra com ``sample_id``.

    Returns:
        str: Identificador da sequência, usado na divisão por grupo.
    """
    return sample.sample_id


def validate_index(index: DatasetIndex) -> None:
    """Valida rótulos, identificadores e existência dos frames.

    Args:
        index (DatasetIndex): Índice a ser validado.

    Returns:
        None: A função levanta exceção quando encontra inconsistência.

    Raises:
        ValueError: Se o mapa de rótulos, os identificadores ou as amostras
            forem inválidos.
        FileNotFoundError: Se algum arquivo de frame não existir.
    """
    build_label_map(index.label_map)
    if not index.samples:
        raise ValueError("Dataset index is empty")
    identifiers: set[str] = set()
    for sample in index.samples:
        if sample.sample_id in identifiers:
            raise ValueError(f"Duplicate sample id: {sample.sample_id}")
        identifiers.add(sample.sample_id)
        if sample.raw_label not in index.label_map:
            raise ValueError(f"Raw label {sample.raw_label!r} is absent from the label map")
        for path in sample.frame_paths:
            if not path.is_file():
                raise FileNotFoundError(path)


def normalize_ratios(ratios: Mapping[str, float]) -> dict[str, float]:
    """Normaliza e valida as proporções de treino, validação e teste.

    Args:
        ratios (Mapping[str, float]): Proporções para ``train``, ``val`` e
            ``test``.

    Returns:
        dict[str, float]: Proporções normalizadas na ordem dos subconjuntos.

    Raises:
        ValueError: Se as chaves, os valores ou a soma forem inválidos.
    """
    if set(ratios) != set(SPLIT_NAMES):
        raise ValueError(f"Ratios must define exactly {SPLIT_NAMES}")
    if any(value < 0 for value in ratios.values()):
        raise ValueError("Split ratios cannot be negative")
    total = sum(ratios.values())
    if total <= 0:
        raise ValueError("Sum of split ratios must be positive")
    return {name: ratios[name] / total for name in SPLIT_NAMES}


def allocate_counts(total: int, ratios: Mapping[str, float]) -> dict[str, int]:
    """Distribui grupos entre os três subconjuntos.

    Args:
        total (int): Quantidade de grupos disponíveis.
        ratios (Mapping[str, float]): Proporções normalizadas.

    Returns:
        dict[str, int]: Quantidade de grupos por subconjunto.
    """
    counts = {name: int(total * ratios[name]) for name in SPLIT_NAMES}
    remainder = total - sum(counts.values())
    if remainder:
        target = "test" if ratios["test"] > 0 else "train"
        counts[target] += remainder
    if total >= len(SPLIT_NAMES):
        for name in SPLIT_NAMES:
            if ratios[name] > 0 and counts[name] == 0:
                donor = max(
                    SPLIT_NAMES,
                    key=lambda candidate: (counts[candidate], -SPLIT_NAMES.index(candidate)),
                )
                if counts[donor] > 1:
                    counts[donor] -= 1
                    counts[name] = 1
    return counts


def group_split(
    groups: Sequence[Hashable],
    labels: Sequence[int],
    ratios: Mapping[str, float],
    seed: int = 42,
) -> dict[str, list[Hashable]]:
    """Divide grupos por subconjunto, estratificado pelo rótulo binário.

    Args:
        groups (Sequence[Hashable]): Identificadores de grupo.
        labels (Sequence[int]): Rótulos binários por grupo.
        ratios (Mapping[str, float]): Proporções de treino, validação e teste.
        seed (int): Semente do gerador aleatório.

    Returns:
        dict[str, list[Hashable]]: Grupos atribuídos a cada subconjunto.

    Raises:
        ValueError: Se tamanhos, rótulos ou proporções forem inconsistentes.

    Note:
        Um grupo nunca aparece em mais de um subconjunto.
    """
    if len(groups) != len(labels):
        raise ValueError("groups and labels must have the same length")
    normalized = normalize_ratios(ratios)
    group_labels: dict[Hashable, int] = {}
    for group, label in zip(groups, labels):
        if label not in (0, 1):
            raise ValueError(f"Binary label must be 0 or 1, got {label}")
        previous = group_labels.get(group)
        if previous is not None and previous != label:
            raise ValueError(f"Group {group!r} has inconsistent labels")
        group_labels[group] = label

    rng = random.Random(seed)
    result: dict[str, list[Hashable]] = {name: [] for name in SPLIT_NAMES}
    by_class: dict[int, list[Hashable]] = {0: [], 1: []}
    for group, label in group_labels.items():
        by_class[label].append(group)

    for items in by_class.values():
        shuffled = list(items)
        rng.shuffle(shuffled)
        counts = allocate_counts(len(shuffled), normalized)
        start = 0
        for name in SPLIT_NAMES:
            end = start + counts[name]
            result[name].extend(shuffled[start:end])
            start = end
    return result


def validate_split(split: Mapping[str, Sequence[Hashable]]) -> None:
    """Garante que cada grupo aparece em apenas um subconjunto.

    Args:
        split (Mapping[str, Sequence[Hashable]]): Divisão a ser validada.

    Returns:
        None: A função levanta exceção quando há sobreposição.

    Raises:
        ValueError: Se um grupo aparecer em mais de um subconjunto.
    """
    seen: set[Hashable] = set()
    for name in SPLIT_NAMES:
        for group in split.get(name, []):
            if group in seen:
                raise ValueError(f"Group {group!r} appears in more than one split")
            seen.add(group)
