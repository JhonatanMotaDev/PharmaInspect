"""Montagem do contexto de execução a partir de config e manifesto."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .data import binary_label, build_label_map, group_split, load_index, sample_group, validate_index


def load_config(path: str | Path) -> dict[str, Any]:
    """Lê o arquivo YAML de configuração.

    Args:
        path (str | Path): Caminho do arquivo YAML.

    Returns:
        dict[str, Any]: Configuração carregada.

    Raises:
        OSError: Se o arquivo não puder ser lido.
        yaml.YAMLError: Se o conteúdo não for YAML válido.
    """
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def build_context(
    config_path: str | Path, manifest_override: str | Path | None = None
) -> dict[str, Any]:
    """Carrega dados, valida caminhos e produz os subconjuntos.

    Args:
        config_path (str | Path): Caminho do arquivo YAML de configuração.
        manifest_override (str | Path | None): Manifesto que substitui
            ``data.manifest`` do config.

    Returns:
        dict[str, Any]: Contexto com chaves ``config``, ``label_map``,
            ``splits`` e ``subsets``.

    Raises:
        ValueError: Se nenhum manifesto for definido ou os dados forem
            inválidos.
        FileNotFoundError: Se algum frame do manifesto não existir.

    Note:
        A divisão usa ``sample_id`` como identificador de sequência.
    """
    config = load_config(config_path)
    manifest_value = manifest_override or config["data"]["manifest"]
    if not manifest_value:
        raise ValueError("Define data.manifest in the config or pass --manifest")
    index = load_index(manifest_value)
    validate_index(index)
    label_map = build_label_map(index.label_map)

    groups = [sample_group(sample) for sample in index.samples]
    labels = [binary_label(sample.raw_label, label_map) for sample in index.samples]
    split_groups = group_split(groups, labels, config["splits"], config.get("seed", 42))
    subsets = {
        name: [sample for sample in index.samples if sample_group(sample) in set(members)]
        for name, members in split_groups.items()
    }
    return {
        "config": config,
        "label_map": label_map,
        "splits": split_groups,
        "subsets": subsets,
    }

