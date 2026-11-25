"""Carga do classificador YOLO11-cls e validação do contrato de classes."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def resolve_path(value: str | Path, root: str | Path | None = None) -> Path:
    """Resolve um caminho relativo à raiz do projeto.

    Args:
        value (str | Path): Caminho absoluto ou relativo.
        root (str | Path | None): Raiz de referência; padrão é a raiz do repo.

    Returns:
        Path: Caminho absoluto.
    """
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    return ((Path(root) if root else PROJECT_ROOT) / path).resolve()


def load_base_model(weights: str | Path) -> Any:
    """Carrega pesos iniciais do Ultralytics para treino.

    Args:
        weights (str | Path): Arquivo de pesos ou nome de modelo Ultralytics.

    Returns:
        Any: Instância ``YOLO`` configurada para classificação.

    Raises:
        RuntimeError: Se ``ultralytics`` não estiver disponível.
        ValueError: Se o modelo carregado não for de classificação.
    """
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("ultralytics is required for the YOLO pipeline") from exc

    model = YOLO(str(weights))
    if model.task != "classify":
        raise ValueError(f"Expected a classify model, got task {model.task!r}")
    return model


def load_classifier(
    config: Mapping[str, Any],
    *,
    weights: str | Path | None = None,
    device: str | None = None,
    root: str | Path | None = None,
) -> Any:
    """Carrega o checkpoint binário usado na inferência.

    Args:
        config (Mapping[str, Any]): Configuração carregada do YAML.
        weights (str | Path | None): Sobrescreve ``model.weights``.
        device (str | None): Sobrescreve ``inference.device``.
        root (str | Path | None): Raiz para caminhos relativos.

    Returns:
        Any: Modelo Ultralytics com nomes de classe alinhados ao config.

    Raises:
        FileNotFoundError: Se o checkpoint final não existir.
        ValueError: Se o checkpoint não tiver exatamente duas classes ou se a
            configuração de classes for inválida.
    """
    model_config = config.get("model", {})
    class_names = tuple(model_config.get("class_names", ("Clean", "Contaminated")))
    if len(class_names) != 2:
        raise ValueError("model.class_names must contain exactly two class names")

    weight_value = weights or model_config.get("weights")
    if not weight_value:
        raise ValueError("Define model.weights in the configuration")
    weight_path = resolve_path(weight_value, root)
    if not weight_path.is_file():
        raise FileNotFoundError(
            f"YOLO checkpoint not found: {weight_path}. "
            "Train on GPU with scripts/train_yolo.py and copy best.pt to this path."
        )

    model = load_base_model(weight_path)
    original_names = getattr(model, "names", None)
    if original_names is not None and len(original_names) != 2:
        raise ValueError(
            f"Checkpoint has {len(original_names)} classes; train a 2-class YOLO-cls model"
        )
    model.model.names = {index: name for index, name in enumerate(class_names)}
    setattr(model, "_pharma_device", device or config.get("inference", {}).get("device", "cpu"))
    return model


def model_device(model: Any, fallback: str = "cpu") -> str:
    """Retorna o dispositivo configurado para inferência.

    Args:
        model (Any): Modelo carregado por ``load_classifier``.
        fallback (str): Valor usado quando o modelo não traz a anotação.

    Returns:
        str: Nome do dispositivo, por exemplo ``cpu`` ou ``cuda:0``.
    """
    return str(getattr(model, "_pharma_device", fallback))
