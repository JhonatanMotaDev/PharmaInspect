"""Mapa de ativação do YOLO-cls como visualização associada."""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

from .inference import letterbox_preview
from .model import model_device

_RESAMPLING = getattr(Image, "Resampling", Image)
_EXCLUDED_MODULES = {"Classify", "Linear", "AdaptiveAvgPool2d"}


def _sequential_backbone(model: Any) -> Any:
    """Localiza a sequência de blocos realmente usada pelo predictor.

    O ``BasePredictor`` do Ultralytics copia e funde o modelo na primeira
    inferência. Por isso o hook precisa ser registrado no modelo interno do
    predictor, não no wrapper original.

    Args:
        model (Any): Modelo Ultralytics carregado.

    Returns:
        Any: Sequência de blocos do modelo efetivo, ou ``None``.
    """
    backend = getattr(getattr(model, "predictor", None), "model", None)
    if backend is not None:
        inner = getattr(backend, "model", None)
        sequential = getattr(inner, "model", None)
        if sequential is not None:
            return sequential
    return getattr(getattr(model, "model", None), "model", None)


def find_activation_module(model: Any) -> Any:
    """Seleciona o último bloco convolucional antes da cabeça de classe.

    Args:
        model (Any): Modelo Ultralytics carregado.

    Returns:
        Any: Módulo PyTorch que produz um mapa de características 4D.

    Raises:
        RuntimeError: Se a estrutura do modelo não permitir localizar o bloco.
    """
    backbone = _sequential_backbone(model)
    if backbone is None:
        raise RuntimeError("YOLO backbone is not initialized; run an inference first")
    children = list(backbone.children())
    for module in reversed(children):
        if type(module).__name__ not in _EXCLUDED_MODULES:
            return module
    raise RuntimeError("Could not locate a YOLO feature block for the activation map")


def activation_map(
    model: Any,
    image: Image.Image,
    *,
    imgsz: int = 1280,
    device: str | None = None,
) -> np.ndarray:
    """Gera um mapa 2D de ativação média no tamanho da imagem original.

    O mapa é uma visualização associada da atividade do último bloco do
    backbone. Não usa gradientes, não é uma técnica de atribuição, não é
    anotação de partículas e não estabelece causalidade.

    Args:
        model (Any): Modelo Ultralytics carregado.
        image (Image.Image): Imagem original.
        imgsz (int): Resolução quadrada usada na inferência.
        device (str | None): Dispositivo de inferência.

    Returns:
        np.ndarray: Mapa ``float32`` normalizado em ``[0, 1]``.

    Raises:
        RuntimeError: Se o hook não capturar um tensor 4D.

    Note:
        A inferência usa o mesmo canvas letterboxado de ``predict_probs``. O
        mapa é recortado para a região útil do canvas e remapeado para a imagem
        original.
    """
    canvas, geometry = letterbox_preview(image, imgsz)
    selected_device = device or model_device(model)
    if getattr(model, "predictor", None) is None:
        model.predict(source=canvas, imgsz=imgsz, device=selected_device, verbose=False)

    target = find_activation_module(model)
    captured: dict[str, Any] = {}

    def capture(_module: Any, _inputs: Any, output: Any) -> None:
        """Armazena a saída 4D do bloco observado."""
        if getattr(output, "dim", lambda: 0)() == 4:
            captured["feature"] = output.detach()

    handle = target.register_forward_hook(capture)
    try:
        model.predict(source=canvas, imgsz=imgsz, device=selected_device, verbose=False)
    finally:
        handle.remove()

    feature = captured.get("feature")
    if feature is None:
        raise RuntimeError("Activation hook did not capture a 4D feature map")
    channels = feature[0].cpu().numpy().astype(np.float32)
    values = np.abs(channels).mean(axis=0)
    peak = float(values.max())
    if peak <= 0:
        return np.zeros((image.height, image.width), dtype=np.float32)
    values /= peak

    map_height, map_width = values.shape
    x0 = int(round(geometry.offset_x / geometry.size * map_width))
    y0 = int(round(geometry.offset_y / geometry.size * map_height))
    x1 = int(round((geometry.offset_x + geometry.fitted_width) / geometry.size * map_width))
    y1 = int(round((geometry.offset_y + geometry.fitted_height) / geometry.size * map_height))
    cropped = values[max(0, y0) : max(0, y1), max(0, x0) : max(0, x1)]
    if cropped.size == 0:
        cropped = values
    resized = Image.fromarray(np.clip(cropped * 255, 0, 255).astype(np.uint8), "L")
    resized = resized.resize((image.width, image.height), _RESAMPLING.BILINEAR)
    return np.asarray(resized, dtype=np.float32) / 255.0


__all__ = ["activation_map", "find_activation_module"]
