"""Inferência YOLO-cls em alta resolução e preparação visual da entrada."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from PIL import Image

from .dataset import Letterbox, letterbox_geometry
from .model import model_device

_RESAMPLING = getattr(Image, "Resampling", Image)


def _as_numpy(data: Any) -> np.ndarray:
    """Converte a saída de probs do Ultralytics em array NumPy.

    Args:
        data (Any): Tensor PyTorch, array NumPy ou sequência numérica.

    Returns:
        np.ndarray: Array 1D de escores.
    """
    if hasattr(data, "detach"):
        data = data.detach()
    if hasattr(data, "cpu"):
        data = data.cpu()
    if hasattr(data, "numpy"):
        data = data.numpy()
    values = np.asarray(data, dtype=np.float64).reshape(-1)
    return values


def letterbox_preview(
    image: Image.Image,
    imgsz: int,
    fill: tuple[int, int, int] = (114, 114, 114),
) -> tuple[Image.Image, Letterbox]:
    """Gera a visualização quadrada enviada ao YOLO.

    Args:
        image (Image.Image): Imagem original.
        imgsz (int): Lado do canvas quadrado.
        fill (tuple[int, int, int]): Cor do padding, igual ao Ultralytics.

    Returns:
        tuple[Image.Image, Letterbox]: Imagem em ``imgsz × imgsz`` e geometria.
    """
    source = image.convert("RGB")
    geometry = letterbox_geometry(source.width, source.height, imgsz)
    fitted = source.resize(
        (geometry.fitted_width, geometry.fitted_height), _RESAMPLING.BILINEAR
    )
    canvas = Image.new("RGB", (geometry.size, geometry.size), fill)
    canvas.paste(fitted, (geometry.offset_x, geometry.offset_y))
    return canvas, geometry


def predict_probs(
    model: Any,
    image: Image.Image,
    *,
    imgsz: int = 1280,
    device: str | None = None,
    class_count: int = 2,
) -> np.ndarray:
    """Executa a classificação de uma imagem e devolve os escores por classe.

    Args:
        model (Any): Modelo Ultralytics carregado.
        image (Image.Image): Imagem original RGB.
        imgsz (int): Resolução quadrada de entrada.
        device (str | None): Dispositivo; usa o do modelo quando ausente.
        class_count (int): Quantidade de classes consideradas do checkpoint.

    Returns:
        np.ndarray: Par ``[Clean, Contaminated]`` com soma 1.

    Raises:
        RuntimeError: Se o resultado não contiver probabilidades de classe.
        ValueError: Se ``class_count`` não for positivo ou não houver escores.

    Note:
        O canvas letterboxado é enviado ao modelo. O transform padrão de
        classificação do Ultralytics usa ``Resize(shortest_edge)`` e
        ``CenterCrop``; com a imagem original 1600×768 ele recortaria as
        laterais. No canvas quadrado, resize e crop são identidade.
    """
    if class_count <= 0:
        raise ValueError("class_count must be positive")
    canvas, _ = letterbox_preview(image, imgsz)
    results = model.predict(
        source=canvas,
        imgsz=imgsz,
        device=device or model_device(model),
        verbose=False,
    )
    if not results:
        raise RuntimeError("YOLO returned no result for the image")
    probs = getattr(results[0], "probs", None)
    if probs is None:
        raise RuntimeError("YOLO result does not contain classification probabilities")
    values = _as_numpy(probs.data)[:class_count]
    if values.size != class_count:
        raise ValueError(f"Expected {class_count} class scores, got {values.size}")
    total = float(values.sum())
    if total <= 0:
        raise ValueError("Class scores sum to zero")
    return values / total


def predict_batch(
    model: Any,
    images: Sequence[Image.Image],
    *,
    imgsz: int = 1280,
    device: str | None = None,
    class_count: int = 2,
) -> np.ndarray:
    """Classifica várias imagens mantendo a ordem de entrada.

    Args:
        model (Any): Modelo Ultralytics carregado.
        images (Sequence[Image.Image]): Imagens originais.
        imgsz (int): Resolução quadrada de entrada.
        device (str | None): Dispositivo de inferência.
        class_count (int): Quantidade de classes consideradas.

    Returns:
        np.ndarray: Matriz ``N × class_count`` de escores.

    Raises:
        ValueError: Se ``images`` estiver vazia.
    """
    if not images:
        raise ValueError("images cannot be empty")
    return np.stack(
        [
            predict_probs(model, image, imgsz=imgsz, device=device, class_count=class_count)
            for image in images
        ]
    )


__all__ = ["letterbox_preview", "predict_batch", "predict_probs"]
