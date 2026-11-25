"""Composição visual da entrada, do mapa de ativação e das comparações."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from PIL import Image, ImageDraw


def array_to_image(array: object) -> Image.Image:
    """Converte um arranjo de imagem em PIL Image RGB.

    Args:
        array (object): Arranjo 2D ou 3D com dados de imagem.

    Returns:
        Image.Image: Imagem RGB pronta para exibição.

    Note:
        Arranjos ``uint8`` são usados diretamente. Arranjos float na faixa
        ``[-1.5, 1.5]`` são recortados para ``[0, 1]``, o que preserva escores;
        fora dessa faixa são normalizados pelos extremos observados.
    """
    values = np.asarray(array)
    if values.dtype == np.uint8:
        if values.ndim == 2:
            values = np.stack((values, values, values), axis=-1)
        return Image.fromarray(values[..., :3], "RGB")
    values = values.astype(np.float32)
    if values.ndim == 2:
        values = np.stack((values, values, values), axis=-1)
    values = values[..., :3]
    low, high = float(values.min()), float(values.max())
    if low >= -1.5 and high <= 1.5:
        scaled = np.clip(values, 0, 1) * 255
    elif high > low:
        scaled = (values - low) / (high - low) * 255
    else:
        scaled = np.zeros_like(values)
    return Image.fromarray(scaled.astype(np.uint8), "RGB")


def overlay_activation(
    original: Image.Image,
    activation: object,
    alpha: float = 0.45,
) -> Image.Image:
    """Sobreposição colorida de um mapa de ativação à imagem original.

    Args:
        original (Image.Image): Imagem original.
        activation (object): Mapa 2D com valores entre 0 e 1.
        alpha (float): Opacidade do mapa, entre 0 e 1.

    Returns:
        Image.Image: Imagem com o mapa de calor sobreposto.

    Raises:
        ValueError: Se ``activation`` não for bidimensional.
    """
    values = np.asarray(activation, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("activation must be a 2D array")
    low, high = float(values.min()), float(values.max())
    if high > low:
        normalized = (values - low) / (high - low)
    else:
        normalized = np.zeros_like(values)
    red = np.clip(1.5 - np.abs(4 * normalized - 3), 0, 1)
    green = np.clip(1.5 - np.abs(4 * normalized - 2), 0, 1)
    blue = np.clip(1.5 - np.abs(4 * normalized - 1), 0, 1)
    heat = np.stack((red, green, blue), axis=-1)
    heat_image = Image.fromarray((heat * 255).astype(np.uint8), "RGB")
    heat_image = heat_image.resize(original.size)
    opacity = float(np.clip(alpha, 0, 1))
    return Image.blend(original.convert("RGB"), heat_image, opacity)


def compose_comparison(
    original: Image.Image,
    processed: object,
    labels: Sequence[str] = ("Original 1600x768", "YOLO input 1280x1280"),
) -> Image.Image:
    """Monta lado a lado a imagem original e a entrada processada.

    Args:
        original (Image.Image): Imagem original.
        processed (object): Imagem ou arranjo processado.
        labels (Sequence[str]): Títulos das duas metades.

    Returns:
        Image.Image: Comparação com cabeçalho e as duas imagens.

    Raises:
        ValueError: Se ``labels`` tiver menos de dois elementos.
    """
    if len(labels) < 2:
        raise ValueError("labels must contain at least two entries")
    left = original.convert("RGB")
    right = (
        processed.convert("RGB")
        if isinstance(processed, Image.Image)
        else array_to_image(processed)
    )
    if right.size != left.size:
        right = right.resize(left.size)
    width = left.width + right.width
    canvas = Image.new("RGB", (width, left.height + 32), (20, 20, 20))
    canvas.paste(left, (0, 32))
    canvas.paste(right, (left.width, 32))
    draw = ImageDraw.Draw(canvas, "RGBA")
    draw.rectangle((0, 0, left.width, 32), fill=(30, 90, 160, 255))
    draw.rectangle((left.width, 0, width, 32), fill=(160, 60, 60, 255))
    draw.text((8, 8), labels[0], fill=(255, 255, 255, 255))
    draw.text((left.width + 8, 8), labels[1], fill=(255, 255, 255, 255))
    return canvas


__all__ = ["array_to_image", "compose_comparison", "overlay_activation"]
