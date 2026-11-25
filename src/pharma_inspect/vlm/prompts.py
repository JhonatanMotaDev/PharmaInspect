"""Prompts auxiliares do VLM; nunca substituem a decisão do YOLO."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

SYSTEM_PROMPT = (
    "Você é um assistente de inspeção visual farmacêutica. Descreva apenas "
    "características visíveis na imagem e resuma o resultado estruturado "
    "recebido. Não declara composição química, não substitui a classificação "
    "do modelo visual e deve mencionar incerteza quando a evidência for limitada."
)


def build_prompt(result: Mapping[str, Any]) -> str:
    """Monta o resumo estruturado enviado junto com a imagem.

    Args:
        result (Mapping[str, Any]): Resultado da inferência YOLO.

    Returns:
        str: Prompt em linguagem natural com classe, confiança e agregação.
    """
    decision = result.get("decision") or {}
    final_scores = result.get("final_scores") or [0.0, 0.0]
    return (
        "Resultado estruturado do classificador visual (fonte oficial):\n"
        f"- classe: {decision.get('label', 'indefinida')}\n"
        f"- score Clean: {float(final_scores[0]):.4f}\n"
        f"- score Contaminated: {float(final_scores[1]):.4f}\n"
        f"- frames analisados: {result.get('frames', 0)}\n"
        f"- agregação: {result.get('strategy', 'mean')}\n"
        "Descreva observações visuais e limitações sem alterar essa classe."
    )


system_prompt = SYSTEM_PROMPT

__all__ = ["SYSTEM_PROMPT", "build_prompt", "system_prompt"]
