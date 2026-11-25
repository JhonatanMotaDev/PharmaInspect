"""Contrato do cliente VLM, desativado por padrão.

Nenhuma chamada de API é feita nesta etapa. A integração futura deve
implementar ``describe`` sem alterar a classe produzida pelo YOLO.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any, Protocol


class VLMClient(Protocol):
    """Contrato mínimo de um cliente de VLM via API."""

    def describe(self, image: Any, result: Mapping[str, Any]) -> dict[str, Any]:
        """Descreve a imagem e o resultado estruturado do classificador."""


def build_client(config: Mapping[str, Any]) -> VLMClient | None:
    """Cria um cliente quando o VLM estiver explicitamente habilitado.

    Args:
        config (Mapping[str, Any]): Configuração carregada do YAML.

    Returns:
        VLMClient | None: ``None`` quando ``vlm.enabled`` for falso.

    Raises:
        RuntimeError: Se o VLM estiver habilitado sem adaptador implementado ou
            sem a variável de ambiente com a chave de API.
    """
    settings = config.get("vlm") or {}
    if not settings.get("enabled", False):
        return None

    provider = settings.get("provider")
    if not provider:
        raise RuntimeError("vlm.enabled requires vlm.provider")
    api_key_env = settings.get("api_key_env")
    if not api_key_env or not os.environ.get(str(api_key_env)):
        raise RuntimeError(f"VLM API key is missing in environment variable {api_key_env!r}")
    raise RuntimeError(
        f"No VLM adapter implemented for provider {provider!r}. "
        "Implement a client with describe(image, result) in this module."
    )


__all__ = ["VLMClient", "build_client"]
