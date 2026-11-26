"""Métricas, agregação por sequência e relatórios JSON do YOLO-cls."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

STRATEGIES = ("mean", "median", "max", "vote")


def validate_binary(y_true: Sequence[int], y_pred: Sequence[int]) -> None:
    """Valida vetores binários de mesmo tamanho.

    Args:
        y_true (Sequence[int]): Rótulos binários de referência.
        y_pred (Sequence[int]): Rótulos binários previstos.

    Returns:
        None: A função levanta exceção quando encontra inconsistência.

    Raises:
        ValueError: Se os tamanhos divergirem ou algum valor não for 0 ou 1.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    for value in (*y_true, *y_pred):
        if value not in (0, 1):
            raise ValueError(f"Binary values must be 0 or 1, got {value}")


def confusion_matrix(y_true: Sequence[int], y_pred: Sequence[int]) -> list[list[int]]:
    """Conta acertos e erros binários em uma matriz 2x2.

    Args:
        y_true (Sequence[int]): Rótulos binários de referência.
        y_pred (Sequence[int]): Rótulos binários previstos.

    Returns:
        list[list[int]]: Matriz ``[[TN, FP], [FN, TP]]``.

    Raises:
        ValueError: Se os vetores tiverem tamanhos diferentes ou valores
            fora de ``{0, 1}``.
    """
    validate_binary(y_true, y_pred)
    matrix = [[0, 0], [0, 0]]
    for true_value, predicted_value in zip(y_true, y_pred):
        matrix[true_value][predicted_value] += 1
    return matrix


def compute_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> dict[str, Any]:
    """Calcula acurácia, métricas por classe e matriz de confusão.

    Args:
        y_true (Sequence[int]): Rótulos binários de referência.
        y_pred (Sequence[int]): Rótulos binários previstos.

    Returns:
        dict[str, Any]: Métricas com chaves ``accuracy``,
            ``positive_recall``, ``per_class``, ``confusion_matrix`` e
            ``support``.

    Raises:
        ValueError: Se os vetores forem inválidos.

    Note:
        ``per_class`` usa as chaves ``"0"`` e ``"1"`` para ``Clean`` e
        ``Contaminated``.
    """
    matrix = confusion_matrix(y_true, y_pred)
    total = sum(sum(row) for row in matrix)
    accuracy = sum(matrix[index][index] for index in (0, 1)) / total if total else 0.0
    per_class: dict[str, dict[str, float]] = {}
    for index in (0, 1):
        true_positive = matrix[index][index]
        false_positive = matrix[0][index] + matrix[1][index] - true_positive
        false_negative = sum(matrix[index]) - true_positive
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else 0.0
        )
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[str(index)] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": true_positive + false_negative,
        }
    return {
        "accuracy": accuracy,
        "positive_recall": per_class["1"]["recall"],
        "per_class": per_class,
        "confusion_matrix": matrix,
        "support": total,
    }


def write_json_report(path: str | Path, payload: Any) -> Path:
    """Grava um relatório JSON em disco.

    Args:
        path (str | Path): Caminho de destino.
        payload (Any): Objeto serializável em JSON.

    Returns:
        Path: Caminho do arquivo gravado.

    Raises:
        OSError: Se não for possível criar diretórios ou escrever o arquivo.
        TypeError: Se ``payload`` não for serializável.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return target


def metrics_to_lines(metrics: dict[str, Any]) -> list[str]:
    """Formata métricas para saída textual.

    Args:
        metrics (dict[str, Any]): Métricas retornadas por ``compute_metrics``.

    Returns:
        list[str]: Linhas prontas para exibição.
    """
    lines = [f"accuracy: {metrics['accuracy']:.4f}"]
    for name, values in metrics["per_class"].items():
        lines.append(
            f"class {name}: precision={values['precision']:.4f} "
            f"recall={values['recall']:.4f} f1={values['f1']:.4f} "
            f"support={values['support']}"
        )
    lines.append(f"confusion_matrix: {metrics['confusion_matrix']}")
    return lines


def performance_to_lines(performance: dict[str, float]) -> list[str]:
    """Formata métricas de desempenho computacional.

    Args:
        performance (dict[str, float]): Métricas retornadas pelo evaluate.

    Returns:
        list[str]: Linhas prontas para exibição.
    """
    return [
        f"latency_ms_mean: {performance['latency_ms_mean']:.4f}",
        f"latency_ms_p95: {performance['latency_ms_p95']:.4f}",
        f"fps: {performance['fps']:.2f}",
    ]


def aggregate(frame_scores: Sequence[Sequence[float]], strategy: str = "mean") -> np.ndarray:
    """Combina escores de frames em um par normalizado por sequência.

    Args:
        frame_scores (Sequence[Sequence[float]]): Escores por frame, com duas
            colunas.
        strategy (str): Estratégia em ``STRATEGIES``.

    Returns:
        np.ndarray: Par de escores normalizado da sequência.

    Raises:
        ValueError: Se a estratégia for desconhecida, os escores estiverem
            vazios ou a soma agregada for zero.

    Note:
        ``vote`` usa o argmax de cada frame como voto de classe.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown aggregation strategy {strategy!r}; choose from {STRATEGIES}")
    values = np.asarray(frame_scores, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2 or values.shape[0] == 0:
        raise ValueError("frame_scores must be a non-empty array with two columns")

    if strategy == "mean":
        result = values.mean(axis=0)
    elif strategy == "median":
        result = np.median(values, axis=0)
    elif strategy == "max":
        result = values.max(axis=0)
    else:
        votes = values.argmax(axis=1)
        result = np.array([(votes == 0).mean(), (votes == 1).mean()])

    total = float(result.sum())
    if total <= 0:
        raise ValueError("Aggregated scores sum to zero")
    return result / total


def apply_threshold(scores: Sequence[float], threshold: float = 0.5) -> int:
    """Converte um par de escores em código binário.

    Args:
        scores (Sequence[float]): Par ``[Clean, Contaminated]``.
        threshold (float): Limiar aplicado ao escore de ``Contaminated``.

    Returns:
        int: 0 para ``Clean`` e 1 para ``Contaminated``.

    Raises:
        ValueError: Se o par for inválido ou o limiar estiver fora de ``[0, 1]``.
    """
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or values.shape[0] != 2:
        raise ValueError("Expected one pair of class scores")
    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be between 0 and 1")
    return int(values[1] >= threshold)


def performance_metrics(latencies_ms: Sequence[float]) -> dict[str, float]:
    """Resume latência média, p95 e FPS a partir de medições por imagem.

    Args:
        latencies_ms (Sequence[float]): Latências em milissegundos.

    Returns:
        dict[str, float]: Métricas ``latency_ms_mean``, ``latency_ms_p95`` e ``fps``.

    Raises:
        ValueError: Se a lista estiver vazia ou contiver valores não positivos.
    """
    values = np.asarray(latencies_ms, dtype=np.float64)
    if values.size == 0 or np.any(values <= 0):
        raise ValueError("latencies_ms must be non-empty and positive")
    total_seconds = float(values.sum()) / 1000.0
    return {
        "latency_ms_mean": float(values.mean()),
        "latency_ms_p95": float(np.percentile(values, 95)),
        "fps": float(values.size / total_seconds),
    }


def wilson_interval(successes: int, total: int, z: float = 1.96) -> list[float]:
    """Intervalo de Wilson para uma proporção binomial.

    Args:
        successes (int): Quantidade de acertos.
        total (int): Quantidade de ensaios.
        z (float): Quantil normal; 1.96 corresponde a 95%.

    Returns:
        list[float]: ``[inferior, superior]``; ``[0, 1]`` quando ``total`` é zero.

    Raises:
        ValueError: Se ``successes`` estiver fora de ``[0, total]``.

    Note:
        Ao contrário do bootstrap, não degenera quando não há erros observados.
    """
    if total == 0:
        return [0.0, 1.0]
    if not 0 <= successes <= total:
        raise ValueError("successes must be between 0 and total")
    p = successes / total
    denom = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denom
    margin = z * np.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denom
    return [float(max(0.0, center - margin)), float(min(1.0, center + margin))]


def metrics_intervals(y_true: Sequence[int], y_pred: Sequence[int]) -> dict[str, list[float]]:
    """Intervalos de Wilson de 95% para acurácia e recall por classe.

    Args:
        y_true (Sequence[int]): Rótulos binários de referência.
        y_pred (Sequence[int]): Rótulos binários previstos.

    Returns:
        dict[str, list[float]]: Chaves ``accuracy``, ``recall_clean`` e
            ``recall_contaminated``.

    Raises:
        ValueError: Se os vetores forem inválidos.
    """
    matrix = confusion_matrix(y_true, y_pred)
    total = sum(sum(row) for row in matrix)
    return {
        "accuracy": wilson_interval(matrix[0][0] + matrix[1][1], total),
        "recall_clean": wilson_interval(matrix[0][0], sum(matrix[0])),
        "recall_contaminated": wilson_interval(matrix[1][1], sum(matrix[1])),
    }


def cluster_bootstrap_intervals(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    groups: Sequence[int],
    *,
    n_boot: int = 1000,
    seed: int = 42,
) -> dict[str, list[float]]:
    """Bootstrap por grupo (sequência) para métricas por frame.

    Args:
        y_true (Sequence[int]): Rótulos por frame.
        y_pred (Sequence[int]): Previsões por frame.
        groups (Sequence[int]): Índice da sequência de cada frame.
        n_boot (int): Número de reamostragens.
        seed (int): Semente do gerador.

    Returns:
        dict[str, list[float]]: Percentis 2,5 e 97,5 de ``accuracy``,
            ``recall_clean`` e ``recall_contaminated``.

    Raises:
        ValueError: Se os vetores forem inválidos, tiverem tamanhos diferentes
            ou ``n_boot`` não for positivo.

    Note:
        Reamostrar sequências inteiras respeita a correlação entre frames do
        mesmo frasco; reamostrar frames subestimaria a incerteza.
    """
    validate_binary(y_true, y_pred)
    if len(groups) != len(y_true) or n_boot <= 0:
        raise ValueError("groups must match the labels and n_boot must be positive")
    truth = np.asarray(y_true)
    pred = np.asarray(y_pred)
    ids = np.asarray(groups)
    unique = np.unique(ids)
    table = np.array(
        [
            [
                ((truth == 0) & (pred == 0) & (ids == g)).sum(),
                ((truth == 0) & (ids == g)).sum(),
                ((truth == 1) & (pred == 1) & (ids == g)).sum(),
                ((truth == 1) & (ids == g)).sum(),
            ]
            for g in unique
        ],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = {"accuracy": [], "recall_clean": [], "recall_contaminated": []}
    for _ in range(n_boot):
        tn, neg, tp, pos = table[rng.integers(0, len(unique), len(unique))].sum(axis=0)
        draws["accuracy"].append((tn + tp) / (neg + pos) if neg + pos else np.nan)
        draws["recall_clean"].append(tn / neg if neg else np.nan)
        draws["recall_contaminated"].append(tp / pos if pos else np.nan)
    return {
        name: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]
        for name, v in draws.items()
    }


__all__ = [
    "STRATEGIES",
    "aggregate",
    "apply_threshold",
    "compute_metrics",
    "cluster_bootstrap_intervals",
    "confusion_matrix",
    "metrics_intervals",
    "metrics_to_lines",
    "performance_metrics",
    "performance_to_lines",
    "validate_binary",
    "wilson_interval",
    "write_json_report",
]
