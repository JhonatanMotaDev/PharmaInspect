"""Wrapper fino ao redor do treino Ultralytics (executado em GPU externa)."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .model import load_base_model


def train_classifier(
    config: Mapping[str, Any],
    *,
    data_yaml: str | Path,
    pretrained_weights: str | Path,
    output_weights: str | Path,
    device: str,
    epochs: int | None = None,
    batch_size: int | None = None,
    imgsz: int | None = None,
    project: str | Path | None = None,
    name: str = "yolo11n_cls",
) -> dict[str, Any]:
    """Treina o YOLO11-cls e copia ``best.pt`` para o checkpoint final.

    Args:
        config (Mapping[str, Any]): Configuração carregada do YAML.
        data_yaml (str | Path): ``data.yaml`` exportado ou diretório raiz do
            dataset. O treino de classificação do Ultralytics usa o diretório
            raiz, que deve conter ``train/`` e ``val/``.
        pretrained_weights (str | Path): Pesos iniciais do Ultralytics.
        output_weights (str | Path): Destino do melhor checkpoint.
        device (str): Dispositivo do treino, normalmente ``cuda:0``.
        epochs (int | None): Sobrescreve ``training.epochs``.
        batch_size (int | None): Sobrescreve ``training.batch_size``.
        imgsz (int | None): Sobrescreve ``model.imgsz``.
        project (str | Path | None): Diretório de runs do Ultralytics.
        name (str): Nome do run.

    Returns:
        dict[str, Any]: Caminho do checkpoint final e metas do treino.

    Raises:
        FileNotFoundError: Se o dataset não existir ou não tiver ``train/`` e
            ``val/``.
        RuntimeError: Se o Ultralytics não indicar o melhor checkpoint salvo.
        ValueError: Se épocas, batch ou resolução não forem positivos.
    """
    import shutil

    data_path = Path(data_yaml)
    if data_path.is_file():
        data_root = data_path.parent
    elif data_path.is_dir():
        data_root = data_path
    else:
        raise FileNotFoundError(data_path)
    for split in ("train", "val"):
        if not (data_root / split).is_dir():
            raise FileNotFoundError(data_root / split)

    training_config = config.get("training", {})
    selected = {
        "epochs": epochs if epochs is not None else int(training_config.get("epochs", 50)),
        "batch_size": (
            batch_size if batch_size is not None else int(training_config.get("batch_size", 8))
        ),
        "imgsz": imgsz if imgsz is not None else int(config.get("model", {}).get("imgsz", 1280)),
    }
    for name, value in selected.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")
    model = load_base_model(pretrained_weights)
    results = model.train(
        data=str(data_root),
        epochs=selected["epochs"],
        batch=selected["batch_size"],
        imgsz=selected["imgsz"],
        lr0=float(training_config.get("learning_rate", 0.001)),
        device=device,
        project=str(project or training_config.get("project", "artifacts/yolo/runs")),
        name=name,
        seed=int(config.get("seed", 42)),
        exist_ok=True,
        verbose=False,
    )
    best = getattr(getattr(model, "trainer", None), "best", None)
    if not best or not Path(best).is_file():
        raise RuntimeError("Ultralytics did not provide a best.pt checkpoint")
    destination = Path(output_weights)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, destination)
    return {
        "weights": str(destination),
        "source_checkpoint": str(best),
        "device": device,
        "metrics": results,
    }


__all__ = ["train_classifier"]
