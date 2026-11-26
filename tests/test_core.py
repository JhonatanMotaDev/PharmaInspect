"""Testes do núcleo da pipeline YOLO11-cls do PharmaInspect."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from pharma_inspect.data import (
    Sample,
    binary_label,
    build_label_map,
    group_split,
    label_name,
    validate_split,
)
from pharma_inspect.evaluation import (
    STRATEGIES,
    aggregate,
    apply_threshold,
    cluster_bootstrap_intervals,
    compute_metrics,
    confusion_matrix,
    performance_metrics,
    wilson_interval,
)
from pharma_inspect.pipeline import load_config
from pharma_inspect.yolo import training as yolo_training
from pharma_inspect.yolo.dataset import export_dataset, letterbox_geometry
from pharma_inspect.yolo.inference import letterbox_preview, predict_batch, predict_probs

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("build_manifest", ROOT / "scripts/build_manifest.py")
build_manifest = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(build_manifest)

RATIOS = {"train": 0.5, "val": 0.25, "test": 0.25}
SCORES = [[0.9, 0.1], [0.2, 0.8], [0.6, 0.4]]


def make_tree(tmp_path: Path) -> Path:
    """Cria uma árvore mínima de classes e sequências.

    Args:
        tmp_path (Path): Diretório temporário do pytest.

    Returns:
        Path: Raiz da árvore criada.
    """
    for name in build_manifest.CLASS_NAMES:
        sequence = tmp_path / name / f"{name}_001"
        sequence.mkdir(parents=True)
        (sequence / "frame.jpg").write_bytes(b"fake")
    return tmp_path


def make_context(tmp_path: Path) -> dict[str, object]:
    """Cria um contexto dividido a partir de um manifesto sintético.

    Args:
        tmp_path (Path): Diretório temporário do pytest.

    Returns:
        dict[str, object]: Contexto compatível com ``export_dataset``.
    """
    manifest = build_manifest.build_manifest(make_tree(tmp_path))
    label_map = build_label_map(manifest["label_map"])
    samples = tuple(
        Sample(
            sample_id=str(entry["sample_id"]),
            frame_paths=tuple(Path(path) for path in entry["frame_paths"]),
            raw_label=str(entry["raw_label"]),
        )
        for entry in manifest["samples"]
    )
    split = group_split(
        [sample.sample_id for sample in samples],
        [binary_label(sample.raw_label, label_map) for sample in samples],
        RATIOS,
        seed=42,
    )
    subsets = {
        name: [sample for sample in samples if sample.sample_id in set(members)]
        for name, members in split.items()
    }
    return {
        "subsets": subsets,
        "label_map": label_map,
        "config": {"seed": 42, "model": {"imgsz": 1280}},
    }


class FakeResult:
    """Resultado mínimo com probabilidades de classe."""

    def __init__(self, scores: list[float]):
        self.probs = SimpleNamespace(data=np.asarray(scores, dtype=np.float32))


class FakeModel:
    """Modelo que registra chamadas e devolve escores fixos."""

    def __init__(self, scores: list[float] | None = None):
        self.scores = scores or [0.25, 0.75, 0.0]
        self.calls: list[dict[str, object]] = []

    def predict(self, **kwargs: object) -> list[FakeResult]:
        """Registra os parâmetros e devolve um resultado sintético."""
        self.calls.append(kwargs)
        return [FakeResult(self.scores)]


def test_build_label_map_uses_binary_codes():
    """Mapeia rótulos originais para códigos binários."""
    mapping = build_label_map({"a": "Clean", "b": "Contaminated", "c": "Contaminated"})
    assert mapping == {"a": 0, "b": 1, "c": 1}


def test_build_label_map_requires_both_classes():
    """Exige as duas classes binárias no mapa."""
    with pytest.raises(ValueError):
        build_label_map({"a": "Clean"})


def test_binary_label_and_name_are_consistent():
    """Mantém coerência entre código e nome da classe."""
    mapping = build_label_map({"a": "Clean", "b": "Contaminated"})
    assert binary_label("b", mapping) == 1
    assert label_name(0) == "Clean"
    assert label_name(1) == "Contaminated"


def test_group_split_keeps_each_group_in_one_subset():
    """Mantém cada sequência em um único subconjunto."""
    groups = ["a", "a", "b", "c", "c", "d", "e", "f", "g", "h"]
    labels = [0, 0, 1, 1, 1, 0, 0, 1, 0, 1]
    split = group_split(groups, labels, RATIOS, seed=7)
    validate_split(split)
    assert {group for members in split.values() for group in members} == set(groups)
    for group in set(groups):
        assert sum(group in members for members in split.values()) == 1


def test_confusion_matrix_and_metrics():
    """Conta a matriz 2x2 e calcula métricas por classe."""
    assert confusion_matrix([0, 0, 1, 1], [0, 1, 1, 1]) == [[1, 1], [0, 2]]
    metrics = compute_metrics([0, 0, 1, 1], [0, 1, 1, 1])
    assert metrics["accuracy"] == pytest.approx(0.75)
    assert metrics["positive_recall"] == pytest.approx(1.0)
    assert metrics["per_class"]["1"]["precision"] == pytest.approx(2 / 3)


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_aggregate_returns_normalized_pair(strategy):
    """Normaliza a saída de cada estratégia de agregação."""
    result = aggregate(SCORES, strategy)
    assert result.shape == (2,)
    assert result.sum() == pytest.approx(1.0)


def test_apply_threshold_uses_positive_score():
    """Aplica o limiar ao escore de Contaminated."""
    assert apply_threshold([0.7, 0.3]) == 0
    assert apply_threshold([0.2, 0.8]) == 1
    with pytest.raises(ValueError):
        apply_threshold([0.2, 0.8], 1.5)


def test_performance_metrics_reports_latency_and_fps():
    """Resume latência média, p95 e FPS."""
    values = performance_metrics([2.0, 4.0])
    assert values["latency_ms_mean"] == pytest.approx(3.0)
    assert values["fps"] == pytest.approx(2 / 0.006)
    assert values["latency_ms_p95"] >= values["latency_ms_mean"]


def test_export_dataset_creates_yolo_cls_layout(tmp_path):
    """Exporta split pastas de classe sem duplicar referências."""
    context = make_context(tmp_path)
    output = tmp_path / "yolo"
    summary = export_dataset(context, output)
    assert set(summary["counts"]) == {"train", "val", "test"}
    assert (output / "data.yaml").read_text(encoding="utf-8").count("Clean") >= 1
    assert (output / "export_manifest.json").is_file()
    linked = list((output / "train").glob("*/*"))
    assert linked
    assert all(path.is_symlink() for path in linked)
    assert len(linked) == sum(summary["counts"]["train"].values())


def test_export_dataset_respects_frame_limit(tmp_path):
    """Limita a quantidade de frames por sequência na exportação."""
    context = make_context(tmp_path)
    output = tmp_path / "yolo_limited"
    summary = export_dataset(context, output, max_frames_per_sequence=1)
    expected = sum(len(sample.frame_paths[:1]) for sample in context["subsets"]["train"])
    assert sum(summary["counts"]["train"].values()) == expected


def test_letterbox_geometry_keeps_original_aspect():
    """Calcula escala e padding da imagem 1600x768 em canvas 1280."""
    geometry = letterbox_geometry(1600, 768, 1280)
    assert geometry.scale == pytest.approx(0.8)
    assert geometry.fitted_width == 1280
    assert geometry.fitted_height == 614
    assert geometry.offset_x == 0
    assert geometry.offset_y == 333


def test_letterbox_preview_returns_square_image():
    """Produz a visualização quadrada usada pelo YOLO."""
    preview, geometry = letterbox_preview(Image.new("RGB", (1600, 768)), 1280)
    assert preview.size == (1280, 1280)
    assert geometry.size == 1280


def test_predict_probs_normalizes_first_two_scores():
    """Usa apenas as duas classes configuradas do checkpoint."""
    model = FakeModel()
    scores = predict_probs(model, Image.new("RGB", (8, 8)), imgsz=1280, device="cpu")
    assert np.allclose(scores, [0.25, 0.75])
    assert model.calls[0]["imgsz"] == 1280
    assert model.calls[0]["source"].size == (1280, 1280)


def test_predict_batch_keeps_input_order():
    """Mantém a ordem das imagens ao classificar um lote."""
    scores = predict_batch(FakeModel(), [Image.new("RGB", (4, 4))] * 3, imgsz=1280)
    assert scores.shape == (3, 2)


def test_default_config_declares_yolo():
    """A configuração padrão descreve a arquitetura atual."""
    config = load_config(ROOT / "configs/default.yaml")
    assert config["model"]["family"] == "yolo"
    assert config["model"]["variant"] == "yolo11n-cls"
    assert config["model"]["imgsz"] == 1280


def test_train_classifier_passes_dataset_directory(tmp_path, monkeypatch):
    """O treino de classificação usa a raiz, não o arquivo data.yaml."""
    data_dir = tmp_path / "dataset"
    (data_dir / "train").mkdir(parents=True)
    (data_dir / "val").mkdir()
    data_yaml = data_dir / "data.yaml"
    data_yaml.write_text("{}", encoding="utf-8")
    best = tmp_path / "best.pt"
    best.write_bytes(b"checkpoint")
    captured: dict[str, object] = {}

    class FakeTrainerModel:
        """Modelo sintético que apenas captura os argumentos do treino."""

        def __init__(self) -> None:
            self.trainer = SimpleNamespace(best=str(best))

        def train(self, **kwargs: object) -> dict[str, bool]:
            """Registra os argumentos e finge um treino concluído."""
            captured.update(kwargs)
            return {"ok": True}

    monkeypatch.setattr(yolo_training, "load_base_model", lambda _: FakeTrainerModel())
    result = yolo_training.train_classifier(
        {"training": {}, "model": {"imgsz": 1280}, "seed": 42},
        data_yaml=data_yaml,
        pretrained_weights="base.pt",
        output_weights=tmp_path / "out.pt",
        device="cpu",
        epochs=1,
        batch_size=2,
    )
    assert captured["data"] == str(data_dir)
    assert Path(result["weights"]).is_file()


def test_wilson_interval_is_not_degenerate_without_errors():
    low, high = wilson_interval(24, 24)
    assert 0.8 < low < 1.0 and high == 1.0
    assert wilson_interval(0, 0) == [0.0, 1.0]
    with pytest.raises(ValueError):
        wilson_interval(5, 4)


def test_cluster_bootstrap_is_reproducible_and_bounded():
    truth = [0, 0, 1, 1, 1, 1]
    pred = [0, 1, 1, 1, 1, 0]
    groups = [0, 0, 1, 1, 2, 2]
    first = cluster_bootstrap_intervals(truth, pred, groups, n_boot=200)
    assert first == cluster_bootstrap_intervals(truth, pred, groups, n_boot=200)
    for low, high in first.values():
        assert 0.0 <= low <= high <= 1.0
    with pytest.raises(ValueError):
        cluster_bootstrap_intervals(truth, pred, groups[:-1])
