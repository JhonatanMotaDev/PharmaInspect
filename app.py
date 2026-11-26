"""Interface Streamlit para inspeção de frascos com YOLO11-cls."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from pharma_inspect.data import label_name
from pharma_inspect.evaluation import (
    STRATEGIES,
    aggregate,
    apply_threshold,
    performance_to_lines,
)
from pharma_inspect.pipeline import load_config
from pharma_inspect.vlm import build_client
from pharma_inspect.vlm.prompts import build_prompt, system_prompt
from pharma_inspect.visualization import compose_comparison, overlay_activation
from pharma_inspect.yolo.explain import activation_map
from pharma_inspect.yolo.inference import letterbox_preview, predict_batch
from pharma_inspect.yolo.model import load_classifier, resolve_path


def read_images(uploads: list[Any], max_frames: int) -> tuple[list[Image.Image], list[str]]:
    """Carrega e limita as imagens enviadas.

    Args:
        uploads (list[Any]): Arquivos enviados pelo Streamlit.
        max_frames (int): Quantidade máxima de imagens usadas.

    Returns:
        tuple[list[Image.Image], list[str]]: Imagens válidas e erros por arquivo.
    """
    images = []
    errors = []
    for upload in uploads:
        try:
            images.append(Image.open(upload).convert("RGB"))
        except Exception as exc:
            errors.append(f"Arquivo inválido {upload.name}: {exc}")
    return images[:max_frames], errors


def render_controls(config: dict[str, Any]) -> tuple[float, str, int, bool, str]:
    """Exibe os controles da barra lateral.

    Args:
        config (dict[str, Any]): Configuração carregada do YAML.

    Returns:
        tuple[float, str, int, bool, str]: Limiar, agregação, máximo de frames,
            exibição de ativação e dispositivo de inferência.
    """
    inference_config = config.get("inference", {})
    with st.sidebar:
        st.header("Parâmetros")
        threshold = st.slider(
            "Limiar de decisão",
            min_value=0.0,
            max_value=1.0,
            value=float(inference_config.get("threshold", 0.5)),
            step=0.01,
        )
        configured_strategy = inference_config.get("aggregation", "mean")
        default_strategy = configured_strategy if configured_strategy in STRATEGIES else "mean"
        strategy = st.selectbox(
            "Agregação dos frames",
            options=STRATEGIES,
            index=STRATEGIES.index(default_strategy),
        )
        max_frames_value = inference_config.get("max_frames", 12)
        max_frames = st.number_input(
            "Máximo de frames",
            min_value=1,
            value=int(max_frames_value) if max_frames_value is not None else 12,
        )
        show_activation = st.checkbox(
            "Mostrar mapa de ativação",
            value=bool(inference_config.get("show_activation", False)),
        )
        device = st.selectbox(
            "Dispositivo de inferência",
            options=["cpu", "cuda:0"],
            index=0 if inference_config.get("device", "cpu") == "cpu" else 1,
        )
        st.caption(
            f"Entrada do YOLO: {config['model'].get('imgsz', 1280)}×"
            f"{config['model'].get('imgsz', 1280)} px com letterbox."
        )
    return threshold, strategy, max_frames, show_activation, device


@st.cache_resource(show_spinner="Carregando checkpoint YOLO...")
def cached_model(weights: str, device: str, class_names: tuple[str, ...]) -> Any:
    """Mantém o modelo em memória entre execuções da interface.

    Args:
        weights (str): Caminho absoluto do checkpoint.
        device (str): Dispositivo de inferência.
        class_names (tuple[str, ...]): Ordem das classes do checkpoint.

    Returns:
        Any: Modelo Ultralytics carregado.
    """
    config = {
        "model": {"weights": weights, "class_names": list(class_names)},
        "inference": {"device": device},
    }
    return load_classifier(config, root=ROOT)


def run_inference(
    images: list[Image.Image],
    config: dict[str, Any],
    weights_path: Path,
    device: str,
    threshold: float,
    strategy: str,
    show_activation: bool,
) -> dict[str, Any]:
    """Executa a inferência, a agregação e a visualização associada.

    Args:
        images (list[Image.Image]): Imagens válidas da sequência.
        config (dict[str, Any]): Configuração carregada do YAML.
        weights_path (Path): Checkpoint do YOLO.
        device (str): Dispositivo de inferência.
        threshold (float): Limiar de decisão.
        strategy (str): Estratégia de agregação.
        show_activation (bool): Se deve gerar o mapa de ativação.

    Returns:
        dict[str, Any]: Resultado com imagens, escores, decisão e metadados.
    """
    imgsz = int(config["model"].get("imgsz", 1280))
    class_names = tuple(config["model"].get("class_names", ("Clean", "Contaminated")))
    model = cached_model(str(weights_path), device, class_names)

    frame_scores = predict_batch(model, images, imgsz=imgsz, device=device)
    final_scores = aggregate(frame_scores, strategy)
    code = apply_threshold(final_scores, threshold)
    preview, _ = letterbox_preview(images[0], imgsz)

    activation = None
    activation_error = None
    if show_activation:
        try:
            map_array = activation_map(model, images[0], imgsz=imgsz, device=device)
            activation = overlay_activation(images[0], map_array)
        except Exception as exc:
            activation_error = str(exc)

    return {
        "image": images[0],
        "preview": preview,
        "comparison": compose_comparison(images[0], preview),
        "frame_scores": frame_scores,
        "final_scores": final_scores,
        "code": code,
        "decision": {
            "label": label_name(code),
            "probability": float(final_scores[1]),
            "threshold": threshold,
        },
        "frames": len(images),
        "strategy": strategy,
        "activation": activation,
        "activation_error": activation_error,
    }


def frame_statistics(frame_scores: Any, threshold: float) -> tuple[dict[str, float | int], list[dict[str, Any]]]:
    """Calcula estatísticas descritivas dos escores por frame.

    Args:
        frame_scores (Any): Matriz ``N, 2`` de escores por frame.
        threshold (float): Limiar de decisão.

    Returns:
        tuple[dict[str, float | int], list[dict[str, Any]]]: Resumo e linhas.
    """
    scores = np.asarray(frame_scores)
    contaminated = scores[:, 1]
    margins = np.abs(contaminated - threshold)
    decisions = (contaminated >= threshold).astype(int)
    summary = {
        "media": float(contaminated.mean()),
        "desvio": float(contaminated.std()),
        "minimo": float(contaminated.min()),
        "maximo": float(contaminated.max()),
        "margem_media": float(margins.mean()),
        "frames_clean": int((decisions == 0).sum()),
        "frames_contaminated": int((decisions == 1).sum()),
    }
    rows = [
        {
            "frame": index + 1,
            "clean": round(float(clean_value), 4),
            "contaminated": round(float(contaminated_value), 4),
            "decisao": label_name(int(decision)),
            "margem": round(float(margin), 4),
        }
        for index, (clean_value, contaminated_value, decision, margin) in enumerate(
            zip(scores[:, 0], contaminated, decisions, margins)
        )
    ]
    return summary, rows


def render_statistics(summary: dict[str, float | int], rows: list[dict[str, Any]], frame_scores: Any) -> None:
    """Exibe estatísticas e gráficos dos escores.

    Args:
        summary (dict[str, float | int]): Resumo calculado.
        rows (list[dict[str, Any]]): Escores detalhados por frame.
        frame_scores (Any): Matriz ``N, 2`` de escores por frame.

    Returns:
        None: Os elementos são renderizados no Streamlit.
    """
    st.subheader("Estatísticas dos frames")
    columns = st.columns(4)
    columns[0].metric("Média Contaminated", f"{summary['media']:.4f}")
    columns[1].metric("Desvio", f"{summary['desvio']:.4f}")
    columns[2].metric("Margem média", f"{summary['margem_media']:.4f}")
    columns[3].metric(
        "Frames contaminados", f"{summary['frames_contaminated']}/{len(rows)}"
    )
    st.caption(
        f"Contaminated: mínimo {summary['minimo']:.4f}, máximo {summary['maximo']:.4f}; "
        f"{summary['frames_clean']} frames Clean e "
        f"{summary['frames_contaminated']} frames Contaminated."
    )
    st.bar_chart(
        {
            "Clean": frame_scores[:, 0].tolist(),
            "Contaminated": frame_scores[:, 1].tolist(),
        }
    )
    st.dataframe(rows, width="stretch")
    st.caption("Escores softmax do YOLO, não probabilidades calibradas.")


def render_prediction(result: dict[str, Any], show_activation: bool) -> None:
    """Exibe a classe prevista e as visualizações associadas.

    Args:
        result (dict[str, Any]): Resultado retornado por ``run_inference``.
        show_activation (bool): Se a coluna de ativação foi solicitada.

    Returns:
        None: Os elementos são renderizados no Streamlit.
    """
    columns = st.columns([2, 1, 1] if show_activation else [2, 1])
    left, right = columns[0], columns[1]
    left.image(
        result["comparison"],
        caption="Imagem original e entrada quadrada do YOLO",
        width="stretch",
    )
    right.metric("Decisão", result["decision"]["label"])
    right.metric("Escore Contaminated", f"{result['decision']['probability']:.4f}")
    right.write(f"Limiar: {result['decision']['threshold']:.2f}")
    right.write(f"Frames analisados: {result['frames']}")
    right.write(f"Agregação: {result['strategy']}")
    right.progress(float(result["final_scores"][result["code"]]))
    right.write(f"Clean: {result['final_scores'][0]:.4f}")
    right.write(f"Contaminated: {result['final_scores'][1]:.4f}")
    if show_activation:
        if result.get("activation") is not None:
            columns[2].image(
                result["activation"],
                caption="Mapa de ativação do backbone",
                width="stretch",
            )
        else:
            columns[2].warning(result.get("activation_error") or "Ativação indisponível")
        columns[2].caption(
            "Visualização associada: não é anotação de partícula nem prova causal."
        )


def render_test_report(config: dict[str, Any]) -> None:
    """Exibe métricas e matriz de confusão do relatório do YOLO.

    Args:
        config (dict[str, Any]): Configuração carregada do YAML.

    Returns:
        None: O painel é exibido quando o relatório existe.
    """
    reports_dir = resolve_path(
        config.get("evaluation", {}).get("reports_dir", "artifacts/yolo_reports"), ROOT
    )
    report_path = reports_dir / config.get("evaluation", {}).get("report_name", "test.json")
    if not report_path.is_file():
        st.info(
            "Relatório de teste ausente: rode scripts/evaluate_yolo.py no GPU externo "
            "após o treino para gerar métricas e matriz de confusão."
        )
        return
    report = json.loads(report_path.read_text(encoding="utf-8"))
    metrics = report.get("metrics", {})
    frame_metrics = report.get("frame_metrics", {})
    performance = report.get("performance", {})
    with st.expander("Métricas do conjunto de teste"):
        st.metric("Acurácia por sequência", f"{metrics.get('accuracy', 0.0):.4f}")
        st.metric(
            "Recall Contaminated", f"{metrics.get('positive_recall', 0.0):.4f}"
        )
        st.metric("Acurácia por frame", f"{frame_metrics.get('accuracy', 0.0):.4f}")
        if performance:
            for line in performance_to_lines(performance):
                st.caption(line)
        matrix = metrics.get("confusion_matrix", [[0, 0], [0, 0]])
        st.subheader("Matriz de confusão por sequência")
        st.dataframe(
            {
                "Classe real": ["Clean", "Contaminated"],
                "Previsto Clean": [matrix[0][0], matrix[1][0]],
                "Previsto Contaminated": [matrix[0][1], matrix[1][1]],
            },
            hide_index=True,
            width="stretch",
        )
        st.caption(f"Protocolo: {report.get('protocol', {})}")
        st.json(metrics)


def render_vlm(config: dict[str, Any], image: Image.Image, result: dict[str, Any]) -> None:
    """Exibe a interpretação opcional do VLM sem alterar a decisão.

    Args:
        config (dict[str, Any]): Configuração carregada do YAML.
        image (Image.Image): Imagem original.
        result (dict[str, Any]): Resultado da inferência YOLO.

    Returns:
        None: O painel é exibido quando o VLM estiver configurado.
    """
    settings = config.get("vlm") or {}
    if not settings.get("enabled", False):
        st.caption("VLM desativado: espaço reservado para integração futura via API.")
        return
    try:
        client = build_client(config)
        if client is None:
            return
        description = client.describe(image, result)
    except Exception as exc:
        st.error(f"Falha na análise semântica (a decisão YOLO permanece válida): {exc}")
        return
    with st.expander("Análise semântica (VLM, auxiliar)"):
        st.json(description)
        st.caption(f"System prompt: {system_prompt}")
        st.caption(f"Prompt do resultado: {build_prompt(result)}")


def main() -> None:
    """Executa a interface de inferência.

    Returns:
        None: A função interage com o Streamlit e não retorna valor.
    """
    st.set_page_config(page_title="PharmaInspect", layout="wide")
    config = load_config(ROOT / "configs/default.yaml")
    st.title(config.get("app", {}).get("title", "PharmaInspect — YOLO11-cls"))
    st.caption(
        "Triagem binária de frascos com YOLO11-cls em alta resolução. "
        "A classe oficial vem do modelo visual; o VLM, quando ativo, é apenas auxiliar."
    )

    threshold, strategy, max_frames, show_activation, device = render_controls(config)
    uploads = st.file_uploader(
        "Selecione uma imagem ou várias imagens da mesma sequência",
        type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
        accept_multiple_files=True,
    )
    if not uploads:
        st.info("Nenhuma imagem selecionada.")
        st.stop()

    images, errors = read_images(uploads, max_frames)
    for error in errors:
        st.error(error)
    if not images:
        st.error("Nenhuma imagem válida foi carregada.")
        st.stop()

    weights_path = resolve_path(config["model"]["weights"], ROOT)
    if not weights_path.is_file():
        st.error(
            f"Checkpoint não encontrado: {weights_path}. "
            "Exporte o dataset, treine em GPU com scripts/train_yolo.py e copie o best.pt."
        )
        st.stop()

    try:
        result = run_inference(
            images, config, weights_path, device, threshold, strategy, show_activation
        )
    except Exception as exc:
        st.error(f"Falha na inferência: {exc}")
        st.stop()

    render_prediction(result, show_activation)
    summary, rows = frame_statistics(result["frame_scores"], threshold)
    render_statistics(summary, rows, result["frame_scores"])
    render_vlm(config, result["image"], result)
    render_test_report(config)


if __name__ == "__main__":
    main()
