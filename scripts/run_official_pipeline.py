"""Executa o protocolo oficial completo em uma máquina CUDA.

Etapas, nesta ordem obrigatória: verifica o dataset, treina 512 e 1280, faz a
varredura oficial de limiar/agregação com o checkpoint novo, congela os
parâmetros em ``configs/official.yaml`` e avalia o teste. O config padrão não é
alterado, para que os hashes do MANIFEST.YAML continuem válidos.

Uso::

    .venv/bin/python scripts/run_official_pipeline.py --device cuda:0
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "artifacts/yolo_reports"
LOGS_DIR = ROOT / "artifacts/pipeline_logs"
CHECKPOINT_DIR = ROOT / "artifacts/yolo"


def parse_args() -> argparse.Namespace:
    """Define os argumentos da linha de comando.

    Returns:
        argparse.Namespace: Argumentos interpretados.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/default.yaml"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--pretrained", default="yolo11n-cls.pt")
    parser.add_argument("--resume", action="store_true", help="Pula etapas já concluídas")
    return parser.parse_args()


def sha256(path: Path) -> str:
    """Calcula o SHA-256 de um arquivo.

    Args:
        path (Path): Arquivo a ser lido.

    Returns:
        str: Digest hexadecimal.

    Raises:
        FileNotFoundError: Se o arquivo não existir.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_step(name: str, command: list[str], log_path: Path, skip: bool = False) -> bool:
    """Executa uma etapa espelhando a saída no terminal e em um log.

    Args:
        name (str): Nome da etapa.
        command (list[str]): Comando e argumentos.
        log_path (Path): Arquivo de log.
        skip (bool): Quando True, apenas informa que a etapa foi pulada.

    Returns:
        bool: True quando a etapa foi executada com sucesso ou pulada.

    Raises:
        SystemExit: Se o comando terminar com erro.
    """
    if skip:
        print(f"[pipeline] pulando {name}: saída já existe", flush=True)
        return False
    print(f"\n[pipeline] === {name} ===", flush=True)
    print(f"[pipeline] $ {' '.join(command)}", flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            print(line, end="", flush=True)
        code = process.wait()
    elapsed = time.perf_counter() - started
    log.write(f"\nExit code: {code}\nElapsed: {elapsed:.2f}s\n")
    if code != 0:
        raise SystemExit(f"[pipeline] {name} falhou (código {code}); veja {log_path}")
    print(f"[pipeline] {name} concluída em {elapsed:.1f}s", flush=True)
    return True


def is_official(path: Path) -> bool:
    """Verifica se um relatório de validação já é oficial.

    Args:
        path (Path): Arquivo JSON de validação.

    Returns:
        bool: True quando o arquivo existe e tem status oficial.
    """
    if not path.is_file():
        return False
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("status") == "official"
    except (json.JSONDecodeError, OSError):
        return False


def as_dict(value: Any) -> dict[str, Any]:
    """Converte namespace ou objeto simples em dicionário.

    Args:
        value (Any): Valor lido do checkpoint.

    Returns:
        dict[str, Any]: Dicionário equivalente, ou vazio quando não conversível.
    """
    if isinstance(value, dict):
        return value
    if hasattr(value, "__dict__"):
        return dict(vars(value))
    return {}


def inspect_checkpoint(path: Path, expected_imgsz: int) -> dict[str, Any]:
    """Confere os metadados de treino gravados em um checkpoint.

    Args:
        path (Path): Arquivo `.pt` a ser inspecionado.
        expected_imgsz (int): Resolução esperada no treino.

    Returns:
        dict[str, Any]: Metadados resumidos e resultado da checagem.

    Raises:
        FileNotFoundError: Se o checkpoint não existir.
    """
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        import torch
    except ImportError as error:
        raise SystemExit(f"[pipeline] torch indisponível para inspecionar {path}") from error

    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    checkpoint = as_dict(payload)
    train_args = as_dict(checkpoint.get("train_args"))
    train_results = as_dict(checkpoint.get("train_results"))
    train_metrics = as_dict(checkpoint.get("train_metrics"))
    epoch = checkpoint.get("epoch")
    fitness = checkpoint.get("best_fitness")
    fitness_value = float(fitness) if isinstance(fitness, (int, float)) else None

    checks = {
        "has_train_args": bool(train_args),
        "has_train_results": bool(train_results or train_metrics),
        "best_fitness_present": fitness_value is not None,
        "epoch_recorded": isinstance(epoch, (int, float)) and int(epoch) >= 0,
        "imgsz_matches": int(train_args.get("imgsz", expected_imgsz)) == expected_imgsz,
    }
    return {
        "path": str(path),
        "sha256": sha256(path),
        "epoch": int(epoch) if isinstance(epoch, (int, float)) else None,
        "best_fitness": round(fitness_value, 6) if fitness_value is not None else None,
        "train_imgsz": train_args.get("imgsz"),
        "checks": checks,
        "ok": all(checks.values()),
    }


def freeze_config(source: Path, destination: Path, selection: dict[str, Any]) -> None:
    """Grava o config com limiar e agregação escolhidos na validação.

    Args:
        source (Path): Configuração original.
        destination (Path): Configuração congelada a ser gravada.
        selection (dict[str, Any]): Escolha vinda da varredura.

    Returns:
        None: O arquivo de destino é sobrescrito.
    """
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    inference = config.setdefault("inference", {})
    inference["threshold"] = float(selection["threshold"])
    inference["aggregation"] = str(selection["aggregation"])
    inference["max_frames"] = None
    inference["imgsz"] = int(selection.get("imgsz", inference.get("imgsz", 1280)))
    destination.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )


def main() -> None:
    """Executa o pipeline completo e imprime o resumo final.

    Returns:
        None: O resumo é enviado para a saída padrão.
    """
    args = parse_args()
    python = sys.executable
    config_path = Path(args.config).resolve()
    frozen_path = ROOT / "configs/official.yaml"
    validate_path = REPORTS_DIR / "validation.json"
    test_path = REPORTS_DIR / "test.json"
    checkpoint_512 = CHECKPOINT_DIR / "yolo11n-cls-512.pt"
    checkpoint_1280 = CHECKPOINT_DIR / "yolo11n-cls-1280.pt"
    summary_path = REPORTS_DIR / "pipeline_summary.json"
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    run_step(
        "verificação do dataset",
        [python, str(ROOT / "scripts/verify_dataset.py")],
        LOGS_DIR / "00_verify.log",
    )

    for imgsz, destination in ((512, checkpoint_512), (1280, checkpoint_1280)):
        run_step(
            f"treino {imgsz}",
            [
                python,
                str(ROOT / "scripts/train_yolo.py"),
                "--config", str(config_path),
                "--data", str(ROOT / "data/dedup"),
                "--weights", args.pretrained,
                "--output-weights", str(destination),
                "--device", args.device,
                "--imgsz", str(imgsz),
            ],
            LOGS_DIR / f"train_{imgsz}.log",
            skip=args.resume and destination.is_file(),
        )

    if not checkpoint_512.is_file() or not checkpoint_1280.is_file():
        raise SystemExit("[pipeline] checkpoints ausentes após o treino")

    checkpoint_checks = {
        name: inspect_checkpoint(path, imgsz)
        for name, path, imgsz in (
            ("512", checkpoint_512, 512),
            ("1280", checkpoint_1280, 1280),
        )
    }
    for name, info in checkpoint_checks.items():
        print(f"[pipeline] checkpoint {name}: {json.dumps(info['checks'], sort_keys=True)}", flush=True)
    failed = [name for name, info in checkpoint_checks.items() if not info["ok"]]
    if failed:
        raise SystemExit(
            f"[pipeline] checkpoint(s) sem metadados de treino válidos: {', '.join(failed)}"
        )

    run_step(
        "varredura oficial",
        [
            python,
            str(ROOT / "scripts/tune_validation.py"),
            "--config", str(config_path),
            "--weights", str(checkpoint_1280),
            "--imgsz", "1280",
            "--device", args.device,
            "--status", "official",
            "--output", str(validate_path),
        ],
        LOGS_DIR / "tune_validation.log",
        skip=args.resume and is_official(validate_path),
    )
    validation = json.loads(validate_path.read_text(encoding="utf-8"))
    if validation.get("status") != "official":
        raise SystemExit("[pipeline] validation.json não é oficial; abortando")
    selection = validation["selection"]

    freeze_config(config_path, frozen_path, selection)
    run_step(
        "avaliação no teste",
        [
            python,
            str(ROOT / "scripts/evaluate_yolo.py"),
            "--config", str(frozen_path),
            "--weights", str(checkpoint_1280),
            "--imgsz", "1280",
            "--device", args.device,
            "--output", str(test_path),
        ],
        LOGS_DIR / "evaluate_test.log",
    )
    test = json.loads(test_path.read_text(encoding="utf-8"))

    summary = {
        "status": "official",
        "device": args.device,
        "selection": {
            "aggregation": selection["aggregation"],
            "threshold": selection["threshold"],
            "balanced_accuracy_validation": selection["balanced_accuracy"],
        },
        "frozen_config": str(frozen_path.relative_to(ROOT)),
        "checkpoints": {
            name: {
                "path": str(Path(info["path"]).relative_to(ROOT)),
                "sha256": info["sha256"],
                "epoch": info["epoch"],
                "best_fitness": info["best_fitness"],
                "train_imgsz": info["train_imgsz"],
                "checks": info["checks"],
            }
            for name, info in checkpoint_checks.items()
        },
        "validation": str(validate_path.relative_to(ROOT)),
        "test": str(test_path.relative_to(ROOT)),
        "test_metrics": test["metrics"],
        "test_frame_metrics": test["frame_metrics"],
        "baseline_majority_accuracy": 0.921053,
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print("\n[pipeline] resumo final")
    print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
