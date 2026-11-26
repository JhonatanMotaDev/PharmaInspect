# Handoff para a equipe de treino

Na máquina CUDA, a partir da raiz do repo:

```bash
.venv/bin/python scripts/run_official_pipeline.py --device cuda:0
```

O script faz tudo nessa ordem: verifica hashes e symlinks do dataset, treina 512 e 1280 (só `imgsz` muda), confere `epoch`, `best_fitness`, `train_args` e `train_results` dentro de cada `.pt` (o checkpoint antigo é reprovado aqui), roda a varredura oficial com o checkpoint de 1280, grava o `threshold` e a `aggregation` escolhidos em `configs/official.yaml`, avalia o teste e salva `artifacts/yolo_reports/pipeline_summary.json`. Logs em `artifacts/pipeline_logs/`. Se uma etapa falhar, corrija e rode com `--resume` para não repetir o que já terminou.

Copie o repo com os symlinks intactos (`rsync -a --links`). Se o filesystem não mantiver atalhos, regenere com `.venv/bin/python scripts/build_dedup_dataset.py` — os hashes têm que bater. O `validation.json` que vem no HD é `dry_run` e será sobrescrito; não use o checkpoint antigo `yolo11n-cls.pt` para nada.

Devolva os dois `.pt`, o `validation.json`, o `test.json`, o `pipeline_summary.json`, os logs e o `configs/official.yaml`. Compare sempre com o baseline de 92,1%. O teste tem só 6 clusters `Clean`, então mande matriz por cluster e por frame, com análise sem o cluster maior de 285 frames.
