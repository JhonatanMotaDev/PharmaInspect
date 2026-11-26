# PharmaInspect — pacote de reprodução

Branch `repro/minimal`: apenas o código e os documentos necessários para reproduzir o protocolo oficial de classificação de frascos farmacêuticos. A interface Streamlit, o módulo VLM e o mapa de ativação ficam no `main`.

Usamos o **ResiDual** ([página oficial](https://ditto.ing.unimore.it/residual/)), publicado por AImageLab/Unimore no ICPR 2024. O download exige cadastro. Licença [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/); uso acadêmico e não comercial. Publicação de referência: Rosati et al., *Identifying Impurities in Liquids of Pharmaceutical Vials*, [DOI 10.1007/978-3-031-78447-7_18](https://doi.org/10.1007/978-3-031-78447-7_18).

## Reprodução

O dataset não é versionado. Baixe o ZIP da página oficial e coloque em `data/raw/ResiDual_Dataset.zip`, descompactando em `data/raw/ResiDual_Dataset`. Não apague o ZIP: a verificação confere o hash dele. Depois, da raiz do repositório:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/build_manifest.py
python scripts/build_dedup_dataset.py
python scripts/verify_dataset.py
python scripts/run_official_pipeline.py --device cuda:0
```

O quarto comando precisa dar `"ok": true`; ele confere os hashes do ZIP, do config, dos manifestos e do script gerador, além dos 13.870 symlinks e das contagens por split. O quinto treina 512 e 1280, verifica os metadados dos checkpoints, faz a varredura de limiar/agregação em validação, congela os parâmetros em `configs/official.yaml` e avalia o teste. Se uma etapa falhar, corrija e rode com `--resume`.

Saídas em `artifacts/`: `yolo/yolo11n-cls-512.pt`, `yolo/yolo11n-cls-1280.pt`, `yolo_reports/validation.json`, `yolo_reports/test.json`, `yolo_reports/pipeline_summary.json` e logs em `pipeline_logs/`.

## Resultado esperado

- 730 sequências originais → 493 clusters visuais (RMS<2 em miniatura, union-find); 344 treino / 73 validação / 76 teste, seed 42.
- Zero pares quase idênticos atravessando splits.
- Baseline majoritário no teste: **92,1%**. O teste tem apenas 6 clusters `Clean`.
- O relatório final deve ser comparado a esse baseline, com matriz por cluster e por frame e análise de sensibilidade sem o maior cluster `Clean`.

Hashes de procedência, protocolo completo e limitações: [`agents/VALIDAÇÃO.md`](agents/VALIDAÇÃO.md). Conta do vazamento que motivou o dedup: [`agents/AUDITORIA_DATASET.md`](agents/AUDITORIA_DATASET.md). Ordem de execução e entregáveis: [`agents/TREINO_HANDOFF.md`](agents/TREINO_HANDOFF.md).

**Limitação central:** não existe conjunto externo (outra câmera/iluminação/sessão) e treino/inferência usam geometrias diferentes. O resultado demonstra separação neste conjunto, não generalização operacional.
