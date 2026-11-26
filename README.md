# PharmaInspect

Protótipo acadêmico de inspeção visual de frascos farmacêuticos. O sistema classifica cada imagem como `Clean` ou `Contaminated` com **YOLO11-cls**, agrega os frames de uma sequência em uma decisão por frasco e apresenta o resultado em uma interface Streamlit.

## O que o projeto faz

- Classifica frames em alta resolução (1280×1280) com `YOLO11n-cls`.
- Preserva a proporção original com letterbox; não distorce o frasco. O canvas já letterboxado é enviado ao YOLO, porque o transform padrão de classificação do Ultralytics recortaria as laterais do frame 1600×768. Essa é a geometria oficial de inferência; o treino usa o transform padrão do Ultralytics e a diferença está declarada em [`agents/VALIDAÇÃO.md`](agents/VALIDAÇÃO.md).
- Agrupa previsões por sequência (`mean`, `median`, `max` ou `vote`) e aplica um limiar configurável ao escore de `Contaminated`.
- Mostra classe, escore, estatísticas por frame, gráficos, matriz de confusão e um mapa de ativação opcional.
- Mantém um VLM via API como camada auxiliar, desativada por padrão; a decisão oficial é sempre do classificador visual.

## Dataset

Usamos o **ResiDual** ([página oficial](https://ditto.ing.unimore.it/residual/)), conjunto de sequências de frascos com líquido farmacêutico publicado por AImageLab/Unimore no ICPR 2024. O download exige cadastro na página oficial. A licença é [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/); o uso é acadêmico e não comercial.

Publicação de referência: Rosati et al., *Identifying Impurities in Liquids of Pharmaceutical Vials*, [DOI 10.1007/978-3-031-78447-7_18](https://doi.org/10.1007/978-3-031-78447-7_18).

Cada diretório de sequência contém frames do mesmo frasco. Os cinco rótulos originais são binarizados:

| Rótulo original | Classe usada | Código |
|---|---|---:|
| `Clean` | `Clean` | 0 |
| `Glass`, `Plastic`, `Rubber`, `Sand` | `Contaminated` | 1 |

A auditoria encontrou frames quase idênticos em sequências **diferentes**, então a divisão original por diretório não garantia independência. O dataset oficial agora é o deduplicado: 730 sequências são fundidas em 493 clusters visuais, divididos por cluster (seed 42, 70/15/15) em 344 treino, 73 validação e 76 teste. Nenhum par com RMS<2 atravessa splits. O teste oficial tem apenas 6 clusters `Clean` e 70 `Contaminated`; o baseline majoritário é 92,1%. Protocolo, hashes e métricas completas estão em [`agents/VALIDAÇÃO.md`](agents/VALIDAÇÃO.md) e em `data/dedup/MANIFEST.YAML`; a inspeção original está em [`agents/AUDITORIA_DATASET.md`](agents/AUDITORIA_DATASET.md).

> `data/` e `.venv/` são locais e não devem acompanhar o repositório. Coloque o dataset em `data/raw/ResiDual_Dataset` e gere o manifesto antes de executar o projeto.

## Requisitos

- Python 3.11 ou superior (testado em 3.14).
- Dependências listadas em `requirements.txt`, incluindo `ultralytics==8.4.218`, PyTorch, Pillow, NumPy, PyYAML, Streamlit e pytest.

## Instalação

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Como rodar

### 1. Preparar os dados

```bash
python scripts/build_manifest.py
python scripts/build_dedup_dataset.py
```

O primeiro comando gera `data/processed/residual_manifest.json`. O segundo produz o dataset oficial em `data/dedup/`: `MANIFEST.YAML` com hashes e métricas, `manifest.json` compatível com o pipeline, `data.yaml` e `{train,val,test}/{Clean,Contaminated}/` por links simbólicos. Os arquivos brutos não são alterados. Para restringir o agrupamento, use `--rms-threshold`; a nova execução substitui apenas uma saída gerada pelo próprio script.

### 2. Abrir a interface

```bash
streamlit run app.py
```

A interface lê `artifacts/yolo_reports/test.json`, configurável em `evaluation.report_name`. O checkpoint em `artifacts/yolo/yolo11n-cls.pt` é o modelo antigo (treinado em 512) e não deve ser usado como resultado oficial: ele foi avaliado no dataset vazado. Treino e avaliação do dataset oficial são executados em GPU externa (Colab/Kaggle); veja a seção seguinte.

### 3. Rodar os testes

```bash
python -m pytest -q
```

## Treino e avaliação em GPU externa

Monte o repositório e instale `requirements.txt`. O treino usa somente `data/dedup`; a avaliação também precisa de `data/raw` e do manifesto para reconstruir os clusters.

Na máquina que tem `data/raw`:

```bash
python scripts/build_manifest.py
python scripts/build_dedup_dataset.py
```

Copie `data/dedup` para a GPU junto com `data/raw` e `data/processed`. Rode **dois treinos**, mudando apenas a resolução:

```bash
python scripts/train_yolo.py --data data/dedup --imgsz 512 \
  --weights yolo11n-cls.pt \
  --output-weights artifacts/yolo/yolo11n-cls-512.pt --device cuda:0
python scripts/train_yolo.py --data data/dedup --imgsz 1280 \
  --weights yolo11n-cls.pt \
  --output-weights artifacts/yolo/yolo11n-cls-1280.pt --device cuda:0
```

Avalie no mesmo teste, com o mesmo limiar e agregação, escolhidos antes na validação:

```bash
python scripts/evaluate_yolo.py --weights artifacts/yolo/yolo11n-cls-1280.pt \
  --imgsz 1280 --device cuda:0 --output artifacts/yolo_reports/test.json
```

O Ultralytics baixa os pesos iniciais na primeira execução. Copie de volta o checkpoint e `artifacts/yolo_reports/test.json`. O protocolo completo de aceite está em [`agents/VALIDAÇÃO.md`](agents/VALIDAÇÃO.md).

## Configuração

| Chave | Padrão | Uso |
|---|---|---|
| `seed` | `42` | Reprodutibilidade da divisão e do treino. |
| `splits` | `0.7 / 0.15 / 0.15` | Proporção por cluster visual. |
| `data.manifest` | `data/dedup/manifest.json` | Manifesto oficial por cluster. |
| `data.dedup_manifest` | `data/dedup/MANIFEST.YAML` | Hashes, métricas e validação do dataset. |
| `data.yolo_dir` | `data/dedup` | Dataset YOLO-cls oficial. |
| `model.imgsz` | `1280` | Lado do canvas quadrado. |
| `training.epochs` / `batch_size` | `50` / `8` | Parâmetros do treino externo. |
| `inference.threshold` | `0.5` | Limiar aplicado ao escore de `Contaminated`; escolher em validação. |
| `inference.aggregation` | `mean` | Agregação dos frames da sequência; escolher em validação. |
| `inference.preprocessing` | `letterbox` | Geometria registrada no relatório de avaliação. |
| `inference.max_frames` | `null` | Sem truncamento: avalia todos os frames do cluster. |
| `evaluation.report_name` | `test.json` | Nome do relatório lido pela interface. |
| `vlm.enabled` | `false` | Nenhuma chamada de API por padrão. |

## Resultados no conjunto de teste (split antigo, com vazamento)

> **Não use como evidência de generalização.** Estes números foram medidos no split por diretório, em que 41 das 112 sequências de teste tinham frames quase idênticos a frames de treino/validação (ver [`agents/VALIDAÇÃO.md`](agents/VALIDAÇÃO.md)). O protocolo oficial é o dataset deduplicado em `data/dedup/`.

Checkpoint treinado em GPU externa e avaliado em CPU com `scripts/evaluate_yolo.py` (112 sequências de teste, 1.344 frames, `mean`, limiar 0,5, `imgsz` 1280). Os pesos e o relatório JSON não são versionados.

| Nível | Acurácia | Recall `Clean` | Recall `Contaminated` | Matriz `[[TN,FP],[FN,TP]]` |
|---|---|---|---|---|
| Sequência (n=112) | 100% (IC95% 96,7–100) | 100% (IC95% 86,2–100) | 100% (IC95% 95,8–100) | `[[24,0],[0,88]]` |
| Frame (n=1.344) | 98,5% (IC95% 97,7–99,2) | 93,1% (IC95% 90,6–95,6) | 100% | `[[268,20],[0,1056]]` |

- IC por sequência: Wilson. IC por frame: bootstrap reamostrando sequências inteiras (1.000 repetições, seed 42).
- Só há 24 sequências `Clean` no teste: "nenhum erro observado" não equivale a 100% de generalização.
- Em `imgsz` 512 a inferência piora (92,0% por sequência; recall `Clean` 62,5%), então 1280 é o tamanho correto. O `train_args` do checkpoint registra 512, divergência ainda não explicada.
- A latência foi medida com duas avaliações concorrentes em CPU e é apenas indicativa.

## Estrutura

```text
app.py                         interface Streamlit
configs/default.yaml           configuração do pipeline
requirements.txt               dependências
scripts/
├── build_manifest.py          manifesto por sequência a partir do dataset
├── build_dedup_dataset.py     dedup visual, split por cluster e MANIFEST.YAML
├── export_yolo_dataset.py     manifesto -> dataset YOLO-cls
├── train_yolo.py              treino Ultralytics (GPU externa)
├── run_official_pipeline.py   verifica, treina 512/1280, varre, congela e avalia
├── tune_validation.py         varredura de limiar/agregação no split de validação
├── evaluate_yolo.py           avaliação por frame e por sequência
└── verify_dataset.py          checagem de hashes e links do dataset oficial
src/pharma_inspect/
├── data.py                    manifesto, rótulos e divisão por sequência
├── evaluation.py              agregação, métricas e matriz de confusão
├── pipeline.py                contexto a partir de config + manifesto
├── visualization.py           composição e mapa de calor
├── yolo/                      dataset, inferência, explicação e treino
└── vlm/                       contrato da extensão opcional
tests/                         testes do fluxo
agents/                        escopo, decisões e auditoria do dataset
```

## Documentação

- [Escopo do projeto](agents/ESCOPO.md)
- [Decisões de arquitetura](agents/DECISOES_ARQUITETURA.md)
- [Auditoria do dataset](agents/AUDITORIA_DATASET.md)
- [Validação e protocolo oficial](agents/VALIDAÇÃO.md)
- [Handoff para a equipe de treino](agents/TREINO_HANDOFF.md)

## Limitações

- O checkpoint versionado é antigo (512) e foi avaliado no dataset vazado; não é evidência de generalização.
- O dataset deduplicado tem só 6 clusters `Clean` no teste: o baseline majoritário é 92,1% e o recall de `Clean` terá incerteza alta.
- O agrupamento é visual e heurístico (RMS<2 em miniatura), não uma identidade física de frasco.
- Treino e inferência usam geometrias diferentes (CenterCrop vs. letterbox); a diferença está declarada e deve ser tratada em experimento futuro.
- Escores softmax não são probabilidades calibradas. Limiar e agregação devem ser escolhidos em validação, nunca no teste.
- A decisão é binária: o projeto não identifica o material contaminante, não localiza partículas, não gera bounding boxes e não declara conformidade farmacêutica.
- O mapa de ativação é uma visualização associada: não usa gradientes, não é anotação de partícula e não prova causalidade.
- O VLM não tem adaptador implementado: ativá-lo sem implementação gera erro explícito, sem resposta inventada.

## Créditos

Dataset e artigo: AImageLab/Unimore (ICPR 2024), licença CC BY-NC-SA 4.0. Classificador: Ultralytics YOLO11-cls. Interface: Streamlit.
