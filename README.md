# PharmaInspect

Protótipo acadêmico de inspeção visual de frascos farmacêuticos. O sistema classifica cada imagem como `Clean` ou `Contaminated` com **YOLO11-cls**, agrega os frames de uma sequência em uma decisão por frasco e apresenta o resultado em uma interface Streamlit.

## O que o projeto faz

- Classifica frames em alta resolução (1280×1280) com `YOLO11n-cls`.
- Preserva a proporção original com letterbox; não distorce o frasco. O canvas já letterboxado é enviado ao YOLO, porque o transform padrão de classificação do Ultralytics recortaria as laterais do frame 1600×768.
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

A divisão é agrupada por sequência (seed 42, 70/15/15): 510 treino, 108 validação e 112 teste. Isso evita que frames quase idênticos da mesma sequência apareçam em subconjuntos diferentes. O detalhamento está em [`agents/AUDITORIA_DATASET.md`](agents/AUDITORIA_DATASET.md).

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
python scripts/export_yolo_dataset.py --output data/yolo
```

O manifesto fica em `data/processed/residual_manifest.json`; a exportação cria `data/yolo/{train,val,test}/{Clean,Contaminated}/` sem alterar os originais. Em ambientes sem links simbólicos, use `--copy`; para um ensaio rápido, use `--max-frames-per-sequence 2`.

### 2. Abrir a interface

```bash
streamlit run app.py
```

Sem um checkpoint treinado, a interface avisa que `artifacts/yolo/yolo11n-cls.pt` está ausente. Treino e avaliação são executados em GPU externa (Colab/Kaggle); veja a seção seguinte.

### 3. Rodar os testes

```bash
python -m pytest -q
```

## Treino e avaliação em GPU externa

Monte o repositório e instale `requirements.txt`. O treino usa somente o dataset exportado; a avaliação também precisa do manifesto e dos frames brutos para reconstruir os subconjuntos.

Na máquina que tem `data/raw` (ou na GPU, se o dataset estiver montado):

```bash
python scripts/build_manifest.py
python scripts/export_yolo_dataset.py --copy --output data/yolo
```

Na GPU, com `data/yolo`, `data/raw` e `data/processed` disponíveis:

```bash
python scripts/train_yolo.py --data data/yolo \
  --weights yolo11n-cls.pt \
  --output-weights artifacts/yolo/yolo11n-cls.pt \
  --device cuda:0
python scripts/evaluate_yolo.py --device cuda:0
```

O Ultralytics baixa os pesos iniciais na primeira execução. Copie de volta `artifacts/yolo/yolo11n-cls.pt` e `artifacts/yolo_reports/test.json` para usar a interface com o modelo treinado. Para outra variante, ajuste `--weights` e `model.variant` em `configs/default.yaml` juntos.

## Configuração

| Chave | Padrão | Uso |
|---|---|---|
| `seed` | `42` | Reprodutibilidade da divisão e do treino. |
| `splits` | `0.7 / 0.15 / 0.15` | Proporção por sequência. |
| `model.imgsz` | `1280` | Lado do canvas quadrado. |
| `model.weights` | `artifacts/yolo/yolo11n-cls.pt` | Checkpoint usado na inferência. |
| `training.epochs` / `batch_size` | `50` / `8` | Parâmetros do treino externo. |
| `inference.threshold` | `0.5` | Limiar aplicado ao escore de `Contaminated`. |
| `inference.aggregation` | `mean` | Agregação dos frames da sequência. |
| `inference.max_frames` | `12` | Limite de frames analisados por sequência. |
| `vlm.enabled` | `false` | Nenhuma chamada de API por padrão. |

## Estrutura

```text
app.py                         interface Streamlit
configs/default.yaml           configuração do pipeline
requirements.txt               dependências
scripts/
├── build_manifest.py          manifesto por sequência a partir do dataset
├── export_yolo_dataset.py     manifesto -> dataset YOLO-cls
├── train_yolo.py              treino Ultralytics (GPU externa)
└── evaluate_yolo.py           avaliação por frame e por sequência
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

## Limitações

- Nenhum checkpoint treinado acompanha o repositório; nenhuma métrica YOLO é apresentada antes do treino real.
- Escores softmax não são probabilidades calibradas. Limiar e agregação devem ser escolhidos em validação, nunca no teste.
- A decisão é binária: o projeto não identifica o material contaminante, não localiza partículas, não gera bounding boxes e não declara conformidade farmacêutica.
- O mapa de ativação é uma visualização associada: não usa gradientes, não é anotação de partícula e não prova causalidade.
- O VLM não tem adaptador implementado: ativá-lo sem implementação gera erro explícito, sem resposta inventada.

## Créditos

Dataset e artigo: AImageLab/Unimore (ICPR 2024), licença CC BY-NC-SA 4.0. Classificador: Ultralytics YOLO11-cls. Interface: Streamlit.
