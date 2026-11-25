# Decisões de arquitetura

Estado atual do PharmaInspect. Cada decisão registra a escolha, o motivo e a consequência para quem vai executar ou revisar o projeto.

## 1. YOLO11-cls como classificador único

**Escolha:** `yolo11n-cls` classifica cada frame; a decisão por frasco vem depois, pela agregação dos escores.

**Motivo:** evita implementação manual de perda, otimizador e loop de treino, mantendo a unidade operacional por sequência.

**Consequência:** a lógica própria fica restrita a dados, agregação, inferência, explicação e interface.

## 2. Divisão agrupada por sequência

**Escolha:** `group_split()` mantém todos os frames de um `sample_id` no mesmo subconjunto, estratifica pela classe binária e usa seed 42 (510/108/112 sequências). `validate_split()` proíbe sobreposição.

**Motivo:** frames de um mesmo frasco são quase idênticos; dividi-los aleatoriamente inflaria a avaliação.

**Consequência:** a avaliação por sequência é independente; a proporção 70/15/15 é viável nos dois grupos.

## 3. Entrada 1280×1280 com letterbox

**Escolha:** `model.imgsz = 1280`, escala `min(1280/W, 1280/H)` com padding neutro. Para 1600×768: 1280×614 mais 333 linhas de padding acima e abaixo.

**Motivo:** o dataset tem 1600×768; redimensionar diretamente para um quadrado deformaria o frasco e resoluções menores descartariam detalhes de partículas.

**Consequência:** a geometria é calculada em `yolo/dataset.py`, o canvas quadrado é enviado ao modelo e o mesmo mapa é usado na interface e no recorte do mapa de ativação. Isso é necessário porque o transform padrão de classificação do Ultralytics aplica `Resize(shortest_edge)` e `CenterCrop`, que recortariam as laterais do frame original.

## 4. Agregação e limiar na decisão

**Escolha:** `mean`, `median`, `max` e `vote` combinam os escores por frame; `apply_threshold()` decide `Contaminated` pelo escore da classe 1.

**Motivo:** partículas podem aparecer em apenas parte da sequência, e a escolha da agregação deve ser feita em validação.

**Consequência:** a saída continua binária, sem categoria intermediária; o limiar nunca é ajustado no teste.

## 5. Exportação explícita para YOLO-cls

**Escolha:** `scripts/export_yolo_dataset.py` converte o manifesto em `data/yolo/{train,val,test}/{Clean,Contaminated}/` e grava `data.yaml` como metadado.

**Motivo:** o classificador do Ultralytics lê diretórios por classe, enquanto o projeto usa um manifesto por sequência.

**Consequência:** o wrapper de treino aceita o diretório raiz ou o `data.yaml`, valida `train/` e `val/` e mantém a ordem das classes fixa (`Clean` = 0, `Contaminated` = 1).

## 6. Treino em GPU externa, sem pesos versionados

**Escolha:** nenhum treino local; `scripts/train_yolo.py` salva `best.pt` em `artifacts/yolo/yolo11n-cls.pt`.

**Motivo:** a máquina de desenvolvimento não tem GPU e o checkpoint é grande.

**Consequência:** o repositório não contém pesos treinados nem métricas de um modelo inexistente; a interface informa o checkpoint ausente.

## 7. Avaliação por frame e por sequência

**Escolha:** `scripts/evaluate_yolo.py` calcula métricas de classificação, matriz de confusão e desempenho, e grava `artifacts/yolo_reports/test.json`.

**Motivo:** a agregação muda a unidade de decisão; as duas visões precisam ser distinguíveis.

**Consequência:** a interface mostra apenas números do relatório gerado pelo checkpoint em uso.

## 8. Explicação por ativação, com ressalvas

**Escolha:** hook no último bloco convolucional do modelo efetivamente usado pelo predictor; média das magnitudes dos canais, remapeada para a imagem original.

**Motivo:** compatibilidade com o YOLO sem depender de gradientes.

**Consequência:** a interface declara que o mapa não é atribuição, não é anotação de partícula e não prova causalidade.

## 9. VLM separado e desativado

**Escolha:** `vlm/client.py` define apenas `describe(image, result)`; `vlm.enabled: false` desativa chamadas de API.

**Motivo:** a decisão oficial deve permanecer na pipeline visual, sem credenciais no repositório.

**Consequência:** sem adaptador, ativar o VLM gera erro explícito; com adaptador, a resposta é exibida separadamente.

## 10. Interface e verificação

**Escolha:** `app.py` apresenta classe, escore, comparação original/letterbox, estatísticas, matriz de confusão e ativação opcional. A verificação local usa 23 testes e `compileall`, sem treino.

**Motivo:** a revisão deve ser possível sem GPU e sem interpretar código para entender o resultado.

**Consequência:** comandos de execução estão no [`README.md`](../README.md); nenhum resultado numérico é inventado.
