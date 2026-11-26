# Decisões de arquitetura

Estado atual do PharmaInspect. Cada decisão registra a escolha, o motivo e a consequência para quem vai executar ou revisar o projeto.

## 1. YOLO11-cls como classificador único

**Escolha:** `yolo11n-cls` classifica cada frame; a decisão por frasco vem depois, pela agregação dos escores.

**Motivo:** evita implementação manual de perda, otimizador e loop de treino, mantendo a unidade operacional por sequência.

**Consequência:** a lógica própria fica restrita a dados, agregação, inferência, explicação e interface.

## 2. Divisão agrupada por sequência

**Escolha:** `group_split()` mantém todos os frames de um `sample_id` no mesmo subconjunto, estratifica pela classe binária e usa seed 42. A auditoria mostrou que `sample_id` = diretório não é uma unidade independente, então o manifesto oficial passou a ter um cluster visual por amostra: 344/73/76 em treino/validação/teste. `validate_split()` proíbe sobreposição e o MANIFEST.YAML registra zero pares RMS<2 entre splits.

**Motivo:** frames quase idênticos em diretórios diferentes inflariam a avaliação.

**Consequência:** a unidade de decisão é o cluster, não o diretório original. O teste oficial tem poucos clusters `Clean` (6) e baseline majoritário de 92,1%; ver [`VALIDAÇÃO.md`](VALIDAÇÃO.md).

## 3. Entrada 1280×1280 com letterbox

**Escolha:** `model.imgsz = 1280`, escala `min(1280/W, 1280/H)` com padding neutro. Para 1600×768: 1280×614 mais 333 linhas de padding acima e abaixo.

**Motivo:** o dataset tem 1600×768; redimensionar diretamente para um quadrado deformaria o frasco e resoluções menores descartariam detalhes de partículas.

**Consequência:** a geometria é calculada em `yolo/dataset.py`, o canvas quadrado é enviado ao modelo e o mesmo mapa é usado na interface e no recorte do mapa de ativação. Isso é necessário porque o transform padrão de classificação do Ultralytics aplica `Resize(shortest_edge)` e `CenterCrop`, que recortariam as laterais do frame original. O treino usa esse transform padrão; a inferência usa letterbox. A diferença é registrada como `inference.preprocessing` no relatório e tratada como limitação declarada, não como vantagem de resolução.

## 4. Agregação e limiar na decisão

**Escolha:** `mean`, `median`, `max` e `vote` combinam os escores por frame; `apply_threshold()` decide `Contaminated` pelo escore da classe 1.

**Motivo:** partículas podem aparecer em apenas parte da sequência, e a escolha da agregação deve ser feita em validação.

**Consequência:** a saída continua binária, sem categoria intermediária; o limiar nunca é ajustado no teste.

## 5. Exportação explícita para YOLO-cls

**Escolha:** `scripts/export_yolo_dataset.py` converte o manifesto em `data/yolo/{train,val,test}/{Clean,Contaminated}/` e grava `data.yaml` como metadado.

**Motivo:** o classificador do Ultralytics lê diretórios por classe, enquanto o projeto usa um manifesto por sequência.

**Consequência:** o wrapper de treino aceita o diretório raiz ou o `data.yaml`, valida `train/` e `val/` e mantém a ordem das classes fixa (`Clean` = 0, `Contaminated` = 1).

## 6. Treino em GPU externa, sem pesos versionados

**Escolha:** nenhum treino local; `scripts/train_yolo.py` salva o melhor checkpoint em `artifacts/yolo/`. O arquivo atual `yolo11n-cls.pt` pertence à rodada antiga (512) e não é o modelo oficial.

**Motivo:** a máquina de desenvolvimento não tem GPU e o checkpoint é grande.

**Consequência:** o treino oficial produz `yolo11n-cls-512.pt` e `yolo11n-cls-1280.pt` no mesmo dataset. Cada checkpoint deve preservar `train_args`, `train_results` e fitness; o protocolo em [`VALIDAÇÃO.md`](VALIDAÇÃO.md) define o critério de aceite.

## 7. Avaliação por frame e por sequência

**Escolha:** `scripts/evaluate_yolo.py` calcula métricas de classificação, matriz de confusão, intervalos e desempenho, e grava `artifacts/yolo_reports/test.json` (nome configurável). `max_frames: null` avalia todos os frames do cluster.

**Motivo:** a agregação muda a unidade de decisão; as duas visões precisam ser distinguíveis e o protocolo precisa ser auditável.

**Consequência:** o relatório registra hashes do manifesto, pesos, pré-processamento, limiar, agregação e latências brutas; a interface mostra apenas o relatório do checkpoint em uso.

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

## 11. Dataset oficial deduplicado

**Escolha:** `scripts/build_dedup_dataset.py` remove duplicatas exatas por SHA-256, agrupa sequências com frames quase idênticos (RMS<2 em miniatura 80×38) por union-find e exporta `data/dedup/` com split por cluster, `manifest.json`, `data.yaml`, links simbólicos e `MANIFEST.YAML`.

**Motivo:** a divisão original por diretório não garantia independência: 41/112 sequências de teste tinham vizinhos quase idênticos em treino/validação.

**Consequência:** `data/manifest`, `data/yolo_dir` e `inference.max_frames` do config apontam para o dataset novo. Hashes SHA-256 de ZIP, manifestos, config, script e arquivos exportados estão no MANIFEST.YAML. Não foi adotado DVC: para um projeto acadêmico, os hashes no YAML bastam e não exigem remote, cache ou comandos extras.

## 12. Rastreabilidade sem DVC

**Escolha:** cada rodada registra hashes em arquivos texto versionáveis, e não um sistema externo de versionamento de dados.

**Motivo:** o dataset é grande, local e não versionado; DVC adicionaria configuração e infraestrutura sem mudar a conclusão científica.

**Consequência:** qualquer pessoa com os mesmos arquivos recomputa os hashes e compara com o MANIFEST.YAML. Para publicar, basta anexar o arquivo e os hashes em `agents/VALIDAÇÃO.md`.

## 11. Intervalos de confiança na avaliação

**Escolha:** `evaluate_yolo.py` grava `metrics_ci95` (Wilson, por sequência) e `frame_metrics_ci95` (bootstrap por sequência, `--n-bootstrap`, padrão 1000).

**Motivo:** o teste tem poucas sequências `Clean` (24); uma métrica pontual de 100% esconde a incerteza. Wilson não degenera sem erros, e reamostrar frames isolados subestimaria a incerteza por causa da correlação dentro da sequência.

**Consequência:** resultados devem ser citados com o intervalo. Ver "Resultados no conjunto de teste" no README.
