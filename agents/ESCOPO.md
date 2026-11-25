# Escopo do PharmaInspect

## Objetivo

Construir uma aplicação interativa de visão computacional para triagem de frascos farmacêuticos a partir de imagens. A saída é binária: `Clean` ou `Contaminated`. O resultado é apresentado com escore, estatísticas por frame, visualizações e métricas de avaliação.

## Problema e dados

O usuário fornece uma imagem ou os frames de uma sequência de inspeção. Cada sequência representa um frasco; frames da mesma sequência são altamente correlacionados e não podem ser tratados como amostras independentes.

Fonte: **ResiDual** ([página oficial](https://ditto.ing.unimore.it/residual/)), [artigo ICPR 2024](https://doi.org/10.1007/978-3-031-78447-7_18), licença [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). Rótulos originais: `Clean`, `Glass`, `Plastic`, `Rubber`, `Sand`. `Clean` permanece como classe 0; os quatro tipos de partícula são agrupados em `Contaminated`, classe 1.

## Incluído

- Classificação por frame com YOLO11-cls em 1280×1280 com letterbox.
- Agregação dos escores por sequência: `mean`, `median`, `max` ou `vote`, com limiar configurável.
- Divisão treino/validação/teste agrupada por sequência, estratificada e com seed 42.
- Comparação entre imagem original e imagem enviada ao modelo.
- Estatísticas por frame, gráfico de barras e tabela detalhada.
- Mapa de ativação do último bloco convolucional como visualização associada.
- Avaliação por frame e por sequência: acurácia, precisão, recall, F1, matriz de confusão, latência média, p95 e FPS.
- Interface Streamlit com seleção de entrada e ajuste de parâmetros sem editar código.
- Extensão opcional de VLM por API, separada da decisão visual e desativada por padrão.

## Não incluído

- Identificar se o contaminante é vidro, plástico, borracha ou areia.
- Localizar partículas, gerar bounding boxes ou máscaras.
- Estimar tamanho, massa ou concentração de contaminantes.
- Declarar conformidade regulatória ou substituir inspeção humana em produção.
- Processamento em tempo real ou captura ao vivo.

## Fluxo

1. O manifesto por sequência é gerado a partir do dataset somente leitura.
2. A divisão é derivada por sequência, com verificação de sobreposição.
3. O dataset é exportado no formato YOLO-cls (`train`, `val`, `test` × `Clean`, `Contaminated`).
4. O classificador é treinado em GPU externa; o checkpoint final volta para o repositório local.
5. A interface infere os frames enviados, agrega os escores e aplica o limiar.
6. A avaliação grava métricas e matrizes de confusão no conjunto de teste.
7. O VLM, quando habilitado, apenas comenta a imagem e o resultado estruturado.

## Parâmetros na interface

Limiar de decisão, estratégia de agregação, máximo de frames por sequência e exibição do mapa de ativação. Os valores padrão estão em `configs/default.yaml`.

## Saída e interpretação

A classe oficial vem do YOLO. O valor por classe é um escore softmax, tratado como confiança do modelo e não como probabilidade calibrada. O mapa de ativação mostra regiões ativas do último bloco convolucional; não é técnica de atribuição por gradientes, não é anotação de partículas e não estabelece causalidade.

## Avaliação

As métricas são calculadas no conjunto de teste, separado por sequência. A escolha de limiar e agregação deve usar apenas validação. O relatório só existe depois de um treino real com o checkpoint correspondente; valores anteriores não são reaproveitados.

## Extensão multimodal

O VLM é auxiliar: descreve características visíveis e resume o resultado, sem alterar a classe oficial. `vlm.enabled: false` desativa qualquer chamada de API. Sem adaptador implementado, a ativação falha com mensagem explícita em vez de produzir uma resposta fictícia.

## Evidências esperadas

- Código executável e testes automatizados.
- Interface utilizável com entrada real do dataset.
- Checkpoint e relatório de teste gerados em GPU externa.
- Registro das decisões em `DECISOES_ARQUITETURA.md` e das características dos dados em `AUDITORIA_DATASET.md`.

## Limitações

Protótipo acadêmico, restrito ao ResiDual e ao domínio de aquisição observado. Imagens de outros equipamentos, iluminações ou câmeras podem não seguir o mesmo comportamento. Detalhes em [`README.md`](../README.md).
