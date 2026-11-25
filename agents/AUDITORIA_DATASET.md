# Auditoria do dataset ResiDual

Resumo semântico da inspeção local do dataset usado pelo PharmaInspect. Fonte oficial: [ResiDual — AImageLab/Unimore](https://ditto.ing.unimore.it/residual/), publicado no ICPR 2024 ([artigo](https://doi.org/10.1007/978-3-031-78447-7_18)), licença [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). O dataset original foi tratado como somente leitura.

## O que é

Sequências de frames de frascos com líquido farmacêutico, adquiridos com câmera Matrix Vision sob iluminação LED. Cada diretório de sequência mostra o mesmo frasco em frames consecutivos. Há frascos limpos e frascos com partículas de vidro, plástico, borracha/silicone ou areia.

## O que foi medido

| Aspecto | Resultado observado |
|---|---|
| Estrutura | `ResiDual_Dataset/<classe>/<classe>_NNN/<frame>.jpg` |
| Classes originais | `Clean`, `Glass`, `Plastic`, `Rubber`, `Sand` |
| Sequências | 730 |
| Arquivos JPEG locais | 13.880 legíveis |
| Frames únicos | 13.870, conforme a página oficial; 10 arquivos locais são cópias exatas com sufixo `(1)` |
| Resolução e formato | 1600×768, JPEG, RGB, `uint8`, em 100% dos arquivos |
| Frames por sequência | 19 na maioria; `Sand_011` tem 28 e `Sand_012` tem 20, ambos com duplicatas internas |
| Após binarização | 153 sequências `Clean` (2.907 frames) e 577 `Contaminated` (10.973 arquivos) |

Não havia README, licença, metadados ou anotações dentro da árvore baixada; as informações de origem e licença vêm da página oficial.

## O que isso significa para o projeto

- **Unidade de amostra:** a sequência, não o frame. A divisão treino/validação/teste é agrupada por `sample_id`, estratificada e determinística (seed 42). Isso impede que frames do mesmo frasco apareçam em subconjuntos diferentes.
- **Desbalanceamento:** `Contaminated` tem cerca de 3,8 vezes mais frames que `Clean`. O relatório destaca o recall por classe, e pesos ou amostragem, se usados, ficam restritos ao treino.
- **Duplicatas:** as 10 duplicatas exatas são internas a duas sequências; não atravessam a divisão. Foram mantidas para preservar o dataset original.
- **Entrada do modelo:** a proporção 2,08:1 foi preservada com letterbox em 1280×1280, em vez de redimensionamento direto para quadrado.
- **Cor e contraste:** tons ciano e vermelho aparecem em todas as classes em proporções próximas. Isso descreve os arquivos, não explica a origem das cores nem identifica contaminantes.
- **Pré-processamento:** não há ROI fixa, filtro de contraste ou redução de ruído aplicados sem validação; a imagem original continua disponível para comparação.

## O que não foi determinado

Não há evidência local para afirmar que um diretório corresponde a um único frasco físico, qual é o intervalo entre frames ou se o modelo generaliza para outros equipamentos e iluminações. A inspeção visual das folhas de contato não foi registrada como conclusão humana. Nada aqui sustenta uso em produção farmacêutica.

## Uso

O manifesto `data/processed/residual_manifest.json` é o contrato derivado desta auditoria. Ele associa cada `sample_id` aos seus frames e ao rótulo original; `scripts/build_manifest.py` o reproduz a partir de `data/raw/ResiDual_Dataset`.
