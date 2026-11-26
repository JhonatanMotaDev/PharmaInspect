# Validação e protocolo oficial

Este documento fecha a preparação de dados e o protocolo de avaliação do PharmaInspect. Ele registra **qual dataset é oficial**, como ele foi construído, quais hashes o identificam e o que a equipe de treino deve entregar para que o resultado seja interpretável.

**Decisão:** o dataset oficial é `data/dedup/`. O treino de referência é `YOLO11-cls` em **1280×1280**; o braço 512 existe apenas como comparação controlada, com o mesmo split, seed e pré-processamento de treino.

**Situação dos resultados antigos:** os relatórios `test_512.json` e `test_1280.json` e o checkpoint `artifacts/yolo/yolo11n-cls.pt` foram produzidos no split antigo, que continha vazamento visual. Eles continuam disponíveis como histórico, mas **não são evidência de generalização**. O valor de 353 ms atribuído ao 1280 não existe em nenhum artefato: o relatório registra 160,68 ms.

**Estado da varredura:** `artifacts/yolo_reports/validation.json` existe, mas está marcado como `status: dry_run`. Ele foi gerado com o checkpoint antigo e **não define** limiar nem agregação oficiais. A escolha oficial deve ser refeita após o treino no dataset deduplicado, sobrescrevendo esse arquivo com `status: official`.

---

## 1. Por que o dataset mudou

A auditoria em [`AUDITORIA_DATASET.md`](AUDITORIA_DATASET.md) confirmou que a divisão por diretório de sequência não separa unidades visuais independentes: 41 das 112 sequências de teste tinham pelo menos um frame quase idêntico a um frame de treino/validação de outra sequência, e 20 delas entre os 12 frames efetivamente avaliados. Como não há `vial_id` no dataset público, a solução adotada foi agrupar por similaridade visual.

O problema atingia desproporcionalmente a classe `Clean` (18/24 sequências de teste, contra 23/88 contaminadas) — exatamente a classe cujo recall saltou de 62,5% para 100% entre 512 e 1280.

---

## 2. Método de deduplicação

Implementado em `scripts/build_dedup_dataset.py`, determinístico e sem dependências novas além das já usadas pelo projeto (Pillow, NumPy, PyYAML).

1. **Duplicatas exatas:** SHA-256 de cada JPEG; dentro de cada sequência, mantém-se a primeira ocorrência. As 10 cópias com sufixo `(1)` em `Sand_011`/`Sand_012` foram removidas. A fonte `data/raw` não é alterada.
2. **Assinatura visual:** miniatura 80×38 em tons de cinza e chave grosseira 20×10 quantizada em 4 bits, usada apenas para pré-selecionar candidatos.
3. **Quase duplicatas:** para cada par de frames de sequências diferentes na mesma chave grosseira, calcula-se o RMS na miniatura. Pares com **RMS < 2,0** são ligados.
4. **Agrupamento:** union-find transforma sequências conectadas em um cluster. Nenhum cluster mistura `Clean` e `Contaminated`; se misturasse, o script abortaria.
5. **Divisão:** `group_split()` estratificado por classe, seed 42, 70/15/15, aplicado aos **clusters**, nunca a frames. `validate_split()` proíbe repetição.
6. **Exportação:** `data/dedup/{train,val,test}/{Clean,Contaminated}/` por links simbólicos, com `data.yaml`, `manifest.json` (compatível com `load_index`) e `MANIFEST.YAML`.

O limite RMS<2 é um critério operacional, não uma verdade sobre identidade física: ele foi escolhido porque separa pares visualmente indistinguíveis (RMS mediano ~0,4–0,8 na região do frasco) de pares aleatórios entre sequências (mediana ~10,9).

---

## 3. Resultados medidos

| Métrica | Valor |
|---|---:|
| Sequências originais | 730 |
| Entradas de frame no manifesto original | 13.880 |
| Duplicatas exatas removidas | 10 |
| Frames retidos | 13.870 |
| Pares quase idênticos entre sequências (RMS<2) | 2.578 |
| Sequências envolvidas em algum par | 277 |
| Clusters finais | 493 |
| Clusters com mais de uma sequência | 40 |
| Maior cluster (sequências / frames) | 40 / 760 |
| Pares quase idênticos atravessando splits | **0** |
| Duplicatas exatas atravessando splits | **0** |
| Clusters com classes misturadas | **0** |
| Links simbólicos quebrados | **0** |

| Split | Clusters | `Clean` | `Contaminated` | Frames | Baseline majoritário |
|---|---:|---:|---:|---:|---:|
| Treino | 344 | 24 | 320 | 9.937 | 93,02% |
| Validação | 73 | 5 | 68 | 2.128 | 93,15% |
| Teste | 76 | 6 | 70 | 1.805 | 92,11% |

Contagens de arquivos exportados: treino 1.729 `Clean` / 8.208 `Contaminated`; validação 722 / 1.406; teste 456 / 1.349. Tempo de geração: ~20 s.

### Consequência estatística importante

O dataset deduplicado é **muito mais difícil** de avaliar do que o antigo: o baseline "sempre Contaminated" sobe de 78,6% para **92,1%**, porque sequências `Clean` quase idênticas foram fundidas em poucos clusters. O teste oficial tem apenas **6 unidades `Clean`**, e uma delas (`cluster_0018`) concentra 15 sequências e 285 frames. Qualquer recall de `Clean` terá intervalo largo e dependerá bastante dessa unidade. Por isso o relatório final deve mostrar:

- matriz por cluster **e** por frame;
- resultados com e sem `cluster_0018` (análise de sensibilidade);
- IC agrupado por cluster, nunca por frame tratado como independente.

---

## 4. Hashes de procedência

Sha-256 completos, também presentes em `data/dedup/MANIFEST.YAML`:

```text
raw_zip            1c71f907c9ba1a07336a63a70babf6436fe12903b5208c16f716b5409f14226b
original_manifest  6a5e5cc200dd11282b51ed1cd0ba4c3116ccfcba366b6b2e24d03a093d90b569
config             cedf8405f58ad048f159491f51a314982b8a55d2b283ba5260baf4d3874993da
generator_script   33ec8a3ddfdb0e6272a521a290eec5ec365d2667382272f735524ea2cf9880f3
portable_manifest  d55083a8bc7944ee49c8a7f939a50b5d8510a91ca345030a31bf68896689b1fd
data_yaml          c3184799062f640a644e5093be66c5a3224017b6ff40d191bc005b95a5245177
export_manifest    746641aa97db4674f3a4910f2c7384881851df0970207e1be4760a5cde9ae4e1
```

Cada cluster no MANIFEST.YAML traz também o SHA-256 de todos os seus frames, na mesma ordem de `frame_paths`. Isso permite detectar qualquer alteração posterior sem reabrir o ZIP.

**DVC:** não foi adotado. O projeto é acadêmico, os dados são locais e não versionados; DVC exigiria remote, cache e comandos extras sem mudar a conclusão científica. Os hashes no MANIFEST.YAML cumprem a rastreabilidade necessária. Se o projeto crescer, o próximo passo seria um `dvc.yaml` com um único stage apontando para `scripts/build_dedup_dataset.py` — não antes disso.

---

## 5. Protocolo de treino e avaliação

### 5.1 Treino

Dois braços, mesma máquina CUDA, mesmo dataset, mesma seed (42), mesmos epochs/batch/optimizer/lr; **apenas `imgsz` muda**:

```bash
python scripts/train_yolo.py --data data/dedup --imgsz 512 \
  --weights yolo11n-cls.pt --output-weights artifacts/yolo/yolo11n-cls-512.pt --device cuda:0
python scripts/train_yolo.py --data data/dedup --imgsz 1280 \
  --weights yolo11n-cls.pt --output-weights artifacts/yolo/yolo11n-cls-1280.pt --device cuda:0
```

Critérios de aceite do checkpoint:

- `train_args`, `train_results`, `train_metrics` e `best_fitness` preservados no `.pt` (o checkpoint antigo registra `epoch=-1`, `best_fitness=None`);
- seleção do melhor checkpoint por fitness/métrica de validação — nunca pelo último epoch e nunca pelo teste;
- log com GPU, versões de torch/CUDA/ultralytics, comando exato, seed e duração;
- ordem de classes conferida: `Clean=0`, `Contaminated=1`.

### 5.2 Escolha de limiar e agregação

Antes de abrir o teste:

1. rodar a avaliação no split de validação com os quatro aggregators (`mean`, `median`, `max`, `vote`) e um sweep de limiar;
2. escolher a combinação por balanced accuracy / recall de `Clean` na validação;
3. gravar `validation.json` com a varredura e a escolha;
4. só então congelar `inference.threshold` e `inference.aggregation` em `configs/default.yaml`.

A varredura é executada por:

```bash
python scripts/tune_validation.py --weights artifacts/yolo/yolo11n-cls-1280.pt \
  --imgsz 1280 --device cpu --output artifacts/yolo_reports/validation.json
```

Ela gera escores de todos os frames dos 73 clusters de validação, testa 4 agregações × 14 limiares (0,30 a 0,95 em passos de 0,05) e seleciona por balanced accuracy, com desempate por recall de `Clean`, acurácia e proximidade de 0,5. O relatório salva a grade completa, a matriz de cada combinação, os escores e os hashes do checkpoint e do manifesto.

#### Dry-run já executado — não usar como escolha oficial

Rodei a varredura com o único checkpoint disponível, o antigo (`21190e30…4a7f608d…`), em 73 clusters e 2.128 frames. Resultado: **49 das 56 combinações** chegaram a balanced accuracy 1,0 e matriz `[[5,0],[0,68]]`. O desempate escolheu `mean` + `0,5`, mas a escolha é arbitrária: `vote` acerta em toda a grade e `max` acerta a partir de 0,55.

Isso é exatamente o comportamento esperado quando o modelo já viu conteúdo quase idêntico aos clusters de validação. A varredura **não tem poder de discriminação** com esse checkpoint; portanto o `validation.json` atual está marcado como `dry_run` e deve ser regenerado com `status: official` depois do treino no dataset deduplicado. O config continua com os valores padrão (`mean`, `0,5`) até essa regeneração.

### 5.3 Avaliação no teste

```bash
python scripts/evaluate_yolo.py --weights artifacts/yolo/yolo11n-cls-1280.pt \
  --imgsz 1280 --device cuda:0 --output artifacts/yolo_reports/test.json
```

Regras fechadas:

- `inference.max_frames: null` — avalia **todos** os frames de cada cluster (média de 13,2; de 19 a 760);
- relatório por cluster e por frame, com escores brutos salvos;
- IC de Wilson por cluster e bootstrap agrupado pelo mesmo cluster (`cluster_bootstrap_intervals`);
- latências individuais em `latencies_ms`, com warm-up, CPU/threads, versões e hash do checkpoint registrados; FPS é derivado da latência média, não capacidade de linha de produção;
- hash do `manifest.json` avaliado é gravado no campo `protocol.dataset_manifest_sha256`;
- comparar sempre contra o baseline de 92,1%, não contra zero.

### 5.4 Pré-processamento: limitação declarada

Treino usa o transform padrão do Ultralytics (`Resize(shortest_edge)` + `CenterCrop`). Inferência/UI usam letterbox quadrado (`inference.preprocessing: letterbox`). Com frame 1600×768, o frasco aparece em escalas diferentes nos dois caminhos. A comparação 512 vs 1280 é válida como **comparação de resolução de inferência com pesos e protocolo idênticos**, mas não autoriza concluir que "treinar em 1280" foi a causa de uma eventual diferença. Para isso seria necessário um terceiro experimento com o mesmo pré-processamento nas duas pontas.

---

## 6. O que ainda não está resolvido

- Não existe identificador de frasco físico; o agrupamento RMS<2 é heurístico e pode tanto fundir frascos distintos quanto deixar passar cópias com mais ruído.
- O teste tem 6 unidades `Clean`; o recall dessa classe será impreciso mesmo com o protocolo correto.
- Não há conjunto externo (outra câmera, iluminação ou sessão), então o resultado mesmo assim não prova generalização operacional.
- O mismatch treino/inferência permanece declarado, não corrigido.
- Limiar e agregação ainda precisam ser escolhidos na validação e registrados antes do teste.

Enquanto esses pontos não forem endereçados, a conclusão científica permanece: o modelo pode separar bem **este** conjunto, mas a generalização em amostras independentes não está demonstrada.
