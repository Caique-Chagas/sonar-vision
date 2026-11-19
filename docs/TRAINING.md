# Treinamento customizado

## BR V1 — detector de letreiro usado no pipeline

YOLO11n COCO, adaptado a uma classe `destination_sign`; não inicializado pelo
modeloBangkok. Base:33fotografias brasileiras licenciadas,30grupos,29painéis e
4negativos. Split21/6/6imagens e19/5/5painéis. Anotação do módulo frontal principal
completo; revisão visual do agente, sem segunda anotação humana independente.
A base se chama commons-br-v2; a versão do detector resultante é BR V1.

40épocas,640px,batch2,MPS,seed42,workers0,AMPfalse,AdamW,lr0=0,001,lrf0,05,
warmup3,patience12,mosaic0,5,sem flips ou mixup. Tempo acumuladoCSV436,946s.
Melhor validação interna:mAP50=0,995,mAP50-95=0,7355. Não extrapolar esses números
à frota brasileira. Mesmos autores e cenários limitam independência.

Evidências completas numéricas: [CSV](../reports/training/br-v1/results.csv),
[args reais](../reports/training/br-v1/args.yaml),
[curvas](../reports/training/br-v1/results.png) e
[fontes/anotações](datasets/br-v1-sources.json).

## V3 — detector de ônibus usado na composição

YOLO11n com classes0`bus` e1`destination_sign`. Dados:26frames anotados,
16treino/5val/5teste. VeículosSantosval/teste distintos, mas cenário e autoria
correlacionados; não se alega independência estatística plena. SP pertence ao
conjunto de desenvolvimento/treino, portanto a demonstração não é teste externo.

Configuração40épocas máximas,960px,batch2,MPS,seed42,patience8. Early stopping
encerrou em9épocas; checkpoint selecionado anteriormente naépoca1. Não houve
retreinamento na preparação desta publicação. O V3 falhou no letreiro(0/5painéis
no pequeno teste anterior); a classe ônibus foi mantida e o letreiro usa BR V1.

[CSV](../reports/training/bus-v3/results.csv),
[args](../reports/training/bus-v3/args.yaml) e
[curvas](../reports/training/bus-v3/results.png).
Vídeos pessoais declarados pelo usuário, autorizados paraML no desenvolvimento;
originais e frames não são distribuídos. Sem esses dados, treinoV3exato não é
reproduzível pelo clone. Os pesos finais estão disponíveis, sem exigir treino.

## Baseline Bangkok

Preservado o piloto120famílias,84/18/18,12épocas,YOLO11n eMPS.
Teste18imagens/17caixas:precision0,936,recall0,471,mAP50=0,644,mAP50-95=0,288.
OCR8/9cropslegíveis;8outros excluídos por ilegibilidade. Não é resultado brasileiro.
[CSV e evidência](../reports/training/evidence.json),
[métricas](../reports/evaluation/detector_metrics.json) e releasev0.1.0-academic.

## Reprodução do processo

O pacote inclui preparação/validação de datasets e treino na CLI
`sonar-vision prepare`, `sonar-vision train`, `sonar-vision evaluate`.
Configurações e métodos adicionais estão em `sonar_br/` e `configs/`.
As fontes devem ser adquiridas e licenciadas antes de treinar; mantenha grupos
independentes, hashes e splits congelados. O notebookColab é uma alternativa
executável, não prova de que aquele ambiente foi utilizado.

Os pesos de runtime são exportações portáveis com tensores idênticos. Metadados de
caminhos/otimizador foram removidos; isso não constitui novo treinamento ou ganho
científico. [Modelo e licença](MODEL_CARD.md). Nenhum datasetRoboflow de direitos
fotográficos incertos foi usado no treinoBR V1.
