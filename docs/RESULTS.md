# Resultados e limites

## Vídeos reais selecionados de desenvolvimento

[Contagens estruturadas](../reports/final-video-counts.json) e
[saídas por frame](../reports/video-evidence/). Os JSONL preservam as caixas
previstas, confidências, OCR bruto, código candidato, estado e timestamp, sem
pixels de vídeo ou caminhos pessoais. Os testes recalculam os totais.
Isso verifica consistência dos registros, não substitui inspeção visual independente.

| Perfil | Frames | Ônibus | Letreiro/OCR | Código completo | CODE_STABLE | Áudio |
|---|---:|---:|---:|---:|---:|---:|
| SãoPaulo854–866s |360|360|271|55corretos637P-10|21|1evento6,8s|
| SantosA |600|600|418|0|0|0|
| SantosB |600|600|452|0|0|0|

O código de referência de SãoPaulo foi verificado visualmente antes da avaliação
seletiva, sem fornecer a transcrição ao detector/OCR. Vídeo selecionado porque
apresenta maior cobertura e leituras repetidas, não benchmark de generalização.
Mesmo original/cenário foi usado no desenvolvimento; não é holdout independente.
O LED alterna código e destino;55/360não deve ser interpretado como acurácia da linha.
Nenhum candidato incorreto entre os55aceitos nessa passagem, com denominador
muito pequeno e selecionado para inferir taxa de falsos positivos geral.

Código estável:3votos,75%,8amostras,5Hz,span0,3s,OCR≥0,7,YOLO≥0,35,janela3s.
Associação exclusiva ao ônibus; duplicatas de pixels e origens tracking/interpolação
não contam como novas observações. CODE_STABLE não verifica linha operacional.
O texto da caixa é o OCR do frame atual, não a transcrição manual ou o voto temporal.

Persistem23transições de painel detectado para ausente no SP. IoUadjacente mediana
0,92856/n247pares; sem gabarito exaustivo para afirmar qualidade geométrica geral.
Caixas não foram mantidas artificialmente na ausência. O layout eliminou270frames
com sobreposição de etiquetas emSP,452emSantosB e39pareslabel/ROIemSantosA;
zero sobreposições/oclusões nos1560frames após mudança. Métrica de layout, não mAP.

Playback observado:59,2067/60,0381/29,8663FPS exibidos paraA/B/SP, mantendo10,01/
10,01/12s. Inferência originalSP:medianaCPU347ms/frame. Modelo não executa no
playback. Anúncio foi agendado em6,8s; exportAAC teve início acima de amplitude0,01
em6,80485s. VozLuciana offline; não é narração do aluno. Revisão de inteligibilidade
humana e avaliação assistiva com usuários permanecem pendentes.

## Fotografia estática licenciada

[Imagem no README](assets/ocr-campinas.jpg): Campinas, autorFasouzafreitas,
CC BY-SA3.0. BRV1 prevê painel comconfiança0,8163747; OCR retorna333TERMINAL com
confiançaaproximada0,997. É conhecido exemplo de desenvolvimento; sem código
completo ou consenso temporal. Não é resultado automático sobre um vídeo.

## Detecção e OCR históricos

O baselineBangkok teve mAP50=0,6439 e recall0,4706 em18imagens/17caixas;OCR8/9
no subconjunto legível,8outrosrecortes excluídos. Esses resultados não são brasileiros.
BRV1 teve validação interna pequena muito favorável e comportamento variável
fora desse domínio; não se omitem as falhas de transferência. V3deletreiro falhou,
por isso a composição usaBRV1. [Treinamento e fontes](TRAINING.md).

Nenhum resultado autoriza embarque. Não foram demonstradas integração operacional
SPTrans, confirmação de sentido/parada, representatividade nacional ou garantia
contra anúncios de códigos erroneamente estáveis em outros cenários.
