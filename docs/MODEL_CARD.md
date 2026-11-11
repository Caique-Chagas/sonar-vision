# Modelo do piloto acadêmico

## Tarefa e procedência

YOLO11n ajustado para `destination_sign`, região visual de número de linha.
Inicialização oficial YOLO11n; treino real com 120 famílias do Bangkok-Bus-Dataset
(SAINT70239, CC BY 4.0), split 84/18/18, 12 épocas, 416 px, batch 4, seed 42,
Apple M1/MPS. Evidência em reports/training e métricas em reports/evaluation.

## Arquivo distribuído

Download público: [best.pt](https://github.com/Caique-Chagas/sonar-vision/releases/download/v0.1.0-academic/best.pt).
O script `python scripts/download_model.py` confere limite de tamanho e SHA-256,
sem sobrescrever pesos existentes de conteúdo diferente.

| Identificador | SHA-256 |
| --- | --- |
| Checkpoint original avaliado | e132edf296f451b73295f077b54f39f94ce6e1cb214770ad04f14d1c6dee4e77 |
| Arquivo distribuído | 9de9b7abdf4e543fe6976d0461110d02656973d78edeb7684882d3b904aaccc9 |
| Estado numérico dos tensores | 1a4dee74b27b6ffff148c8d1e4ffe4fe87076fd2b792117d09df93cf06a03ddd |

O arquivo público normaliza caminhos locais nos metadados serializados.
**Não houve retreinamento nem alteração dos parâmetros numéricos.** Todos os
nomes/dtypes/shapes/bytes do state_dict foram comparados, com o mesmo hash antes
e depois. A saída de inferência em frame de controle também permaneceu igual.
A evidência dessa transformação está em reports/model-distribution.json.

As métricas históricas referem-se ao hash original; o hash diferente da distribuição
resulta da serialização/metadados, não de um novo experimento ou melhora do modelo.
Configurações, métricas e logs publicados tiveram somente caminhos de máquina
normalizados; resultados numéricos foram preservados.

## Limites e uso

Precision 0,9358, recall 0,4706, mAP50 0,6439 e mAP50-95 0,2884 em 18 imagens
estrangeiras, 17 caixas. O recall é limitado. A divisão é por família de fotografia;
identidades de veículo/sequência não estão disponíveis. Não há validação temporal
brasileira ou garantia operacional. Não usar como autorização de embarque.

Código/modelo distribuídos sob as condições AGPL-3.0-only adotadas para Ultralytics;
dados mantêm CC BY 4.0. Não são redistribuídas coleções completas de imagens.
A release inclui hashes e esta procedência; pesos finais não substituem imagens
rotuladas nem o estado completo de um treinamento interrompido.
