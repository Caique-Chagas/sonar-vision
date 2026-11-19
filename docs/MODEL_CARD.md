# Modelos finais e distribuição

Release [v0.2.0-fiap](https://github.com/Caique-Chagas/sonar-vision/releases/tag/v0.2.0-fiap)
contém somente dois pesos essenciais, checksums, licença, proveniência e validação
de portabilidade. [Manifesto](../reports/runtime-models.json).

| Peso | Arquitetura/classes | Uso |
|---|---|---|
| br-v1.pt | YOLO11n, destination_sign | Localização do letreiro |
| bus-v3.pt | YOLO11n, bus/destination_sign | Somente ônibus; não promover a classe letreiro |

Pesos originais foram preservados no desenvolvimento. A exportação para publicação
remove caminhos locais e metadados de treinamento dispensáveis; tensores,
formatos e shapes foram comparados porSHA256 e permaneceram idênticos.
Predições originais/portáveis foram comparadas no mesmo frame permitido, sem
retreinamento. [Recibo](../reports/model-portability-validation.json).
Alterar metadados muda o hashdoarquivo, por isso os hashes originais e públicos
são informados separadamente no manifesto. Isso não é uma melhoria do modelo.

BRV1 deriva deYOLO11n/Ultralytics e treino customizado em fontesCommons comlicenças
registradas. V3 deriva deYOLO11n e dados pessoais declarados pelo usuário paraML.
A publicação autorizada disponibiliza os parâmetros gerados, não as gravações.
Nenhuma procedência independente dessas gravações é inventada. [Dados](DATA_SOURCES.md).

Código e modelos do projeto:AGPL-3.0-only, preservando licença/atribuiçõesUltralytics.
Fotografias conservam suas licenças específicas. O baselineBangkok permanece na
releasev0.1.0-academic e não foi sobrescrito.

Limites: amostras pequenas, domínio restrito, nenhuma validação operacional de
linha/sentido/embarque, confiança não calibrada. CODE_STABLEé consenso textual.
Não usar autonomamente por passageiros. LerPTarquivos somente de fonte confiável:
checkpointsPyTorch podem conter objetos serializados. O downloader do projeto
confere hash e tamanho antes da instalação; não aceite um hash obtido de fonte
não confiável.

OCR usa modelos oficiais PaddleOCR(PP-OCRv5) obtidos pelo pacote na primeira
execução; eles não são reempacotados nesta release. Interfaces e tratamento de
falhas estão disponíveis no código, sem transcrição esperada embutida no runtime.
