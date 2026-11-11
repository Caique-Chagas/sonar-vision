# Qualidade e reprodução da edição acadêmica

Verificação local: **PASSED**, 2026-11-10T20:59:09.911118-03:00.
Ambiente novo e independente, Python 3.12/macOS, instalado a partir desta cópia
com `pip install '.[dev,report]'`, sem depender da instalação do desenvolvimento.

| Check | Resultado | Evidência |
| --- | --- | --- |
| pytest | 108 testes aprovados, sem falhas; 7.355 s | reports/pytest.xml e reports/qa/tests.log |
| Cobertura de statements do pacote | 80.88% | reports/coverage.json |
| Ruff lint/formato | PASS | reports/qa/lint.log; format.log |
| mypy strict | PASS, 35 arquivos de fonte | reports/qa/types.log |
| pip check | PASS | reports/qa/dependencies.log |
| Download de modelo | Hash, limite de tamanho e preservação de arquivo existente testados | tests/unit/test_model_download.py |

São 104 testes do piloto e quatro testes de integridade de distribuição.
Os testes usam backends substituídos quando apropriado; não medem precisão de
visão. A cobertura é de statements, não de branches. Um aviso de depreciação de
compatibilidade Starlette/httpx foi registrado; não houve teste falhando nem
supressão do aviso. NumPy foi limitado a <2.4 para compatibilidade de tipos e PaddleX.

## Evidências anteriores preservadas

Modelo YOLO11n realmente treinado: 12 épocas, 84/18/18 famílias de Bangkok.
Detector em teste: precision 0,935762; recall 0,470588; mAP50 0,643910 e
mAP50-95 0,288400. OCR: oito de nove recortes legíveis exatos, com oito outros
excluídos. Não são medidas de identificação temporal brasileira por encontro.

A interface do piloto foi verificada em Chrome desktop/mobile, sem pageerror ou
transbordamento horizontal observado. Vídeo técnico de 148,75 s teve decodificação
completa de áudio/vídeo conferida; revisão acústica e assistiva humana pendente.
O vídeo registra 104 testes, quantidade existente quando foi produzido.

## Privacidade, procedência e integridade

Somente nomes/RMs dos integrantes fornecidos para a entrega são identificações
pessoais intencionalmente publicadas. Caminhos de máquina nos metadados e logs
foram normalizados; hostname foi retirado do JUnit, sem alterar contagens ou resultados.
Segredos, datasets completos, ambientes e arquivos privados não integram esta edição.
Os pesos da release têm metadados portáveis e todos os tensores iguais ao checkpoint
original; veja MODEL_CARD.md e reports/model-distribution.json.

A lista de hashes está em reports/SHA256SUMS. A normalização de caminhos não é
novo treinamento. Resultados numéricos históricos permaneceram inalterados.

## CI e limitações

O workflow testa Python 3.11 e 3.12 no Ubuntu, com Actions oficiais fixadas por SHA.
Resultados remotos: https://github.com/Caique-Chagas/sonar-vision/actions/workflows/qa.yml.
Windows, notebook Colab e validação de campo não foram executados.
Veja VALIDATION.md para métricas não medidas e restrições de uso.
