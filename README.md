# SONAR VISION

[![QA](https://github.com/Caique-Chagas/sonar-vision/actions/workflows/qa.yml/badge.svg)](https://github.com/Caique-Chagas/sonar-vision/actions/workflows/qa.yml)

**Visão computacional para apoiar a identificação de linhas de ônibus.**

Projeto acadêmico do SONAR — Global Solution FIAP 2026.2, CitizenTech,
Engenharia de Software, disciplina **Applied Computer Vision**.

[Relatório PDF](reports/SONAR_VISION_FIAP.pdf) ·
[Vídeo demonstrativo narrado](https://github.com/Caique-Chagas/sonar-vision/raw/refs/heads/main/artifacts/demo/pilot-narrated.mp4) ·
[Legendas](artifacts/demo/pilot-narrated.srt) ·
[Modelo do piloto](https://github.com/Caique-Chagas/sonar-vision/releases/tag/v0.1.0-academic)

## Problema e objetivo

Uma pessoa cega ou com baixa visão pode conhecer a linha desejada e não conseguir
identificar o letreiro do ônibus que se aproxima. O SONAR VISION combina detecção
do letreiro, OCR e consolidação de leituras para emitir informação visual em texto,
eventos JSON e áudio, preservando incertezas.

**Protótipo acadêmico, com piloto treinado e avaliado em imagens de ônibus de Bangkok.**
Ainda não foi validado para identificação temporal de linhas brasileiras por veículo.
Identificar o código não confirma destino, sentido, parada ou condições de embarque.

## Escopo e tecnologias

- **YOLO11n / Ultralytics / PyTorch:** detector customizado da região do código.
- **PaddleOCR:** leitura de letras latinas e dígitos na região prevista.
- **OpenCV:** vídeo incremental, recortes, pré-processamento e caixas visuais.
- **Pydantic e FastAPI:** contratos validados e API de sessões.
- **HTML, CSS, JavaScript e Web Speech:** interface leve, teclado, texto e áudio.
- **pytest, Ruff, mypy e GitHub Actions:** verificações automatizadas.

Não implementa a plataforma completa SONAR, hardware, RPA ou integração oficial
SPTrans. O catálogo de demonstração é fictício e explicitamente identificado.

## Arquitetura

```mermaid
flowchart LR
    S[Vídeo] --> D[YOLO customizado]
    D --> T[Tracking de letreiros]
    T --> R[Extração de ROI]
    R --> O[PaddleOCR]
    O --> N[Normalização]
    N --> V[Consenso temporal]
    V --> M[Comparação de linha]
    M --> E[VisionEvent JSON]
    E --> A[API / interface / áudio]
```

A confirmação exige múltiplos frames distintos por track. Códigos abreviados,
correções ambíguas e leituras instáveis permanecem incertos. Detector e OCR mantêm
scores separados; não há uma probabilidade global inventada.
[Arquitetura e contratos](docs/ARCHITECTURE.md).

## Instalação e demonstração

Python **3.11 ou 3.12**. Execução local verificada em Python 3.12/macOS Apple M1;
a CI testa Python 3.11 e 3.12 no Ubuntu. Windows ainda não foi validado.

```sh
git clone https://github.com/Caique-Chagas/sonar-vision.git
cd sonar-vision
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install '.[dev,report]'
uvicorn sonar_vision.api.app:app --host 127.0.0.1 --port 8000
```

Abra **http://127.0.0.1:8000**, escolha **Simulação** e informe `971D-10`.
Esse modo funciona sem pesos e demonstra MATCH, NON_MATCH e UNCERTAIN com cenas
artificiais. Ele não comprova reconhecimento real desses códigos por YOLO/OCR.
A voz depende do navegador, das vozes instaladas e de interação do usuário.

No Windows, use `py -3.12 -m venv .venv` e `.venv\Scripts\activate`.
Após alterar o código, reinstale com `python -m pip install --no-deps .`.

### Inferência real

```sh
python -m pip install '.[vision,paddle]'
python scripts/download_model.py
SONAR_VIDEO_HOSTS=upload.wikimedia.org uvicorn sonar_vision.api.app:app --host 127.0.0.1 --port 8000
```

O script baixa os pesos da release e verifica SHA-256 antes de instalar em
`models/best.pt`. Os parâmetros numéricos são idênticos aos do piloto; somente
caminhos locais dos metadados foram normalizados para distribuição.
[Procedência, hashes e limitações do modelo](docs/MODEL_CARD.md).
O PaddleOCR obtém seus modelos oficiais no primeiro uso.

A API aceita URL **HTTPS direta** MP4/WebM/AVI/MJPEG, porta 443, em host autorizado.
Páginas YouTube, HLS, RTSP e webcam não são suportados neste MVP. A CLI aceita arquivo local:

```sh
sonar-vision infer --local --source amostra-autorizada.mp4 --route 971D-10 --max-frames 100
```

Uma fonte pública utilizada somente para testar transporte foi
[Road traffic in Ljubljana, Sounds of Changes, CC BY 3.0](https://commons.wikimedia.org/wiki/File:Road_traffic_in_Ljubljana.webm).
Nela não houve código confirmado. URLs e formatos têm limites documentados em
[arquitetura](docs/ARCHITECTURE.md); uma página web não é necessariamente um stream.

## Treinamento customizado e dados

O treinamento foi **realmente executado**, e não substituído por inferência COCO.
YOLO11n foi inicializado com pesos oficiais e ajustado à classe `destination_sign`,
remapeada de **Bus Line Number**, no
[Bangkok-Bus-Dataset, SAINT70239, CC BY 4.0](https://universe.roboflow.com/saint70239/bangkok-bus-dataset).

| Configuração | Experimento |
| --- | --- |
| Dados | 120 famílias de fotografias, uma versão publicada por família |
| Divisão | 84 treino / 18 validação / 18 teste |
| Modelo | YOLO11n, 12 épocas, 416 px, batch 4, seed 42 |
| Hardware | Apple M1, MPS, 8 GiB RAM |
| Seleção | Melhor modelo pela validação; avaliação reservada separada |
| Evidências | [Configuração](configs/pilot.yaml), [CSV](reports/training/results.csv), [gráficos](reports/training/results.png), [execução](reports/training/evidence.json) |

O YAML original usava as mesmas imagens em treino/validação/teste e foi substituído.
O preparador verifica labels, hashes e duplicatas entre divisões. Não existem IDs
confiáveis de veículo/sequência: separar famílias não comprova independência por
ônibus. Aumentações anteriores do publicador continuam presentes.

```sh
python scripts/download_bangkok_sample.py --count 120
sonar-vision prepare --manifest data/bangkok-pilot-raw/manifest.csv --output data/bangkok-pilot
sonar-vision train --data data/bangkok-pilot/data.yaml --config configs/pilot.yaml --output artifacts/training/smoke --smoke --device mps
sonar-vision train --data data/bangkok-pilot/data.yaml --config configs/pilot.yaml --output artifacts/training/pilot --device mps
sonar-vision evaluate --data data/bangkok-pilot/data.yaml --weights artifacts/training/pilot/run/weights/best.pt --output reports/new-evaluation --device mps
```

Em GPU CUDA, troque `--device mps` por `--device 0`. Treino completo em CPU é
recusado. Não é necessário baixar vídeos manualmente. Diretórios existentes são
preservados; use outros nomes para novos experimentos.
[Metodologia e Colab](docs/TRAINING.md) · [Fontes e licenças](docs/DATA_SOURCES.md).

## Resultados verificáveis e limites

| Avaliação | Resultado | Limite |
| --- | --- | --- |
| Detector — 18 imagens / 17 caixas de teste | Precision **0,9358**; recall **0,4706**; mAP50 **0,6439**; mAP50–95 **0,2884** | Dados estrangeiros; IDs de sequência desconhecidos |
| OCR — 9 recortes legíveis | **8/9 exatos (88,9%)** | Outros 8/17 recortes excluídos; transcrição preliminar com assistência de IA |
| Integração — 18 imagens completas | 6 regiões detectadas e 6 chamadas OCR | Frames isolados; nenhum consenso temporal foi fabricado |
| Vídeo estrangeiro completo | 193 frames processados; nenhum letreiro detectado | Não comprova OCR/identificação de linha em vídeo |
| Identificação brasileira por encontro / falso positivo / rejeição | **NÃO MEDIDA** | Falta conjunto temporal brasileiro independente com ground truth |

[Detector](reports/evaluation/detector_metrics.json) ·
[OCR e denominadores](reports/evaluation/ocr_metrics.json) ·
[Integração](reports/evaluation/pipeline_stills.json) ·
[Limitações e validação](docs/VALIDATION.md).

O recall é limitado; limiares temporais não foram calibrados para operação.
Scores não são probabilidades calibradas. Avisos de operações MPS não determinísticas
impedem prometer reprodução bit a bit, apesar das sementes e versões registradas.
A documentação registra desenvolvimento e transcrições preliminares com assistência
de IA, sem atribuir contribuições individuais que não foram documentadas.

## Testes e API

```sh
python -m pytest
ruff check .
ruff format --check src scripts tests notebooks
mypy src
python scripts/verify.py
```

A suíte de publicação contém **108 testes** (104 do piloto e quatro de integridade do download). Ela verifica software e casos de falha com backends
substituídos quando apropriado; não representa precisão de visão.
[Relatório de QA](docs/QA_REPORT.md) · [Resultados automatizados](reports/qa-results.json).

| Endpoint | Finalidade |
| --- | --- |
| `GET /health` | Disponibilidade e presença do modelo |
| `POST /api/sessions` | Iniciar análise real ou simulada com limites |
| `GET /api/sessions/{id}` | Estado, eventos e desempenho |
| `DELETE /api/sessions/{id}` | Cancelar |
| `GET /api/sessions/{id}/frame` | Último frame da demonstração |
| `GET /api/sessions/{id}/events` | Eventos JSON sem imagens |
| `GET /api/sessions/{id}/legacy` | Adaptação ao contrato SONAR |

Documentação local em `http://127.0.0.1:8000/docs`.
O serviço usa localhost, valida fontes para reduzir SSRF e limita bytes, FPS,
frames, duração, sessões e fila. Não oferece autenticação para exposição pública.

## Materiais acadêmicos e integrantes

- [Relatório em PDF](reports/SONAR_VISION_FIAP.pdf) e [fonte Markdown](docs/REPORT.md).
- [Vídeo narrado — 2min29s](https://github.com/Caique-Chagas/sonar-vision/raw/refs/heads/main/artifacts/demo/pilot-narrated.mp4), [legendas](artifacts/demo/pilot-narrated.srt), [transcrição](docs/VIDEO_NARRATION.md) e [créditos](artifacts/demo/pilot-narrated.credits.txt).
- [Gravação da interface simulada](artifacts/demo/simulation-browser.webm).
- [Roteiro e reprodução dos materiais](docs/DEMO_VIDEO.md).

O vídeo mostra treino, inferência real em imagem estrangeira e interface identificada
como simulação. Não demonstra confirmação de linha brasileira em movimento.
A voz sintetizada do vídeo não comprova a voz do navegador; revisão acústica,
legendas e tecnologias assistivas ainda precisam de avaliação humana.

| Integrante | RM |
| --- | --- |
| Caique Chagas Nitta | 551943 |
| Rodrigo Resende Viana | 551057 |
| Gabriel Torres Luiz | 98600 |
| Otavio Vitoriano Da Silva | 552012 |
| Eduardo Fedeli | 550132 |

Professores: **Arnaldo Viana e Paulo Rodrigues**. Prazo informado no briefing:
**19/11/2026, 23h55, horário de São Paulo**.

## Licença e integridade

Código sob [AGPL-3.0-only](LICENSE). O dataset Bangkok mantém CC BY 4.0; mídias
Wikimedia mantêm suas licenças e créditos. O código de terceiros do espelho do
dataset não foi incorporado. [Referências e licenças](docs/REFERENCES.md).

Dados completos e pesos não são versionados no Git. Os pesos distribuídos estão
na release; seu hash e a identidade dos tensores estão documentados no model card.
As evidências públicas têm somente caminhos locais normalizados e metadados de
máquina removidos, sem alterar resultados. `reports/SHA256SUMS` permite conferir
os materiais com `python scripts/verify_artifacts.py`.
