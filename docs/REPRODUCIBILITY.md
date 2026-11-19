# Instalação e reprodução

## O que acompanha o clone

Código, testes com fixtures, configurações, CSVs de treinamento, métricas,
saídas numéricas dos vídeos e PDF oficial. Não acompanha vídeos pessoais,
datasets completos, caches, ambientes Python ou gravação acadêmica.
As contagens publicadas podem ser recalculadas pelos testes a partir dos JSONL,
mas o gabarito visual dos vídeos privados não pode ser reavaliado apenas pelo clone.

## Testar sem modelos nem vídeos

Python3.11 ou3.12:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install '.[dev,video]'
.venv/bin/python -m pytest
.venv/bin/ruff check .
.venv/bin/mypy src
.venv/bin/python -m pip check
```

A CI executa esse caminho em Ubuntu. Não usa GPU, mídia privada, Roboflow,
modelos ou tokens. Fixtures de imagem/vídeo são geradas pelos testes e não medem
qualidade de visão. Os testes de GUI/voz nativa foram realizados separadamente
no macOS; não se alega execução gráfica no Ubuntu.

## Pesos finais, sem treinamento

```sh
.venv/bin/python -m pip install '.[vision,paddle]'
.venv/bin/python scripts/download_runtime_models.py
```

O downloader confere tamanho/SHA256, aceita somente a release oficial e não
sobrescreve conteúdo diferente. Destinos históricos relativos foram mantidos
para compatibilidade com o pipeline:

| Modelo | Arquivo instalado |
|---|---|
| BR V1 | sonar_br/artifacts/train-v1/run/weights/best.pt |
| V3 ônibus | artifacts/demo/sonar-video-v3/training/full-f5829c9ab4/run/weights/best.pt |

As origens e checksums estão em `reports/runtime-models.json`. Para outro diretório
use `--root`; passe caminhos alternativos às APIs de detecção quando integrar o
código. O modelo Bangkok anterior continua disponível pela releasev0.1.0-academic
e por `scripts/download_model.py`; não é o detector final de letreiros brasileiros.

Dependências ML verificadas no desenvolvimento:Ultralytics8.4.209,
PyTorch2.15.0,PaddleOCR3.7.0. Os extras permitem intervalos de versões; diferenças
numéricas podem ocorrer. Para repetir o ambiente ML específico, instale essas
versões. PaddleOCR pode baixar seus modelos oficiais na primeira execução;
execução offline exige cache já instalado. Não é necessário refazer treino.

## Inferência real em exemplo permitido

O clone inclui `examples/campinas.jpg`,fotoCC BY-SA3.0sem anotações. Depois dos
pesos/dependênciasML, execute sem fornecer texto esperado ou caixa manual:

```sh
.venv/bin/python scripts/analyze_photo.py --image examples/campinas.jpg --output artifacts/photo-example
```

O resultado contém caixa prevista,ROI,OCRbruto eJSON. É fotografia estática conhecida,
não consenso emvídeo nem evidência independente de generalização. Fonte/créditos em
[examples/README.md](../examples/README.md). O output deve ser novo para preservar
experimentos anteriores. A figura anotada doREADME não é entrada do OCR.

## Preparar vídeos próprios ou autorizados

A apresentação desktop com voz foi validada emmacOS AppleM1, com OpenCV Cocoa,
FFmpeg e ferramentas CommandLineTools para compilar o helperSwift. Luciana(pt_BR)
precisa estar instalada para gerar anúncio offline de código estável. A síntese
é acessibilidade do aplicativo, não narração acadêmica.

Escolha segmentos de até20s de MP4s que você possa processar. Não substitua os
exemplos privados por download de terceiros sem permissão. Para habilitar os cinco
comandos de demonstração, prepare os três perfis:

```sh
.venv/bin/python scripts/prepare_demo.py --profile sao-paulo --video /caminho/sp.mp4 --start 0 --seconds 12 --authorized
.venv/bin/python scripts/prepare_demo.py --profile santos-a --video /caminho/a.mp4 --start 0 --seconds 10 --authorized
.venv/bin/python scripts/prepare_demo.py --profile santos-b --video /caminho/b.mp4 --start 0 --seconds 10 --authorized
./scripts/demo_final.sh
./scripts/demo_final.sh all
```

`/caminho/...` são entradas do usuário, não links de dados disponíveis neste repo.
Os nomes dos perfis são identificadores da interface; use conteúdo correspondente
ou adapte a integração. Não se promete que um novo vídeo terá detecção/OCR positivo.
O script gera inferências reais em todos os frames, depois aplica avaliação temporal,
layout, síntese elegível e exportaçãoAAC. O processamento não é em tempo real.
Concluídos os três perfis, o catálogo confere hashes e ativa o player existente.

Para apenas um perfil, a versão bruta pré-processada também pode ser aberta:
`./scripts/demo_final.sh sao-paulo --presentation-version historical`.
Essa versão não tem a nova avaliação temporal/áudio incorporada. Não reutilizar um
perfil com outro segmento silenciosamente: preserve o diretório existente.

O comando `--mode live` permanece disponível para os perfis do ambiente de
desenvolvimento/configuração local. O clone não cria automaticamente essas fontes.
Nenhuma reprodução de MP4 anotado carrega YOLO/OCR novamente.

## Limites de reprodução

É possível reproduzir código, testes, carregamento dos pesos e processamento de
novos vídeos autorizados. Não é possível reproduzir exatamente as contagens de
São Paulo/Santos ou o treinamentoV3 sem as gravações privadas. O manifestoBRV1
lista fotografias e licenças públicas, mas a disponibilidade futura das fontes
externas não é garantida. Não usar benchmark histórico como novo teste independente.

Após alterar o pacote, reinstale normalmente com `pip install --no-deps .`.
Os testes usam a fonte via configuraçãopytest; a instalação é verificada em
smoke separado. Nenhuma proteção de site.py ou arquivo.pth deve ser desativada.
