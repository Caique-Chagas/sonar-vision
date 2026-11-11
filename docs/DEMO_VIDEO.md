# Demonstração e gravação

## Material já produzido

- `artifacts/demo/pilot-narrated.mp4`: apresentação técnica de 148,75 s,
  1600×900, H.264/AAC, voz local Luciana pt-BR. Mostra treino real, métricas,
  detecção/OCR reais em imagem reservada e captura de interface explicitamente
  simulada. Não comprova reconhecimento temporal de linha brasileira.
- `pilot-narrated.srt`: legendas com temporização aproximada por capítulo,
  sujeitas à revisão. Créditos completos em `pilot-narrated.credits.txt` e
  transcrição em `docs/VIDEO_NARRATION.md`.
- `reports/demo-video.json`: decodificação completa de áudio/vídeo aprovada;
  áudio não silencioso, sem revisão acústica humana. `reports/demo-inference.json`
  preserva a caixa e os scores reais da imagem ilustrativa, que não é novo benchmark.

- `artifacts/demo/simulation-browser.webm`: gravação real da interface no Chromium,
  aproximadamente 13 segundos, sem áudio. **Conteúdo de visão simulado**,
  identificado na imagem e nos anúncios textuais. Mostra MATCH, NON_MATCH e UNCERTAIN.
- `artifacts/demo/simulation-desktop.png` e `simulation-mobile.png`: capturas
  efetivas de desktop/mobile; verificadas visualmente.
- `reports/real-video-session.json` e `real-video-full-session.json`: execução
  real em vídeo estrangeiro, sem detecções/identificação. Não são gravação de tela.
- `reports/brazil-exploratory-stream.json`: ensaio de 40 s iniciais da reportagem
  Câmara SP; 80 frames, sem detecções. Trecho inicial contém apresentação em
  estúdio, portanto não demonstra leitura de letreiro brasileiro.

Não apresentar a gravação simulada como demonstração de YOLO/OCR real.
O vídeo piloto narrado já está produzido. Ainda faltam revisão humana do áudio
e complementação com sequência brasileira anotada para comprovar a identificação.

## Roteiro sugerido — 3 a 5 minutos

1. **0:00–0:30 — Problema e limites.** Apresentar o passageiro que conhece a linha,
   mas não distingue o ônibus. Dizer: “Protótipo acadêmico de identificação;
   não confirma destino ou condições de embarque”.
2. **0:30–1:10 — Treino comprovado.** Mostrar configs/pilot.yaml, evidence.json,
   results.csv, gráfico real e best.pt. Explicar dataset estrangeiro, classe e
   divisão 84/18/18; não sugerir que foi treinado em São Paulo.
3. **1:10–2:00 — Execução real.** Mostrar uma fonte autorizada e o modo real.
   Explicar caixas, ROI e OCR. Se nenhum letreiro for identificado, mostrar
   essa falha com honestidade e apresentar resultados reais de imagens de teste.
4. **2:00–2:50 — Comportamento e integração.** Mostrar a simulação explicitamente
   rotulada, os três estados, eventos JSON e ausência de destino/tempo parado.
   Demonstrar teclado e anúncio sem instrução de embarque.
5. **2:50–3:30 — Métricas.** mAP50 0,644, recall 0,471, OCR 8/9 legíveis com oito
   exclusões. Explicar que são medidas distintas e não validam SPTrans.
6. **3:30–4:00 — QA e pendências.** Mostrar tests, relatório e próximo teste
   brasileiro. Exibir integrantes e link GitHub quando publicado.

## Gravar

1. Execute o servidor conforme README e abra no navegador.
2. Prepare uma sequência real licenciada/anotada e confira o código solicitado
   visualmente antes da gravação. Não use confirmação artificial como evidência real.
3. Ative voz, inicie com um clique e teste “Ouvir última informação”. Escolha
   voz pt-BR instalada quando disponível. Texto deve permanecer legível.
4. Grave com OBS ou ferramenta de captura do sistema, habilitando microfone e
   áudio do navegador conforme o sistema permitir. No macOS, gravação de tela
   pode exigir permissão local de privacidade que o agente não pode conceder.
5. Revise a gravação, incluindo falhas/limitações, e exporte MP4 com resolução
   suficiente para ler resultados. Remova telas com credenciais antes de publicar.
6. Publique no destino da entrega com a visibilidade exigida pela FIAP e verifique
   o link em uma sessão sem autenticação, quando a entrega exigir acesso público.

## Regravar a simulação automaticamente

```sh
.venv/bin/python -m pip install '.[browser]'
.venv/bin/playwright install ffmpeg
# Em máquinas sem Chrome instalado: .venv/bin/playwright install chromium
.venv/bin/python scripts/browser_smoke.py
```

A automação preserva a CSP e verifica o DOM pelo navegador; não desativa
proteções de segurança. A gravação headless é silenciosa e não comprova
disponibilidade de voz nem experiência com leitores de tela.

## Créditos de mídia

Bangkok-Bus-Dataset: SAINT70239, CC BY 4.0; imagens de teste e curvas próprias
de treino, conforme DATA_SOURCES. Road traffic in Ljubljana: Sounds of Changes,
CC BY 3.0. Apartes Grandes Reportagens — SP 24h — Bloco 1/3: Câmara SP,
CC BY 3.0, via Wikimedia Commons. Referencie os links das páginas de origem
e identifique caixas, recortes ou outras transformações ao usar essas mídias.

## Reproduzir o vídeo piloto narrado

```sh
.venv/bin/python -m pip install '.[vision,paddle,report,video]'
.venv/bin/python scripts/build_demo_video.py
```

A narração usa `say -v Luciana` no macOS. Outros sistemas precisam de áudio local
adequado. O script usa o modelo, uma imagem de teste, gráfico e captura já
preservados; não refaz treinamento. Quadros intermediários e áudio AIFF ficam
em artifacts/demo/build (ignorado pelo Git). O MP4 deve ser acompanhado dos
créditos e legendas. Não confundir a narração sintetizada com a voz da Web Speech.
