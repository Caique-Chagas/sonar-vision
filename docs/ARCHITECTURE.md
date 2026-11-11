# Arquitetura e semântica dos contratos

## Execução

FastAPI atende requisições curtas; `SessionManager` coordena até duas sessões,
com no máximo uma real para limitar a memória. Um subprocesso spawn carrega YOLO
e PaddleOCR e entrega JPEG + eventos em fila de três itens. Cancelamento ou prazo
máximo termina o processo. Os últimos 200 eventos, um JPEG e até 1.200 latências
ficam em memória; no máximo 20 sessões são retidas, com descarte das finalizadas.
Não há banco ou armazenamento de frames. Reiniciar o serviço apaga as sessões.

`VideoSource` processa frames de forma incremental. O relay local recebe apenas
um caminho aleatório por sessão; busca o host HTTPS autorizado em IP público
validado e fixado, com SNI e certificado originais. Redirecionamentos não são
seguidos. Abertura/leitura têm timeout FFmpeg; limites de transferência e de
processamento encerram a fonte. Uma interrupção sem erro explícito do transporte
pode ser indistinguível de EOF; o sistema encerra, sem reconectar e carregar
evidências de um veículo anterior para outro. Não é um servidor de streaming público.

## Evidência visual

- YOLO só aceita pesos com classe destination_sign/LED-Panel. Pesos COCO genéricos
  não substituem o modelo customizado. A versão inclui SHA-256 do checkpoint.
- Caixas são recortadas ao frame e têm área positiva. OCR lê somente a ROI.
- Pré-processamento padrão: ROI original com escala limitada. CLAHE existe para
  experimentos, mas não foi escolhido como superior sem ensaio comparativo.
- Tracking IoU leve, TTL de um segundo; colisões/associações ambíguas geram IDs
  novos. Não há ReID, associação ônibus-painel ou garantia em oclusões/cruzamentos.
- O normalizador preserva o texto, trata maiúsculas, espaços/hífens e usa catálogo
  opcional. Abreviações e substituições OCR são sugestões incertas, mesmo quando
  existe apenas uma candidata no catálogo. Não infere sufixo ou sentido.
- Consenso padrão: pelo menos três frames distintos, intervalo mínimo 0,3 s,
  até oito observações em três segundos, proporção mínima 0,75, OCR ≥0,70 e
  detector ≥0,35. Última observação também precisa apoiar a vencedora.
- Valores são configuráveis por `SONAR_MIN_VOTES`, `SONAR_WINDOW_FRAMES`,
  `SONAR_MIN_RATIO`, `SONAR_MIN_OCR`, `SONAR_MIN_DETECTION`,
  `SONAR_EVIDENCE_SECONDS` e `SONAR_MIN_SPAN_SECONDS`. Não foram calibrados
  para operação. Ausências prolongadas invalidam a evidência; nenhum frame isolado confirma.

## VisionEvent 1.0

Implementação: `core/contracts.py`. IDs UUID e timestamp UTC com fuso.

| Campo | Semântica |
| --- | --- |
| status | CONFIRMED: código visual com consenso; UNCERTAIN: sem identificação suficiente |
| match_status | MATCH: código completo igual ao solicitado; NON_MATCH: código diferente; UNCERTAIN: leitura ou contexto insuficiente |
| detection_confidence | Média aritmética dos scores YOLO nos frames que apoiam o código confirmado; null sem consenso |
| ocr_confidence | Média dos scores OCR dos frames de apoio; no frame, menor score entre leituras que contêm candidatos; null sem consenso |
| frames_considered | Observações da janela para o mesmo track; inclui leituras vazias ou rejeitadas |
| supporting_frames | Quantas observações elegíveis apoiam o código vencedor; pode ser insuficiente para confirmar |
| raw_texts | Textos das observações na janela, preservados para auditoria; não contém imagens |
| destination_detected | Sempre null neste MVP; número não prova destino/sentido |
| source / simulated | simulation/true obrigatoriamente juntos; demais fontes false |
| model_version | Identificação SHA-256 do checkpoint, ou etiqueta explícita SIMULATED |

Confirmação visual pode coexistir com match_status UNCERTAIN se o código não
estiver no catálogo operacional fornecido. O anúncio continua incerto. Não existe
score contextual numérico: a correspondência é uma decisão categórica. Scores de
YOLO/OCR não são probabilidades calibradas e não são multiplicados.

O legado fornece espera_id, linha_lida, horario_da_passagem, tempo_parado=null e
confianca=ocr_confidence. `confianca_tipo=ocr_mean_uncalibrated` torna essa
adaptação explícita; não representa chance de embarque ou confiança geral.

## Áudio

Deduplicação no servidor por track/código/status e no navegador por event_id.
Uma mudança de incerto para confirmado pode gerar novo anúncio. Tracks perdidos
podem reaparecer como IDs novos; não há garantia de deduplicação entre essas
identidades. Repetição manual é uma ação explícita do usuário.
Web Speech pode depender de vozes locais ou serviço do navegador e de interação
do usuário. Verificação acústica/leitor de tela permanece necessária.

## Integrações

`RouteProvider` permite um provedor real SPTrans/GTFS futuro, sem implementação
fictícia de API. `FixtureRouteProvider` só atende simulação e se declara fictício.
Dados de Olho Vivo não são vídeo. Não existe dependência MQTT obrigatória.
Payloads POST são limitados a 16 KiB; a API não oferece uploads ou acesso a
arquivos locais. A CLI permite arquivo local quando solicitado explicitamente.
