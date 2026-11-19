# Arquitetura

O pacote em `src/sonar_vision` separa detecção, OCR, normalização, associação porIoU,
validação temporal, contratos de evento, áudio, API e interface. Adapter de detector
permite modelos diferentes sem fornecer transcrição esperada aoOCR.

A composição desktop usa V3na classebus e BRV1na classedestination_sign, ambos no
frame original. A alternativa de recortar o ônibus antes do BRV1 é experimental,
não default: não melhorou o código do segmento selecionado. A ROIprevista pelo
BRV1 entra no OCR E3; fallbackdesktop só quando necessário. As confidênciasYOLO e
OCR permanecem separadas, sem serem apresentadas como probabilidades calibradas.

`desktop/presentation.py` prepara cada frame uma vez e salva predições ligadas aos
pixels porSHA256. `desktop/evidence.py` agrega observações distintas por veículo,
recusando duplicatas e origens interpoladas/rastreadas como votos. Os estados da
apresentação sãoNO_DETECTION(letreiro),PANEL_DETECTED,OCR_CANDIDATE eCODE_STABLE.
LINE_VERIFIEDé reservado a validação operacional real e nunca emitido na demo.

`desktop/render.py` desenha somente caixas do frame atual; `label_layout.py` evita
que legendas ocultem a ROI. `desktop/player.py` reproduzMP4pré-processado no relógio
original, sem modelos. O helperSwift mantém a janelaCocoa responsiva e pré-carrega
áudioAVFoundation. `speech/offline.py` sintetiza somente códigos estáveis usando
Luciana/macOS; o áudio é separado de narração acadêmica.

FastAPI e frontendleve oferecem simulação identificada, uploadMP4 e fotografia.
Uploads são limitados; processamento é isolado e recursos compartilhados por jobs.
A simulação não é evidência de visão. A API operacional preserva contratos de
incerteza e não confirma embarque. [Reprodução](REPRODUCIBILITY.md).

As ferramentas `sonar_br/` preservam métodos de preparação/anotação, diagnóstico
separado de detector/OCR e avaliação de exemplos congelados. Algumas dependem de
dados históricos ausentes do clone e não integram o caminho básico da CI/runtime.
