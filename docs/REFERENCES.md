# Referências técnicas consultadas

Consulta em 10/11/2026. Versões realmente instaladas ficam em
`reports/environment.txt`; a documentação online pode evoluir independentemente.

- [Ultralytics: YOLO11](https://docs.ultralytics.com/models/yolo11/): arquitetura nano requerida.
- [Treinamento](https://docs.ultralytics.com/modes/train/): train, device, checkpoints e resume.
- [Inferência](https://docs.ultralytics.com/modes/predict/): Results/boxes e processamento incremental.
- [Validação](https://docs.ultralytics.com/modes/val/): split=test, precision/recall e mAP.
- [Instalação](https://docs.ultralytics.com/quickstart/): pacotes e dependências OpenCV.
- [Licenciamento Ultralytics](https://www.ultralytics.com/license): AGPL e opção Enterprise.
- [Roboflow: exportação de versões](https://docs.roboflow.com/developer/export-data):
  SDK version.download, formato YOLO e chave privada; código não registra links assinados.
- [PaddleOCR 3.x](https://www.paddleocr.ai/main/en/version3.x/pipeline_usage/OCR.html):
  predict, rec_texts e rec_scores; orientação de documento desativada para letreiros.
- [Modelos de reconhecimento](https://www.paddleocr.ai/main/en/version3.x/module_usage/text_recognition.html):
  en_PP-OCRv5_mobile_rec para letras latinas e dígitos; nenhum destino é inferido.
- [OpenCV VideoCapture](https://docs.opencv.org/4.x/d8/dfe/classcv_1_1VideoCapture.html)
  e [timeouts](https://docs.opencv.org/4.10.0/d4/d15/group__videoio__flags__base.html):
  CAP_FFMPEG, abertura/leitura limitadas e release.
- [PyTorch MPS](https://docs.pytorch.org/docs/2.15/notes/mps.html): disponibilidade no Apple Silicon.
- [FastAPI](https://fastapi.tiangolo.com/): validação e API; processamento pesado é
  isolado em subprocesso, sem execução no event loop.
- [SpeechSynthesis / MDN](https://developer.mozilla.org/en-US/docs/Web/API/SpeechSynthesis):
  suporte variável, vozes locais e controle de reprodução.
- [GNU AGPL v3](https://www.gnu.org/licenses/agpl-3.0.html): texto integral em LICENSE.
- GitHub Actions oficiais: [checkout](https://github.com/actions/checkout),
  [setup-python](https://github.com/actions/setup-python) e
  [upload-artifact](https://github.com/actions/upload-artifact). Tags v7 verificadas
  pela API GitHub; workflow fixa SHAs completos. Resultado da execução remota consultável na aba Actions do repositório.

Este repositório adota AGPL-3.0-only para o código. Dados mantêm suas próprias
licenças e atribuições. Uma futura distribuição fechada/comercial precisa avaliar
as condições aplicáveis; não há licença comercial adquirida.

- [imageio-ffmpeg oficial](https://github.com/imageio/imageio-ffmpeg), consultado
  em 10/11/2026 para obter o executável FFmpeg multiplataforma. Extra opcional
  `video`, sem modificar o pipeline de inferência. Codificação local de MP4
  H.264/AAC; nenhum serviço de vídeo ou voz pago utilizado.
