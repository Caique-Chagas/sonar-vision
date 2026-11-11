# Treinamento e persistência

## Experimento executado

`configs/pilot.yaml`: YOLO11n pré-treinado oficialmente, 12 epochs, resolução
416, batch 4, patience 4, seed 42. Sinais pequenos motivam resolução acima do
smoke 320; dataset pequeno e 8 GiB de RAM motivam batch/epochs modestos.
Espelhamento foi desativado para não inverter texto. Foram usados pequenos
deslocamentos, escala/rotação e mosaic reduzido, sem alegar configuração ótima.
O melhor checkpoint foi selecionado pela validação; teste reservado não guiou tuning.

- Dataset: Bangkok, 120 famílias, 84/18/18, uma aumentação publicada por família.
- Classe: Bus Line Number remapeada para destination_sign.
- Hardware: Apple M1, MPS, 8 GiB RAM; Python 3.12.16 e PyTorch 2.15.0.
- Smoke separado: `artifacts/training/pilot-smoke` (1 epoch, 320, batch 2).
- Treino real: `artifacts/training/pilot-01/run`.
- Checkpoints: `weights/best.pt`, `weights/last.pt` e arquivos periódicos de epochs.
- Cópia de inferência: `models/best.pt`, ignorada pelo Git; obtenha a distribuição pela release e confira docs/MODEL_CARD.md.
- Evidências pequenas versionáveis: `reports/training/evidence.json`, `args.yaml`,
  `results.csv`, `results.png` e `reports/evaluation/detector_metrics.json`.
- Hash do melhor modelo: e132edf296f451b73295f077b54f39f94ce6e1cb214770ad04f14d1c6dee4e77.

Treino customizado foi realizado. Isso não torna o modelo validado para São Paulo.
A divisão garante famílias separadas e auditoria de duplicatas, mas identidade de
veículo/sequência não está disponível. Resultado é exploratório.
PyTorch advertiu que algumas operações MPS não são determinísticas; arquivos,
sementes e versões são registrados, sem promessa de reprodução bit a bit.

## Reproduzir ou retomar

Os comandos completos estão no README. `sonar-vision hardware` detecta CUDA/MPS.
O preparador exige um manifesto CSV com image,label,group_id,source,license.
As caixas devem estar em YOLO normalizado, classe única 0. Fonte e licença
precisam ter sido verificadas; o campo textual não substitui essa verificação.
Informe `--sequence-reviewed` somente após conferir agrupamento por filmagem/veículo.
Arquivos alterados depois da auditoria impedem o treino, pois hashes são rechecados.

```sh
.venv/bin/sonar-vision train --data data/bangkok-pilot/data.yaml --config configs/pilot.yaml --output artifacts/training/pilot-01 --device mps --resume artifacts/training/pilot-01/run/weights/last.pt
```

Use esse comando apenas para treino interrompido. Checkpoints finais podem ter
otimizador removido pela biblioteca e não suportar resume. Não reinicie um treino
interrompido automaticamente nem apague runs existentes. Verifique `evidence.json`
e o estado real de last.pt. Guarde dados e checkpoints no mesmo armazenamento
persistente; o checkpoint não contém as imagens.

## Colab preparado

1. Gere `.venv/bin/python scripts/package_source.py`.
2. Envie o ZIP para `Meu Drive/SONAR/sonar-vision-source.zip`.
3. Abra `notebooks/train_colab.ipynb` no [Colab](https://colab.research.google.com/).
4. Autentique sua conta, escolha runtime GPU disponível e monte o Drive.
5. Execute as células. O notebook verifica GPU; não inicia treino completo em CPU.
6. Dados e runs ficam em `Meu Drive/SONAR/sonar-vision/`; copie best.pt para
   `models/best.pt` na máquina de demonstração e preserve hash/evidências.

Notebook preparado e JSON validado localmente; **não executado no Colab** neste
experimento. Disponibilidade de GPU, cotas, preços e autenticação dependem da conta.
Não foi contratado serviço. O caminho público Bangkok não exige chave; exportação
alternativa Roboflow usa `.env`/variável ROBOFLOW_API_KEY, nunca o chat.

## Próximo experimento

Obter dados brasileiros autorizados, incluindo negativos e múltiplos veículos,
anotar caixas e códigos reais por encontro, separar por veículo/filmagem antes de
aumentar dados e reservar um teste independente. Ajustar limiares na validação;
medir precisão/recall do detector, OCR, confirmação errada, rejeição e latência
separadamente. `evaluation/metrics.py` fornece denominadores por encontro.
O protótipo não se declara validado enquanto essa etapa estiver pendente.

## Reprodução pela distribuição pública

Para executar o piloto sem repetir treinamento, instale os extras vision/paddle e
execute `python scripts/download_model.py`. O arquivo da release tem caminhos de
metadados normalizados, tensores idênticos ao checkpoint avaliado e hash próprio.
Os hashes e a transformação estão em MODEL_CARD.md. Nenhum treino é declarado
novo por essa transformação. Checkpoints de otimizador/intermediários não integram
a distribuição: para novos experimentos, use novos diretórios de saída e preserve
seus próprios arquivos best/last.

O notebook foi preparado, mas não executado no Colab. No formato ZIP, somente os
arquivos acadêmicos deste repositório são empacotados; autenticação do Drive e
escolha de GPU ocorrem na conta de quem executa.
