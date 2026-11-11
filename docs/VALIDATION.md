# Validação e limites conhecidos

## Evidência disponível

- YOLO11n customizado realmente treinado em dados estrangeiros, com logs e CSV.
- Detector: 18 imagens/17 caixas reservadas, precision 0,935762, recall 0,470588,
  mAP50 0,643910 e mAP50-95 0,288400.
- OCR: 8/9 recortes legíveis exatos. Outros 8/17 candidatos foram excluídos por
  falta de transcrição confiável. Transcrição preliminar assistida por IA;
  necessita revisão humana e não representa acurácia global de todos os letreiros.
- Pipeline em 18 imagens completas: seis regiões e seis chamadas OCR, sem
  confirmação temporal de um frame isolado.
- Vídeo estrangeiro completo: 193 frames, sem letreiros detectados. Trecho Câmara
  SP: 80 frames iniciais de apresentação em estúdio, sem ônibus. São verificações
  de transporte/processamento, não demonstrações positivas de identificação.
- Regras de normalização, múltiplos tracks, contradições, expiração e contratos
  exercitadas com fixtures. Não confundir testes de software com métricas de campo.

## O que não foi comprovado

Identificação brasileira por encontro, falsos positivos por ônibus, rejeição correta,
associação em cruzamentos/oclusões e calibração temporal operacional: NÃO MEDIDOS.
A revisão acústica de Web Speech, uso com leitores de tela e avaliação com usuários
não foram realizados. O vídeo narrado é uma apresentação do piloto, com simulação
identificada; não comprova leitura de linha brasileira em movimento.

## Independência dos dados

O YAML original tinha treino/validação/teste no mesmo diretório e foi rejeitado.
A nova divisão separa 120 famílias e verifica hashes/dHash. Faltam IDs de
filmagem/veículo; vazamento semântico residual não pode ser descartado.
Resultados de treino não foram usados como substitutos do teste reservado.
Avisos MPS impedem prometer reprodução numérica bit a bit.

## Próxima avaliação recomendada

Usar encontros brasileiros autorizados, com códigos, caixas e IDs de sequência,
separação por veículo/filmagem antes das aumentações, exemplos negativos e teste
reservado. Calibrar limiares somente na validação e medir detector, OCR e decisão
por encontro separadamente, com denominadores, condições e hardware registrados.
O componente fornece informação sobre código, sem confirmar destino ou embarque.

## Compatibilidade de dependências

A edição pública limita NumPy a <2.4, compatível com o intervalo de PaddleX e
com a configuração de análise de tipos Python 3.11. A instalação de NumPy 2.5
em Python 3.12 introduziu stubs com sintaxe incompatível com esse alvo; o ajuste
é de dependência e não altera a implementação do pipeline ou métricas históricas.
