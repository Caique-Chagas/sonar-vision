# Fontes verificadas em 10/11/2026

## Critérios e decisão

Não foram baixados vídeos inteiros para treinamento. Apenas uma amostra limitada
de fotografias públicas e suas anotações foi transferida. Quantidades abaixo são
separadas entre declarações das páginas e contagem efetiva no repositório.

| Fonte | Conteúdo e quantidade verificada | Licença declarada | Acesso e decisão |
| --- | --- | --- | --- |
| [ICMC / AI-Visual-Impairment](https://github.com/icmc-data/AI-Visual-Impairment) | API GitHub: 7.892 arquivos; nenhum .txt/.json/.xml/.csv/.yaml de anotações no inventário. Imagens brasileiras e notebooks; não baixadas nem contadas por imagem. | Ausente: API retorna license=null. Visibilidade pública não basta. | HTTPS público. Não utilizado no treino; precisa esclarecer direitos e labels. |
| [s Workspace / bus-route-number](https://universe.roboflow.com/s-workspace-b9ro3/bus-route-number-zitdp) | Página: 153 originais, classe LED-Panel. Versão 1: 459 exemplos aumentados, 459 treino, 0 validação, 0 teste. Um thumbnail inspecionado: ônibus estrangeiro; não comprova códigos SPTrans. | CC BY 4.0 na página. | Exportação oficial exige chave de conta. Candidato opcional, não selecionado neste piloto. |
| [SAINT70239 / Bangkok-Bus-Dataset](https://universe.roboflow.com/saint70239/bangkok-bus-dataset) e [espelho do autor](https://github.com/llsrwsaint225/Bangkok-Bus-Dataset-YOLOV8) | Inventário no commit cacdb66543150261916da59eaa6ce34d89f6839e: 3.333 imagens de treino e 3.333 labels, mais 8 imagens auxiliares. Página Universe exibe 1.111 originais. Amostra efetivamente baixada/decodificada: 120 famílias, uma versão publicada por família. | README.dataset.txt e página declaram CC BY 4.0. | GitHub raw público, URLs fixadas por commit. Selecionado para piloto do detector; não para comprovar leitura brasileira. |
| [Bangkok type2](https://universe.roboflow.com/saint70239/bangkok-bus-dataset-type2) | Página: 1.111 originais, 7 classes; exemplos de destinos e tipos de ônibus. Sem download. | CC BY 4.0 declarada. | Alternativa não usada; nenhuma métrica incorporada. |
| [Road traffic in Ljubljana](https://commons.wikimedia.org/wiki/File:Road_traffic_in_Ljubljana.webm) | Página: 46,564 s, 1.920×1.080, 30.513.600 bytes; ônibus europeu. | CC BY 3.0, Sounds of Changes. | URL direta Wikimedia candidata a teste de streaming, sem acesso ao YouTube; resultado efetivo registrado em QA. |
| [Apartes / SP 24h / Câmara SP](https://commons.wikimedia.org/wiki/File:Apartes_Grandes_Reportagens_-_SP_24h_-_Bloco_1-3.webm) | Página: 917,265 s, 853×480, 61.748.718 bytes. Testados 40 s iniciais: apresentação em estúdio, sem letreiros. | CC BY 3.0, Câmara SP. | Streaming direto público, 80 frames, 4.521.984 bytes; sem download integral. Sem anotações nem métricas de linha. Busca por offsets falhou no backend/relay; não é capacidade suportada da API. |

## Preparação da amostra Bangkok

- Scripts: `download_bangkok_sample.py` e `sonar-vision prepare`.
- Semente 42, 120 famílias; imagens originais podem já conter aumentações aplicadas
  pelo publicador. Não são apresentadas como fotografias originais sem transformação.
- Classes do publicador: BMTA-bus, Bus Line Number, Bus Side Number,
  Destination Sign, TSB-Bus. Apenas **Bus Line Number (1)** foi remapeada para
  `destination_sign (0)`: região visual que exibe o código. Números de frota e
  texto de destino tailandês foram excluídos.
- Alguns rótulos da classe 1 são polígonos. O envelope alinhado aos eixos é
  calculado a partir dos vértices normalizados; os originais ficam em `original-labels`.
- O YAML do espelho aponta treino/val/teste para o mesmo diretório: foi rejeitado.
- Nova divisão por família: **84 treino, 18 validação e 18 teste**. Hashes SHA-256
  de imagens/labels e manifesto em `artifacts/research/dataset-audit.json`.
- Auditoria rejeita duplicatas exatas e quase duplicatas entre splits (dHash,
  distância Hamming ≤3). Essa heurística não prova independência por veículo.
- Não há IDs confiáveis de vídeo/sequência. `split_assurance` registra
  `origin_group_only_sequence_unknown`. Possível vazamento semântico residual
  impede afirmar generalização operacional. Em novos dados, agrupar por
  veículo/filmagem antes de dividir e manter teste brasileiro reservado.
- Inspeção visual inicial: 12 imagens com caixas sobrepostas; inclui fotos diurnas,
  noturnas, placas impressas, displays e vistas traseiras/laterais. Ruído/sal e
  pimenta e distorções já estão presentes. Qualidade heterogênea, painéis pequenos.

## Atribuição e custos

Bangkok-Bus-Dataset: SAINT70239 / llsrwsaint225, repositório e projeto ligados acima;
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Modificações: subconjunto, remapeamento de classe, envelopes de polígonos e nova divisão.
O código do espelho não foi incorporado. Não foram contratados serviços, comprados
créditos nem enviados exemplos a inferência hospedada. O piloto usa hardware local.
Roboflow/Colab podem ter limites e custos; nenhum plano gratuito é presumido.
