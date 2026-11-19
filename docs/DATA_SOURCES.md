# Datasets, procedência e licenças

## BR V1

33fotografias/30grupos de ônibus brasileiros,29painéis completos e4negativos;
split21/6/6imagens. O [manifesto público](datasets/br-v1-sources.json) inclui autor,
URL, licença, grupos, splits, hashes e geometria, sem redistribuir os pixels.
No treino21imagens:17CC BY2.0,2CC BY-SA3.0,1domíniopúblico,1CC BY3.0.
Fotos deRafaelDelazari concentram a base; não presumir diversidade nacional.

Foram revistos créditos/marca e fontesCommons/Flickr; registros de autoria incerta
foram excluídos. Licença declarada não é automaticamente prova sobre fotografias
de terceiros. O manifesto documenta a base usada; a página original continua a
fonte para condições de cada arquivo. As imagens mantêm suas próprias licenças,
incluindo atribuição e compartilhamento pela mesma licença quando aplicável.

Anotações: painel físico principal completo, não texto isolado. Um revisoragente,
sem segunda anotação humana independente. Mesmoautor/dia/local/veículo foi agrupado
quando identificável. Ausência de duplicatas SHA/dHash não prova independência
estatística. Não reutilizar benchmarks conhecidos como novos testes prospectivos.

## V3 e demonstração de vídeo

26frames rotulados,16/5/5treino/val/teste, derivados de vídeos pessoais declarados
pelo usuário com autorização deML. Vídeos, frames e caches privados não são
publicados; somente parâmetros, pesos gerados e saídas numéricas do projeto.
A autorização de treinamento não é apresentada como uma licença aberta das gravações.
Os dados privados impedem repetir o treinoV3exato ou auditar visualmente essas
passagens pelo clone. O código permite processar outros MP4s autorizados.

## Fotografia do README e figuras do PDF

Fasouzafreitas, [Embarque pelo lado esquerdo–Campinas](https://commons.wikimedia.org/wiki/File:Embarque_pelo_lado_esquerdo_-_Campinas_-_SP.JPG),
obra própria declarada noCommons, CC BY-SA3.0. É exemplo conhecido, não teste novo.
[Créditos e modificações](ASSET_CREDITS.md). Figuras adicionais são mantidas apenas
como fontes do relatório oficial; não são novas avaliações nem decoração.

## Bangkok histórico

[Bangkok-Bus-Dataset/SAINT70239](https://universe.roboflow.com/saint70239/bangkok-bus-dataset)
e [espelho do autor](https://github.com/llsrwsaint225/Bangkok-Bus-Dataset-YOLOV8),
licença declaradaCC BY4.0. Piloto usou120famílias,uma versão por origem,84/18/18.
O inventário anunciado e imagens aumentadas não equivalem a capturas independentes.

## Fontes não incorporadas

DatasetsRoboflow brasileiros com créditos de terceiros insuficientemente
esclarecidos e repositórios sem licença fotográfica adequada não foram usados
no treinoBRV1. Roboflow não é necessário para inferência. Nenhuma licença de mera
visualização é tratada como autorização para treinar ou redistribuir.
Código/pesos seguemAGPL-3.0-only; isso não relicencia imagens e vídeos de terceiros.
