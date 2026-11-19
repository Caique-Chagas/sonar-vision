# Auditoria da base brasileira v2

33 fotos novas,30 grupos,29 painéis completos,4 negativos. Split congelado antes
modelo:21train/6val/6test; painéis19/5/5. Grupos positivos18/4/4. Mesmo autor/dia/local
agrupado nos casos correlacionados deSaritur,Vitoria eRio. SHA/dHash≤3 semsobreposição
com1546 registros anteriores ou entre novas fotos. Não garante independência
estatística nem representatividade de frota/autores. Maioria das fotos é deRafael
Delazari; participaçãoSPTrans limitada. Treino pequeno, elegível apenas para piloto.

Cada foto tem autor, licença, URL, data, dimensões, hash e geometria no manifesto
../datasets/manifests/commons-br-v2.review.json. Fotos próprias eFlickr com autoria
RafaelDelazari/marca correspondente e revisãoFlickreviewR registrada noCommons.
Não usar o nome da licença como prova automática de autoria de terceiros.
O arquivo de autor desconhecido e a foto com crédito/marca não esclarecidos foram
excluídos. Nenhum pixel bruto vai aoGit. CC BY exige crédito/link/alterações; derivados
CC BY-SA devem preservar licença compatível. Não tratar fotos comoAGPL do código.

Retângulos envolvem o módulo físico principal inteiro, incluindo envelope sob
perspectiva, não só as letras. Negativos são vistas traseiras/van sem esse módulo.
Cenas com painéis de fundo não anotáveis e iluminação decorativa ambígua foram
excluídas. Todos os overlays de29 painéis foram inspecionados pelo agente; não houve
segundo anotador humano. Não relatar concordância entre revisores.

Teste6imagens/5 painéis/5 grupos,4 códigos completos antesOCR:8677-10 (pixels8677/10),
679,508,201. A fotoNiturvia tem dígitos duvidosos: referência preliminar2701 foi
retirada antes de qualquer modelo. Não entra no denominador de códigos. Recibo
pre-inference-GT-correction.json preserva a correção, sem usarOCR como gabarito.

Exportação usa os validadores existentes de direitos/bordas, negativos explícitos,
grupo/split, hashes e quase duplicatas. FerramentaBR não reconverte textosRoboflow
em painéis.21 imagens de treino/18 grupos positivos ultrapassam o gate de piloto,
mas não sustentam uma alegação de generalização nacional ou prontidão operacional.
As10fotos do teste prospectivo anterior e val/testCommonsv1 não entram no novo dataset.


Os rótulos visam o painel frontal principal de cada veículo dominante. Pequenos
objetos de fundo/oclusões não formam gabarito exaustivo de todos os veículos da
cena. A contagem eIoU divulgadas referem-se aos painéis de referência explícitos;
uma auditoria de fundo mais ampla e segunda anotação independente continuam
necessárias antes de usar essa base como benchmark operacional. Não se alterou
o conjunto após observar o teste.
