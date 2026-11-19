# Protocolo de validação

Avaliar separadamente localização do ônibus/painel,legibilidade/OCR,normalização,
associação por veículo e evidência temporal. Código de referência é inspeção visual
independente do OCR; não deve ser fornecido ao reconhecedor como resposta esperada.
Treino,validação e teste devem separar capturas/veículos quando identificáveis.

Os vídeos selecionados não são um novo teste independente. Dados numéricos por
frame permitem recalcular contagens e verificar ausência de anúncios nos exemplos
parciais; não permitem revisar as imagens privadas. [Resultados](RESULTS.md).

CODE_STABLErequer observações distintas e não significaLINE_VERIFIED. Nenhum
protocolo desta demonstração consulta uma fonte operacionalSPTrans ou confirma
embarque. Confiança não é probabilidade calibrada. O contrato legado da APIpode
representar correspondência textual no protótipo; isso não comprova uma operação
real integrada aoSonarBordo.

A suíte de softwareusa fixtures; os resultados reais mantêm denominadores,
falhas,escopo e limitações. Não preencher detecção ausente com caixa manual nem
substituir textoOCR por gabarito. Dados futuros usados para ajuste passam a ser
desenvolvimento e não permanecem teste independente.
