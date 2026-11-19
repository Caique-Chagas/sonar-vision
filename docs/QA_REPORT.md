# Qualidade e validação da publicação

Baseline do desenvolvimento:472testes emPython3.11/3.12,com execução nativa dos
três vídeos e do áudio no macOS. A seleção acadêmica preserva464testes de aplicação;
7testes de diário e1de inventário local foram mantidos somente no desenvolvimento.
Cinco regressões novas verificam download verificado e recontagem das evidências:
**469testes públicos**, sem pesos ou vídeos privados na CI.

Ambiente limpoPython3.12 criado para a publicação,instalação normal de`.[dev,video]`.
Pytest469aprovados;Ruff/formato,mypy76fontes e pipcheck exigidos. Python3.11também
é verificado na matriz da CI e localmente. Os resultados efetivos desta publicação
são registrados em [release-validation.json](../reports/release-validation.json).
Uma advertênciaStarlette/httpx não impediu os testes; não se alega erro zero de dependências
além das verificações executadas.

Testes usam mocks e imagens/MP4s de software gerados localmente. Verificam contratos,
exceções,SSR F,limites,ROI,OCRbruto,normalização,conflitos,IoU,consenso,framesduplicados,
áudio,mute/restart,fallback e preparação/renderização. Não medem acurácia científica.
A versão desktop/macOS foi exercitada no desenvolvimento; a CIUbuntu não tenta
abrir janela nativa ou sintetizar vozLuciana.

Pesos portáveis:carregamento,tensores/dtypes/shapes idênticos e predições equivalentes
em duas imagens permitidas,semnovo treinamento. Registros em
[model-portability-validation.json](../reports/model-portability-validation.json).

A auditoria pública examina somente os arquivos selecionados no índiceGit e o
commit a enviar,sem varrer novamente o inventário privado de67mil arquivos.
Sem.env real,mídia pessoal,diário/backups,logs internos ou checkpoints experimentais.
`scripts/verify_publication.py` confere caminhos,links relativos,tamanho e padrões
conhecidos de credenciais. Isso não é garantia absoluta contra qualquer segredo.
Nenhum forcepush ou envio do histórico interno de desenvolvimento.
