# Mapa da publicação acadêmica

| Caminho | Finalidade |
|---|---|
| src/sonar_vision/ | Runtime, contratos, API, visão, OCR, validação, desktop e voz |
| sonar_br/ | Código experimental/treino/avaliação necessário aos métodos e testes; sem datasets/checkpoints |
| native/ | HelperCocoa/AVFoundation reutilizado pelo player |
| scripts/ | Execução, download verificado, preparação e ferramentas de reprodução |
| tests/ | Testes de aplicação e fixtures pequenas; sem requisitos de mídia privada |
| configs/,notebooks/ | Configurações e alternativaColab de treinamento |
| docs/ | Arquitetura, dados, treino, modelos, resultados, reprodução e créditos |
| reports/ | PDFoficial, CSVs/curvas, métricas e evidências numéricas selecionadas |
| .github/workflows/ | QA emPython3.11/3.12 |

O histórico público existente é preservado. Esta atualização parte da branch
pública; commits internos de desenvolvimento não são enviados. Arquivos
retirados da árvore atual continuam preservados localmente e em commits públicos
anteriores: vídeo narrado/simulações do piloto, logs e relatórios temporários,
utilitários obsoletos de empacotamento do vídeo. Não foi usado forcepush.

Fora da publicação:AGENTS,planos/estados/diário do agente, ambientes, credenciais,
backups, snapshots, caches de inferência, dados privados e MP4s. O diário original
continua localmente. Nenhuma exclusão local importante é necessária para organizar
a árvore pública.

Os464testes de aplicação derivados do baseline472 foram preservados;7testes do
helper de diário e1teste do inventário local não integram esta entrega acadêmica.
Novos testes de download e contagem verificam a publicação. Nenhum teste de
aplicação foi desativado para forçar aprovação.
