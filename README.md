# Processos de avaliação da pós-graduação
## PPGCS - Avaliar Publicação Docente/Discente na Pós-graduação

Este trabalho suporta a tomada de decisão da Comissão Gestora do Programa, que a partir da medição contínua das publicações do corpo de pesquisadores, formado pelos docentes (permanentes e colaboradores) e pelo corpo discente, realiza o acompanhamento contínuo das produções, em quantidade e qualidade, orientada pelo impacto das publicações científico-acadêmicas do PPGCS.

Com intuito de entender e aprimorar todo processo de avaliação replica-se com antecedência a geração das mesmas informações que serão usadas pela CAPES para a avaliação periódoca do perfil do corpo docente do Programa de Pós-graduação em Ciências da Saúde – Instituto René Rachou – Fiocruz Minas. 

O processo estuturado a partir deste trabalho permite entedimento e avaliação contínua e ininterrupta do programa, focando em dois momentos no tempo: a Avaliação Bienal (Parcial) e a Avaliação Quadrienal conforme os ditames da avaliação da CAPES para a pós-graduação.

**Avaliação Quadrienal:** Visa replicar e detalhar a última avaliação completa do Programa realizada pela CAPES.

**Avaliação Bienal:** Visa promover a reavaliação de meio termo do corpo docente do programa, para acompanhar o atendimento, manuter e melhorar o desempenho medido pelos indicadores.

Os indicadores que balizam a avaliação são construídos com base nos parâmetros exigidos pela CAPES, em específico para área de avaliação Medicina II, o que permite readequar rumos sempre que necessário para aprimorar o desempenho do programa. 

São considerados os docentes permanentes (DP) e docentes colaboradores (DC), com base nos mesmos parâmetros esperados de impacto (medido por pontuação ponderada relativa ao estrato Qualis Periódicos da área Medicina II das revistas utilizadas nas publicações do período).


Estrutura de pastas, em arquitetura hexagonal e orientação a funções:

*app* contém a lçógica do aplicativo:
  
  *routes.py* especifica as roas HTTP
  
  *controllers* coném as funções que recebem as requisições HTTP  e encaminham para os casos de uso correspondentes
  
  *usecases* coném os casos de uso que definem a lógica do negócio para cada entidade

*domain*  contém as definições de entidades, exceções e outros objetos puros

*infraestructure* tem a lógica de acesso a dados

  *repositories* encapsulam as operações com banco de dados
  
  *adapters* se comunicam com outras APIs
  
O arquivo config.py contém configurações do serviço

O arquivo run.py inicia o serviço Flask

