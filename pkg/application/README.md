No serviço implementado pelo researcher_repository.go, temos as implementações específicas de ResearcherRepository para MongoDB, Dgraph e Neo4j.

Neste arquivo implementamos também a lógica do domínio, como encontrar um pesquisador, salvar um pesquisador, etc. Essa camada se comunica com a camada de infraestrutura através da interface ResearcherRepository.

Finalmente, em main.go, você instanciará suas implementações de ResearcherRepository, criar o serviço ResearcherService e iniciar a lógica de aplicação.