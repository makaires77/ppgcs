## Iniciando desenvolvimento com Monolitos
Sempre iniciaremos nosso desenvolvimento adotando uma arquitetura monolítica, porém totalmente modularizada, mas onde toda a aplicação é construída como uma unidade única. Nessa abordagem, os módulos ou componentes da aplicação interagem diretamente uns com os outros, tipicamente através de chamadas de função ou método. Para gerenciar os eventos de requisição, apresentação de resultados e persistência de dados na aplicação usaremos:

Estrutura de Módulos: Organizamos o código em módulos ou pacotes separados por funcionalidade. Por exemplo, temos um módulo para lidar com as requisições, outro para apresentar os resultados, e outro para interagir com o banco de dados.

ORMs e Bancos de Dados: Para a persistência de dados, usaremos de sistemas de gerenciamento de banco de dados (como o Neo4j, Dgraph, PostgreSQL, MySQL ou SQLite de acordo com a necessidade de cada módulo) e um ORM (Object-Relational Mapping) como Sequelize (para Node.js) ou SQLAlchemy (para Python) para facilitar a interação com o banco de dados.

Gerenciamento de Estado: Quando a complexidade da interação entre os módulos exigir, usaremos uma biblioteca de gerenciamento de estado como Redux (para JavaScript) ou MobX para facilitar o rastreamento de mudanças no estado da sua aplicação e ajudar a manter código mais organizado e previsível.

Testes: Temos uma boa cobertura de testes para os diferentes módulos da aplicação. Testes unitários (para funções individuais), testes de integração (para verificar como diferentes partes da sua aplicação interagem) e testes de ponta a ponta (para testar a aplicação como um todo).

Controle de Versão: O sistema de controle de versão será o Git. Para rastrar mudanças no código ao longo do tempo, colaborar com os demais membros da equipe como desenvolvedores, e reverter mudanças se algo der errado.

Integração Contínua/Entrega Contínua (CI/CD): Utilizamos ferramentas de CI/CD como Jenkins, Travis CI, ou GitHub Actions para automatizar o processo de testes e implantação da aplicação. Para garantir que o código que em desenvolvendo está funcionando como esperado e pode ser entregue aos usuários de maneira eficiente.

Logging e Monitoramento: Usamos ferramentas de logging como Winston ou Bunyan (para Node.js) para rastrear eventos na aplicação. E Prometheus ou New Relic para detectar problemas de desempenho ou falhas.

Cada módulo tem suas próprias necessidades, por isso as ferramentas e abordagens específicas escolhidas podem variar dependendo dos casos de uso, estrutura atual de pastas e arquivos, e outras considerações.

## Opção por microsserviços conforme oportunidade e necessidades
A interação de microserviços numa aplicação pode ser complexa, pois envolver maiores custos em tdesempenho e despsas com outros serviços em nuvem, porém para os casos onde forem considerados vantajosos para aplicação completa, já implementamos módulos no formato de microsserviços em nuvem usando as ferramentas:

RabbitMQ/Message Broker: o RabbitMQ ou outro broker de mensagens (como o Kafka) é útil para a comunicação entre microserviços, especialmente para tarefas assíncronas e intensivas em dados.

Docker/Kubernetes: Para implantar, escalar e gerenciar microserviços, usamos Docker e Kubernetes para empacotar a aplicação em contêineres, tornando-a portátil, fácil de implantar e escalar.

Bancos de Dados: Dependendo das necessidades de cada módulo, usaremos um banco de dados específico: SQL (como o PostgreSQL) ou NoSQL (como o MongoDB) para persistência de dados. Cada microserviço pode ter seu próprio banco de dados, abordagem conhecida como banco de dados por serviço.

Tracing Distribuído: Para monitorar e solucionar problemas em microserviços, uma ferramenta de tracing distribuído (como o Jaeger ou o Zipkin) é útil. Ela permite que você veja como as solicitações percorrem seus serviços e onde os gargalos ou erros estão ocorrendo.

Gateway de API: Usaremos o gateway de API Kong (ou o Apigee) pode fornecer uma interface unificada para seus microserviços, facilitando a gestão da autenticação, do rate limiting, das rotas de serviço e de outras preocupações comuns para realizar todo o gerenciamento de APIs: permitir que os serviços comuniquem-se entre si de maneira eficiente e segura. Nesta aplicação utilizamos o Kong, que é uma plataforma de gerenciamento de APIs e microserviços de código aberto.
