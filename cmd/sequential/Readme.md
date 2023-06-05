Este algoritmo usa apenas execução sequencial. Não há paralelismo nem multiprocessamento envolvido. 
Cada iteração do loop principal processa uma combinação de autor e discente de forma sequencial, aguardando a conclusão antes de passar para a próxima iteração. 
O processamento ocorre em uma única thread. No entanto, o código foi estruturado de forma modular, permitindo que seja introduzido paralelismo no futuro, se necessário.
Por exemplo, executando as comparações de nomes em goroutines separadas ou utilizando bibliotecas específicas para processamento paralelo.

Até a expansão para microsserviços é facilitada por um código bem entruturado em lógica modular, desacoplada.
Com microsserviços há uma complexidade adicional para buscar as vantagens de escala horizontal. Uma infraestrutura adequada para orquestração, balanceamento de carga e monitoramento dos serviços em uma arquitetura de microsserviços pode ser implementada utilizando ferramentas de orquestração de contêineres, como Kubernetes ou Docker Swarm. Esta é uma visão geral de como seria essa infraestrutura:

Contêineres: Os microsserviços seriam empacotados em contêineres, usando tecnologias como Docker. Os contêineres fornecem um ambiente isolado e portátil para executar os serviços, garantindo a consistência e a facilidade de implantação.

Orquestração de Contêineres: O Kubernetes ou o Docker Swarm são exemplos de ferramentas de orquestração de contêineres que ajudam a gerenciar e implantar os contêineres dos microsserviços. Essas ferramentas fornecem recursos para lidar com o escalonamento, a descoberta de serviços, a resiliência e o balanceamento de carga entre os contêineres.

Kubernetes: O Kubernetes é uma plataforma de orquestração de contêineres de código aberto amplamente utilizada. Ele permite implantar, gerenciar e escalar automaticamente os contêineres dos microsserviços, além de oferecer recursos avançados, como o balanceamento de carga, recuperação de falhas, atualizações zero-downtime e autoscaling.

Docker Swarm: O Docker Swarm é uma solução de orquestração de contêineres fornecida pela Docker. Ele permite criar um cluster de hosts Docker e gerenciar a implantação dos contêineres em escala, fornecendo recursos como balanceamento de carga, autoescalabilidade e recuperação de falhas.

Balanceamento de Carga: O balanceamento de carga é essencial para distribuir o tráfego de entrada entre os microsserviços em execução. As ferramentas de orquestração de contêineres, como Kubernetes e Docker Swarm, fornecem funcionalidades embutidas para o balanceamento de carga. Por meio dessas ferramentas, você pode configurar serviços de balanceamento de carga que distribuem o tráfego entre os contêineres dos microsserviços de forma eficiente e escalável.

Monitoramento e Logging: É fundamental monitorar o desempenho e o estado dos microsserviços em execução. Para isso, você pode utilizar ferramentas de monitoramento, como Prometheus, Grafana ou ELK Stack (Elasticsearch, Logstash, Kibana), para coletar métricas, logs e rastreamento de solicitações. Essas ferramentas ajudam a identificar problemas, analisar o desempenho e garantir a disponibilidade dos serviços.

Escalabilidade e Autoscaling: As ferramentas de orquestração de contêineres permitem escalonar automaticamente os microsserviços com base na demanda. Com o Kubernetes ou o Docker Swarm, você pode definir regras de escalabilidade para aumentar ou diminuir a quantidade de réplicas dos contêineres dos microsserviços conforme a carga de trabalho. Além disso, o autoscaling permite ajustar a escala automaticamente com base em métricas de desempenho, como o uso de CPU ou o número de solicitações.