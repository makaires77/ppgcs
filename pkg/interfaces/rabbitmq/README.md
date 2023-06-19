consumer.go: Este arquivo define a estrutura Consumer que representa um consumidor RabbitMQ. Ele possui um método Start() que inicia o consumo de mensagens da fila RabbitMQ e chama a função ProcessarArquivo do objeto scrapLattes para processar cada mensagem recebida.

consume_lattes.go: Este arquivo contém a função ConsumeLattesFromRabbitMQ que é responsável por estabelecer uma conexão com o RabbitMQ, criar um consumidor usando a função NewConsumer e iniciar o consumo de mensagens chamando o método Start() do consumidor.

enqueue_lattes.go: Este arquivo define a estrutura EnqueueLattes responsável por enfileirar a carga de dados do Lattes no RabbitMQ. Ele possui métodos para enfileirar pesquisadores específicos e consumir a fila de pesquisadores a serem carregados.