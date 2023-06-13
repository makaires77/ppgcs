package rabbitmq

import (
	"log"

	"github.com/makaires77/ppgcs/pkg/usecase/load_lattes"
	"github.com/streadway/amqp"
)

// EnqueueLattes é responsável por enfileirar a carga de dados do Lattes.
type EnqueueLattes struct {
	lattesInteractor *load_lattes.Interactor
	conn             *amqp.Connection
	ch               *amqp.Channel
	queueName        string
}

// NewEnqueueLattes cria uma nova instância de EnqueueLattes.
func NewEnqueueLattes(lattesInteractor *load_lattes.Interactor, conn *amqp.Connection, queueName string) (*EnqueueLattes, error) {
	ch, err := conn.Channel()
	if err != nil {
		return nil, err
	}

	queue, err := ch.QueueDeclare(
		queueName,
		false,
		false,
		false,
		false,
		nil,
	)
	if err != nil {
		return nil, err
	}

	return &EnqueueLattes{
		lattesInteractor: lattesInteractor,
		conn:             conn,
		ch:               ch,
		queueName:        queue.Name,
	}, nil
}

// EnqueuePesquisador enfileira o carregamento de dados de um pesquisador do Lattes.
func (e *EnqueueLattes) EnqueuePesquisador(pesquisadorID string) error {
	err := e.ch.Publish(
		"",
		e.queueName,
		false,
		false,
		amqp.Publishing{
			ContentType: "text/plain",
			Body:        []byte(pesquisadorID),
		},
	)
	if err != nil {
		log.Printf("Erro ao enfileirar o carregamento de dados do pesquisador %s: %s\n", pesquisadorID, err)
		return err
	}

	log.Printf("Carregamento de dados do pesquisador %s enfileirado com sucesso!\n", pesquisadorID)

	return nil
}

// ConsumePesquisadores consome a fila de pesquisadores a serem carregados.
func (e *EnqueueLattes) ConsumePesquisadores() {
	msgs, err := e.ch.Consume(
		e.queueName,
		"",
		true,
		false,
		false,
		false,
		nil,
	)
	if err != nil {
		log.Fatalf("Erro ao consumir a fila de pesquisadores: %s\n", err)
	}

	forever := make(chan bool)

	go func() {
		for d := range msgs {
			pesquisadorID := string(d.Body)
			log.Printf("Consumindo carregamento de dados do pesquisador %s\n", pesquisadorID)

			err := e.lattesInteractor.LoadPesquisador(pesquisadorID)
			if err != nil {
				log.Printf("Erro ao carregar dados do pesquisador %s: %s\n", pesquisadorID, err)
			}
		}
	}()

	log.Println("Aguardando a carga de dados dos pesquisadores...")

	<-forever
}
