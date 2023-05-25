package rabbitmq

import (
	"log"

	"github.com/makaires77/ppgcs/pkg/infrastructure/scrap_lattes"
	"github.com/streadway/amqp"
)

type Consumer struct {
	connection  *amqp.Connection
	queueName   string
	scrapLattes *scrap_lattes.ScrapLattes
}

func NewConsumer(connection *amqp.Connection, queueName string, scrapLattes *scrap_lattes.ScrapLattes) *Consumer {
	return &Consumer{
		connection:  connection,
		queueName:   queueName,
		scrapLattes: scrapLattes,
	}
}

func (c *Consumer) Start() error {
	channel, err := c.connection.Channel()
	if err != nil {
		return err
	}

	msgs, err := channel.Consume(
		c.queueName,
		"",
		true,
		false,
		false,
		false,
		nil,
	)
	if err != nil {
		return err
	}

	forever := make(chan bool)

	go func() {
		for d := range msgs {
			log.Printf("Received a message: %s", d.Body)
			c.scrapLattes.ProcessarArquivo(string(d.Body))
		}
	}()

	log.Printf(" [*] Waiting for messages. To exit press CTRL+C")
	<-forever

	return nil
}
