package rabbitmq

import (
	"github.com/makaires77/ppgcs/pkg/infrastructure/scrap_lattes"
	"github.com/streadway/amqp"
)

func ConsumeLattesFromRabbitMQ(connectionString string, queueName string, scrapLattes *scrap_lattes.ScrapLattes) error {
	conn, err := amqp.Dial(connectionString)
	if err != nil {
		return err
	}
	defer conn.Close()

	consumer := NewConsumer(conn, queueName, scrapLattes)

	err = consumer.Start()
	if err != nil {
		return err
	}

	return nil
}
