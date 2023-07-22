// pkg\domain\scrap_lattes\usecases.go
package scrap_lattes

import (
	"context"
	"log"
	"time"

	"go.mongodb.org/mongo-driver/mongo"
	"go.mongodb.org/mongo-driver/mongo/options"

	"github.com/makaires77/ppgcs/pkg/infrastructure/scrap_lattes"
)

// Scraper define a interface para o objeto de scraping.
type Scraper interface {
	ScrapPesquisador(url string) (*Pesquisador, error)
}

// Repository define a interface do repositório para salvar os dados do pesquisador.
type Repository interface {
	SavePesquisador(ctx context.Context, pesquisador *Pesquisador) error
}

// ScrapLattesUseCase representa o caso de uso do scraping de dados do Lattes.
type ScrapLattesUseCase struct {
	scraper    Scraper
	repository Repository
}

// NewScrapLattesUseCase cria uma nova instância de ScrapLattesUseCase.
func NewScrapLattesUseCase(scraper Scraper, repository Repository) *ScrapLattesUseCase {
	return &ScrapLattesUseCase{
		scraper:    scraper,
		repository: repository,
	}
}

// Execute executa o caso de uso do scraping de dados do Lattes.
func (uc *ScrapLattesUseCase) Execute(ctx context.Context, url string) error {
	pesquisador, err := uc.scraper.ScrapPesquisador(url)
	if err != nil {
		log.Printf("Erro ao realizar o scraping do pesquisador: %s\n", err)
		return err
	}

	err = uc.repository.SavePesquisador(ctx, pesquisador)
	if err != nil {
		log.Printf("Erro ao salvar os dados do pesquisador: %s\n", err)
		return err
	}

	log.Println("Dados do pesquisador salvos com sucesso!")

	return nil
}

// MongoRepository é uma implementação do repositório usando o MongoDB.
type MongoRepository struct {
	client     *mongo.Client
	collection *mongo.Collection
}

// NewMongoRepository cria uma nova instância de MongoRepository.
func NewMongoRepository() (*MongoRepository, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	client, err := mongo.Connect(ctx, options.Client().ApplyURI("mongodb://localhost:27017"))
	if err != nil {
		return nil, err
	}

	collection := client.Database("lattes_db").Collection("pesquisadores")

	return &MongoRepository{
		client:     client,
		collection: collection,
	}, nil
}

// SavePesquisador salva os dados do pesquisador no MongoDB.
func (r *MongoRepository) SavePesquisador(ctx context.Context, pesquisador *Pesquisador) error {
	_, err := r.collection.InsertOne(ctx, pesquisador)
	if err != nil {
		return err
	}

	log.Printf("Salvando dados do pesquisador %s no MongoDB\n", pesquisador.Nome)

	return nil
}

// Fecha fecha a conexão com o MongoDB.
func (r *MongoRepository) Close(ctx context.Context) error {
	return r.client.Disconnect(ctx)
}

// Exemplo de uso

func main() {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	repository, err := NewMongoRepository()
	if err != nil {
		log.Fatal(err)
	}
	defer repository.Close(ctx)

	scraper := scrap_lattes.NewScrapLattes() // Implemente isso para criar uma nova instância do objeto de scraping

	usecase := NewScrapLattesUseCase(scraper, repository)

	url := "http://exemplo.com/pesquisador"

	err = usecase.Execute(ctx, url)
	if err != nil {
		log.Fatal(err)
	}
}
