package scrap_lattes

import (
	"log"
	"time"
)

// ScrapLattesUseCase representa o caso de uso do scraping de dados do Lattes.
type ScrapLattesUseCase struct {
	repository Repository
}

// NewScrapLattesUseCase cria uma nova instância de ScrapLattesUseCase.
func NewScrapLattesUseCase(repository Repository) *ScrapLattesUseCase {
	return &ScrapLattesUseCase{
		repository: repository,
	}
}

// Execute executa o caso de uso do scraping de dados do Lattes.
func (uc *ScrapLattesUseCase) Execute(url string) error {
	pesquisador, err := ScrapPesquisador(url)
	if err != nil {
		log.Printf("Erro ao realizar o scraping do pesquisador: %s\n", err)
		return err
	}

	err = uc.repository.SavePesquisador(pesquisador)
	if err != nil {
		log.Printf("Erro ao salvar os dados do pesquisador: %s\n", err)
		return err
	}

	log.Println("Dados do pesquisador salvos com sucesso!")

	return nil
}

// Repository define a interface do repositório para salvar os dados do pesquisador.
type Repository interface {
	SavePesquisador(pesquisador *Pesquisador) error
}

// Exemplo de implementação do repositório

// MongoRepository é uma implementação do repositório usando o MongoDB.
type MongoRepository struct {
	// Configuração do cliente MongoDB e coleção
}

// NewMongoRepository cria uma nova instância de MongoRepository.
func NewMongoRepository() *MongoRepository {
	// Inicialização do cliente MongoDB e coleção
	return &MongoRepository{}
}

// SavePesquisador salva os dados do pesquisador no MongoDB.
func (r *MongoRepository) SavePesquisador(pesquisador *Pesquisador) error {
	// Lógica para salvar os dados no MongoDB
	log.Printf("Salvando dados do pesquisador %s no MongoDB\n", pesquisador.Nome)
	return nil
}

// Exemplo de uso

func main() {
	repository := NewMongoRepository()
	usecase := NewScrapLattesUseCase(repository)

	url := "http://exemplo.com/pesquisador"

	err := usecase.Execute(url)
	if err != nil {
		log.Fatal(err)
	}

	// Aguardar alguns segundos para exibir os logs antes de encerrar o programa
	time.Sleep(3 * time.Second)
}
