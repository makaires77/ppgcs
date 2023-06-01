package neo4j

import (
	"context"
	"fmt"
	"log"

	"github.com/makaires77/ppgcs/pkg/domain/scrap_lattes"
	"github.com/neo4j/neo4j-go-driver/v5/neo4j"
)

type Neo4jWriteLattes struct {
	driver neo4j.DriverWithContext
}

// NewNeo4jWriter cria uma nova instância de Neo4jWriter.
func NewNeo4jWriteLattes(uri, username, password string) (*Neo4jWriteLattes, error) {
	driver, err := neo4j.NewDriverWithContext(uri, neo4j.BasicAuth(username, password, ""))
	if err != nil {
		return nil, err
	}

	return &Neo4jWriteLattes{
		driver: driver,
	}, nil
}

func (w *Neo4jWriteLattes) WritePesquisador(pesquisador *scrap_lattes.Pesquisador) error {
	// Aqui se implementa a lógica para escrever os dados do pesquisador no Neo4j
	ctx := context.Background()

	session := w.driver.NewSession(ctx, neo4j.SessionConfig{AccessMode: neo4j.AccessModeWrite})
	defer session.Close(ctx)

	cypherQuery := "CREATE (p:Pesquisador {id: $id, nome: $nome})"
	parameters := map[string]interface{}{
		"id":   pesquisador.IDLattes,
		"nome": pesquisador.Nome,
	}

	result, err := session.Run(ctx, cypherQuery, parameters)
	if err != nil {
		log.Printf("Erro ao escrever os dados do pesquisador no Neo4j: %s\n", err)
		return err
	}

	log.Println("Dados do pesquisador escritos com sucesso no Neo4j!")
	fmt.Println(result)

	return nil
}

func (n *Neo4jWriteLattes) WritePublicacao(publicacao *scrap_lattes.Publicacao) error {
	// Aqui você implementa a lógica para escrever os dados da publicação no Neo4j

	// Exemplo de escrita no Neo4j
	// Caso ocorra algum erro durante a escrita, você pode retornar um erro
	if publicacao == nil {
		return fmt.Errorf("publicação inválida")
	}

	// Simulação de escrita bem-sucedida
	fmt.Printf("Dados da publicação %s escritos no Neo4j\n", publicacao.Titulo)

	return nil
}

func (n *Neo4jWriteLattes) WritePesquisadorPublicacao(pesquisador *scrap_lattes.Pesquisador, publicacao *scrap_lattes.Publicacao) error {
	// Aqui você implementa a lógica para escrever a relação entre o pesquisador e a publicação no Neo4j

	// Exemplo de escrita no Neo4j
	// Caso ocorra algum erro durante a escrita, você pode retornar um erro
	if pesquisador == nil || publicacao == nil {
		return fmt.Errorf("pesquisador ou publicação inválidos")
	}

	// Simulação de escrita bem-sucedida
	fmt.Printf("Relação entre pesquisador %s e publicação %s escrita no Neo4j\n", pesquisador.Nome, publicacao.Titulo)

	return nil
}

func (n *Neo4jWriteLattes) WriteProjetoPesquisa(projeto *scrap_lattes.ProjetoPesquisa) error {
	// Aqui você implementa a lógica para escrever os dados do projeto de pesquisa no Neo4j

	// Exemplo de escrita no Neo4j
	// Caso ocorra algum erro durante a escrita, você pode retornar um erro
	if projeto == nil {
		return fmt.Errorf("projeto de pesquisa inválido")
	}

	// Simulação de escrita bem-sucedida
	fmt.Printf("Dados do projeto de pesquisa %s escritos no Neo4j\n", projeto.Titulo)

	return nil
}

func (n *Neo4jWriteLattes) WritePesquisadorProjetoPesquisa(pesquisador *scrap_lattes.Pesquisador, projeto *scrap_lattes.ProjetoPesquisa) error {
	// Aqui você implementa a lógica para escrever a relação entre o pesquisador e o projeto de pesquisa no Neo4j

	// Exemplo de escrita no Neo4j
	// Caso ocorra algum erro durante a escrita, você pode retornar um erro
	if pesquisador == nil || projeto == nil {
		return fmt.Errorf("pesquisador ou projeto de pesquisa inválidos")
	}

	// Simulação de escrita bem-sucedida
	fmt.Printf("Relação entre pesquisador %s e projeto de pesquisa %s escrita no Neo4j\n", pesquisador.Nome, projeto.Titulo)

	return nil
}

func (n *Neo4jWriteLattes) WritePatente(patente *scrap_lattes.Patente) error {
	// Aqui você implementa a lógica para escrever os dados da patente no Neo4j

	// Exemplo de escrita no Neo4j
	// Caso ocorra algum erro durante a escrita, você pode retornar um erro
	if patente == nil {
		return fmt.Errorf("patente inválida")
	}

	// Simulação de escrita bem-sucedida
	fmt.Printf("Dados da patente %s escritos no Neo4j\n", patente.Titulo)

	return nil
}

func (n *Neo4jWriteLattes) WritePesquisadorPatente(pesquisador *scrap_lattes.Pesquisador, patente *scrap_lattes.Patente) error {
	// Aqui você implementa a lógica para escrever a relação entre o pesquisador e a patente no Neo4j

	// Exemplo de escrita no Neo4j
	// Caso ocorra algum erro durante a escrita, você pode retornar um erro
	if pesquisador == nil || patente == nil {
		return fmt.Errorf("pesquisador ou patente inválidos")
	}

	// Simulação de escrita bem-sucedida
	fmt.Printf("Relação entre pesquisador %s e patente %s escrita no Neo4j\n", pesquisador.Nome, patente.Titulo)

	return nil
}

func (n *Neo4jWriteLattes) WritePremioTitulo(premio *scrap_lattes.PremioTitulo) error {
	// Aqui você implementa a lógica para escrever os dados do prêmio/título no Neo4j

	// Exemplo de escrita no Neo4j
	// Caso ocorra algum erro durante a escrita, você pode retornar um erro
	if premio == nil {
		return fmt.Errorf("prêmio/título inválido")
	}

	// Simulação de escrita bem-sucedida
	fmt.Printf("Dados do prêmio/título %s escritos no Neo4j\n", premio.Descricao)

	return nil
}

func (n *Neo4jWriteLattes) WritePesquisadorPremioTitulo(pesquisador *scrap_lattes.Pesquisador, premio *scrap_lattes.PremioTitulo) error {
	// Aqui você implementa a lógica para escrever a relação entre o pesquisador e o prêmio/título no Neo4j

	// Exemplo de escrita no Neo4j
	// Caso ocorra algum erro durante a escrita, você pode retornar um erro
	if pesquisador == nil || premio == nil {
		return fmt.Errorf("pesquisador ou prêmio/título inválidos")
	}

	// Simulação de escrita bem-sucedida
	fmt.Printf("Relação entre pesquisador %s e prêmio/título %s escrita no Neo4j\n", pesquisador.Nome, premio.Descricao)

	return nil
}

// Close fecha a conexão com o Neo4j.
func (w *Neo4jWriteLattes) Close() {
	err := w.driver.Close(context.Background())
	if err != nil {
		log.Printf("Erro ao fechar a conexão com o Neo4j: %s\n", err)
	}
}

// Exemplo de uso
/*
func main() {
	uri := "bolt://localhost:7687"
	username := "neo4j"
	password := "password"

	writer, err := NewNeo4jWriter(uri, username, password)
	if err != nil {
		log.Fatal(err)
	}
	defer writer.Close()

	pesquisador := &scrap_lattes.Pesquisador{
		// Preencher os dados do pesquisador
	}

	err = writer.WritePesquisador(pesquisador)
	if err != nil {
		log.Fatal(err)
	}

	log.Println("Dados do pesquisador escritos com sucesso!")

	// Aguardar alguns segundos para exibir os logs antes de encerrar o programa
	time.Sleep(3 * time.Second)
} */
