## Serviço de webscraping com Arquitetura Hexagonal

A arquitetura Ports and Adapters (ou Arquitetura Hexagonal) é um padrão de design que busca criar uma separação clara entre a lógica de negócio do aplicativo e os detalhes técnicos ou de infraestrutura, como acesso a banco de dados, sistema de arquivos, rede e interfaces de usuário. Isso é feito para tornar o código mais modular, testável e fácil de entender e manter.

A lógica de negócio é o scraping dos dados da Plataforma Lattes e o processamento desses dados. 

Os detalhes técnicos ou de infraestrutura são a leitura do arquivo CSV e a gravação no banco de dados Neo4j.

Distribuímos a implementação do código em diferentes arquivos e pacotes de acordo com essas responsabilidades, de forma a seguir as melhores práticas de Ports and Adapters. 

## Papel dos principais arquivos:

**scrap_lattes.go** Contém o core da lógica de negócio. Esse código armazenado na pasta pkg/domain/scrap_laes deve ser totalmente independente de qualquer infraestrutura, lendo dados de um "port" de entrada e escrevendo em um "port" de saída.

**csv_handler.go** É o responsável por ler o arquivo CSV e transformá-lo em uma estrutura de dados que a lógica de negócio possa usar.

**neo4jclient.go** É um "Adapter" que implementa o "port" de saída definido na lógica de negócio, permitindo que ela escreva no banco de dados Neo4j.

**main.go** É responsável por construir todas essas partes e conectá-las juntas.

Além disso, temos testes unitários para cada uma dessas partes, garantindo que elas funcionam corretamente de forma isolada. Esta é apenas uma organização inicial do código. 
A estrutura exata pode variar dependendo das necessidades específicas do seu projeto e das convenções da equipe de desenvolvimento.

## Orientações para refatorar em Ports and Adapters:
Baseado no código e informações já existentes, para ajustar de acordo com o padrão Ports and Adapters (Hexagonal Architecture) pode-se:

### 1. Defina Interfaces para as Portas

Defina interfaces para descrever as operações que sua aplicação deve realizar. No caso de uso `ScrapLattesUseCase`, a interface do repositório é uma port, então você já começou bem. No entanto, você também deve criar uma interface para o scraper que extrai as informações do currículo Lattes.

```go
// Scraper define uma interface para operações de scraping.
type Scraper interface {
	ScrapPesquisador(url string) (*Pesquisador, error)
}
```

### 2. Use Implementações Específicas como Adaptadores

Seu `MongoRepository` é um exemplo de um adaptador. Você também pode criar um `LattesScraper` que implementa a interface `Scraper`:

```go
type LattesScraper struct {
	// ...
}

func (s *LattesScraper) ScrapPesquisador(url string) (*Pesquisador, error) {
	// Lógica de scraping.
}
```

### 3. Injeção de Dependências

Seu caso de uso deve depender das interfaces (portas), não das implementações concretas (adaptadores). As implementações concretas devem ser injetadas no caso de uso:

```go
type ScrapLattesUseCase struct {
	repository Repository
	scraper    Scraper
}

func NewScrapLattesUseCase(repository Repository, scraper Scraper) *ScrapLattesUseCase {
	return &ScrapLattesUseCase{
		repository: repository,
		scraper:    scraper,
	}
}
```

### 4. Use o Princípio da Inversão de Dependências

As classes de alto nível (casos de uso) não devem depender de classes de baixo nível (repositórios, serviços de scraping, etc.). Em vez disso, ambas devem depender de abstrações (as interfaces).

```go
func (uc *ScrapLattesUseCase) Execute(url string) error {
	pesquisador, err := uc.scraper.ScrapPesquisador(url)
	// ...
	err = uc.repository.SavePesquisador(pesquisador)
	// ...
}
```

### 5. Mantenha a Lógica de Negócio e Infraestrutura Separadas

As interfaces e classes de caso de uso (lógica de negócio) devem estar em um pacote separado das implementações de infraestrutura (como `MongoRepository` e `LattesScraper`).

Dessa forma, a estrutura geral de diretórios poderia ficar assim:

```
pkg/
  domain/
    scrap_lattes/
      usecase.go
      repository.go
      scraper.go
  infrastructure/
    mongo_repository.go
    lattes_scraper.go
```

Nesta estrutura, `pkg/domain/scrap_lattes/` contém as definições de suas interfaces (portas) e casos de uso, e `pkg/infrastructure/` contém as implementações concretas dessas interfaces (adaptadores).

### 6. Testes

Com essa estrutura, você pode criar implementações falsas de suas interfaces para testes. Por exemplo, você pode criar um `MockScraper` que implementa `Scraper` para testar `ScrapLattesUseCase` sem fazer chamadas de rede.

Dessa forma, você pode criar testes independentes e facilmente substituir implementações concretas (como mudar de MongoDB para SQL, por exemplo) sem alterar a lógica de negócios.
