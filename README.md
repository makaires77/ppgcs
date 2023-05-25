# Processos de avaliação da pós-graduação
## PPGCS - Avaliar Publicação Docente/Discente na Pós-graduação

Este trabalho suporta a tomada de decisão da Comissão Gestora do Programa, que a partir da medição contínua das publicações do corpo de pesquisadores, formado pelos docentes (permanentes e colaboradores) e pelo corpo discente, realiza o acompanhamento contínuo das produções, em quantidade e qualidade, orientada pelo impacto das publicações científico-acadêmicas do PPGCS.

Com intuito de entender e aprimorar todo processo de avaliação replica-se com antecedência a geração das mesmas informações que serão usadas pela CAPES para a avaliação periódoca do perfil do corpo docente do Programa de Pós-graduação em Ciências da Saúde – Instituto René Rachou – Fiocruz Minas. 

O processo estuturado a partir deste trabalho permite entedimento e avaliação contínua e ininterrupta do programa, focando em dois momentos no tempo: a Avaliação Bienal (Parcial) e a Avaliação Quadrienal conforme os ditames da avaliação da CAPES para a pós-graduação.

**Avaliação Quadrienal:** Visa replicar e detalhar a última avaliação completa do Programa realizada pela CAPES.

**Avaliação Bienal:** Visa promover a reavaliação de meio termo do corpo docente do programa, para acompanhar o atendimento, manuter e melhorar o desempenho medido pelos indicadores.

Os indicadores que balizam a avaliação são construídos com base nos parâmetros exigidos pela CAPES, em específico para área de avaliação Medicina II, o que permite readequar rumos sempre que necessário para aprimorar o desempenho do programa. 

São considerados os docentes permanentes (DP) e docentes colaboradores (DC), com base nos mesmos parâmetros esperados de impacto (medido por pontuação ponderada relativa ao estrato Qualis Periódicos da área Medicina II das revistas utilizadas nas publicações do período).

# Exemplo considerando Aquitetura Hexagonl de Portes e Adapters (Não utilizada neste projeto específico):
Uma estrutura de pastas, em arquitetura hexagonal e orientação a funções, pode de forma genérica conter a seguinte estrutura:

*app* contém a lçógica do aplicativo:
  
  *routes.py* especifica as roas HTTP
  
  *controllers* coném as funções que recebem as requisições HTTP  e encaminham para os casos de uso correspondentes
  
  *usecases* coném os casos de uso que definem a lógica do negócio para cada entidade

*domain*  contém as definições de entidades, exceções e outros objetos puros

*infraestructure* tem a lógica de acesso a dados

  *repositories* encapsulam as operações com banco de dados
  
  *adapters* se comunicam com outras APIs
  
O arquivo config.py contém configurações do serviço

O arquivo run.py inicia o serviço Flask

# Exemplo de Estrutura considerando DDD/EDD:
Este projeto em particular segue uma estrutura com base em DDD (Domain-Driven Design) e EDD (Event-Driven Design) implementada em Go no backend e com Javascript para frontend. A estrutura de diretórios para a fase de desenvolvimento é descrita aqui:

1. **Camada de Domínio (Domain Layer)**: Esta é a camada onde a lógica de negócios principal é colocada e é normalmente o coração de um sistema DDD. No seu projeto, essa camada é refletida principalmente na pasta `pkg/domain`. Aqui, você tem `entities.go` que contém as entidades do domínio (ou seja, os objetos principais que o sistema manipula), `repository.go` que contém as interfaces para a persistência de dados, e `usecases.go` que contém a lógica de negócios principal.

2. **Camada de Aplicação (Application Layer)**: Esta camada coordena as operações de alto nível, envolvendo várias entidades do domínio. No seu projeto, essa camada é refletida na pasta `pkg/usecase` e `cmd`.

3. **Camada de Infraestrutura (Infrastructure Layer)**: Esta camada contém detalhes específicos de implementação, como a forma como os dados são persistidos, a forma como os serviços externos são acessados, etc. No seu projeto, essa camada é refletida principalmente na pasta `pkg/infrastructure`.

4. **Camada de Interface (Interface Layer)**: Esta camada é responsável pela interação com o mundo exterior. Pode ser uma interface do usuário, uma API, ou um consumidor de fila de mensagens. No seu projeto, essa camada é refletida principalmente na pasta `pkg/interfaces`.

5. **Camada de Apresentação (Presentation Layer)**: Esta camada envolve a renderização de dados para o usuário, lidando com a entrada do usuário, etc. No seu projeto, essa camada é refletida na pasta `web`.

6. **Camada de Eventos (Event Layer)**: Considerando que você está usando EDD para o gerenciamento de filas, essa camada cuidará do processamento de eventos em tempo real, onde os eventos são produzidos e consumidos. No seu projeto, a camada de eventos é refletida na pasta `pkg/interfaces/rabbitmq`.

Obs.: Dependendo dos detalhes específicos do seu projeto, o mapeamento exato pode variar, mas se variar muito dificultará a manutenabilidade e compreensão do código.

## Detalhamento de cada arquivo na estrutura de pastas
A estrutura do projeto de desenvolvimento está assim organizada para adicionar lógica de negócios com base em uma visão geral desenvolvida na fase de Arquitetura da Solução:

### Na camada de domínio:
Lógica de negócios nas camadas de domínio: No diretório pkg/domain, há os subdiretorios: publication e scrap_lattes. Ambos contêm entities.go, repository.go e usecases.go. Esses são locais para adicionar a lógica de negócios.

Em entities.go, definimos as entidades do domínio de negócios, como a publicação e detalhes específicos do "scrap_lattes". Você adicionaremos métodos a essas entidades que executam os cálculos ou manipulações de dados.

Em usecases.go, definimos os casos de uso do domínio. Cada caso de uso deve encapsular uma operação de negócios específica.

Validação de Dados: A validação de dados ocorre em vários lugares dependendo da necessidade de fluxo de dados do aplicativo. Em regra, dicionamos a lógica de validação no nível do controlador (antes de passar os dados para o serviço) ou diretamente nos métodos de serviço.

Manipulações de Dados: A manipulação de dados geralmente acontece nos serviços do aplicativo, que são chamados pelo controlador. Os serviços geralmente executam a lógica de negócios principal e interagem com o banco de dados ou outras camadas de persistência.

### Na camada de infraestrutura:
Lógica de negócios nas camadas de infraestrutura: O diretório pkg/infrastructure contém vários serviços de infraestrutura, como dgraph, json_publication, mongo, neo4j e scrap_lattes. Se alguma lógica de negócios for específica para a interação com essas tecnologias, ela pode ser colocada aqui.

Lógica de negócios nas camadas de interface: Em pkg/interfaces, você tem uma interface HTTP e um serviço RabbitMQ. A lógica de negócios relacionada à formatação de mensagens, validação de solicitações ou respostas, e manipulação de erros será colocada aqui.

# Definição para o projeto atual:
No DDD, o foco é sobre o domínio e a lógica de negócios. A ideia é ter uma arquitetura rica e expressiva que se alinha ao domínio do problema. No caso do gerenciamento de pesquisas de um Programa de Pesquisa baseado na produção acadêmica e científica, o domínio seria o "ScrapLattes".

Primeiro, vamos dividir o código em quatro camadas principais, de acordo com o nível de interação com o usuário:

1. **Interface do usuário ou Camada de Apresentação**: Esta camada se preocupa com a interação do usuário. Neste caso, a interação do usuário não está muito explícita.

2. **Aplicação**: Esta camada serve como um canal entre a camada de Interface do usuário e a camada de Domínio. Pode-se introduzir um serviço de aplicação aqui que orquestra as chamadas para a camada de domínio.

3. **Domínio**: Esta camada contém as informações sobre o domínio do problema, as regras de negócio e os objetos de negócio. Neste caso, "Pesquisador", "Publicacao" seriam objetos de domínio.

4. **Infraestrutura**: Esta camada fornece os recursos técnicos para as outras camadas. Aqui estão as operações de banco de dados e a raspagem de dados do Lattes.

Segundo, vamos considerar os eventos como parte integrante do sistema. No caso de EDD, o código seria organizado em torno de produção, detecção e reação a eventos do estado do domínio. Um exemplo seria o evento de um novo pesquisador sendo salvo no banco de dados. Isso poderia disparar um evento que aciona outras partes do código (ou até mesmo outros sistemas) que estão interessados nesse evento.

Aqui estão algumas sugestões para refatorar o código:

Separar a lógica de conexão dos bancos de dados Dgraph,Neo4j,MongoDB em um pacote de infraestrutura separado e utilizar a injeção de dependência para usar esses serviços.

Definir interfaces claras para os repositórios (ex: PesquisadorRepository) na camada de domínio e implementar essas interfaces na camada de infraestrutura.

Utilizar a injeção de dependência para fornecer a implementação concreta do repositório à camada de aplicação.

Introduzir um sistema de manipulação de eventos para lidar com os eventos produzidos pelo sistema. Por exemplo, quando um novo pesquisador é adicionado, um evento poderia ser emitido, o que poderia acionar outras partes do código.

Um sistema de fila lida com o processamento em segundo plano, a raspagem de dados do Lattes pode ser uma operação demorada e é feita em uma tarefa em segundo plano, usando a biblioteca "github.com/gocraft/work" dentre outras.

# Detalhes da execução em Go
## Importação de estruturas
Ao importar um pacote em Go, o caminho do import é geralmente relativo ao $GOPATH/src ou ao diretório raiz do módulo, caso você esteja usando Go modules (recomendado e comum para projetos novos a partir de 2021). Os caminhos absolutos geralmente não são usados em projetos Go, a menos que esteja se referindo a um pacote padrão ou externo. Em outras palavras, independente do diretório raiz do seu projeto deveriamos importar o pacote publication assim:
```go
import (
	"pkg/domain/publication"
)
```

Então no código que vai usar essa estrutura, podemos usar a estrutura `Publication` assim:
```go
var pub publication.Publication

pub.Titulo = "Exemplo de Título"
pub.Ano = "2023"
// etc.
```

Na prática, é mais comum usar uma estrutura de diretórios relativa à raiz do projeto, em vez de uma estrutura de diretórios absoluta.
Importações absolutas só vão funcionar se o seu código estiver sendo executado no mesmo ambiente que a estrutura de diretórios que você forneceu. 

Durante o desenvolvimento, pode-se acabar declarando duas vezes a mesma estrutura, por exemplo, considere que `Publicacao` está declarada uma vez dentro do pacote `publication` e outra vez dentro da estrutura `Pesquisador`. Para fazer referência à mesma estrutura, vamos removê-la da estrutura `Pesquisador` e usar a estrutura `Publication` já definida da seguinte forma:

```go
Publicacoes []publication.Publication `json:"publicacoes"`
```

## Receiver de Função:
Uma estrutura comum será utilizada para manipular as funções em Go. Dada pela forma `(s *ScrapLattes)`, por exemplo, em Go é o que chamamos de receiver, ou receptor em português. Esse é um conceito muito importante na linguagem Go que se assemelha, mas não é exatamente igual, ao conceito de "this" ou "self" em outras linguagens de programação orientadas a objetos como Java ou Python, respectivamente. Aqui, `(s *ScrapLattes)` significa que a função está sendo definida no contexto de uma instância de `ScrapLattes`. A função que está sendo definida pode, então, acessar os campos e métodos dessa instância usando a variável `s`.

Por exemplo, se você tivesse um campo `nome` em `ScrapLattes`, você poderia acessá-lo dentro da função como `s.nome`.

Em Go, os receptores podem ser de dois tipos: valor e ponteiro. Neste caso, `*ScrapLattes` é um receptor do tipo ponteiro. Isso significa que qualquer alteração feita à instância `s` dentro da função será refletida na instância original. Em contraste, se você usasse `ScrapLattes` (sem o asterisco), quaisquer alterações feitas em `s` não seriam refletidas na instância original, pois `s` seria apenas uma cópia.

Os receptores permitem que você defina comportamentos que são específicos a tipos específicos (como classes em linguagens orientadas a objetos), e são a principal maneira de fazer programação orientada a objetos em Go.

# Orientação a Serviços
Gerenciaremos as requisições às funções da aplicação através da orquestração de serviços, como por exemplo, o serviço ServicoProcessamento que encapsula a lógica de processamento de um DadosDocente (informações de cada linha do arquivo CSV). Essa função usa um PesquisadorRepository para salvar os dados e um EventBus para publicar eventos. Também trata de erros e retorna imediatamente se algum passo falhar.

As funções realizarBusca, escolherResultado, abrirCurriculo, analisarCurriculo e extrairDados, envolvem chamadas a pacotes externos ou código de infraestrutura que será muito específico para cada ambiente de produção. Deve-se implementar essas funções de acordo com a necessidade de cada ambiente de produção.

É necessário implementar a interface de usuário e a infraestrutura para manipulação de eventos e armazenamento de dados configuradas bem como a implementação do PesquisadorRepository e do EventBus irão tratar de suas próprias conexões de banco de dados e comunicação entre processos.

Um serviço é responsável por cada chamada da função Processar. Isso torna o serviço "stateless", o que é uma boa prática para garantir que cada chamada seja independente e que o estado não seja compartilhado entre chamadas. Isso simplifica a interação e impõem restrições ao usuário que só poderá fazer uma análise de cada vez, isso pode ser realizado com o uso de sessão de usuário. Caso haja alguma necessidade onde o estado precisa ser compartilhado entre chamadas (como uma sessão de usuário), pode ser preciso reconsiderar esse design.

## Mantendo a consistência com a arquitetura proposta
Precisamos manter a organização da estrutura de diretórios do seu projeto coerente com a Arquitetura de Solução. No geral sobre onde essas interfaces e funções poderiam residir em na arquitetura de software da seguinte forma:

PesquisadorRepository é uma interface de domínio, por isso deve estar no pacote de domínio do seu projeto. Por exemplo, se o seu projeto tiver um diretório domain, essa interface poderia estar em um arquivo chamado pesquisador_repository.go.

Evento é uma estrutura geral que poderia ser usada em todo o sistema para publicar eventos, então você poderia colocá-la em um pacote chamado eventos ou similar. Dependendo do tamanho e complexidade do seu sistema, este pacote poderia estar no nível superior do projeto, ou dentro do pacote domain se for principalmente usado para eventos de domínio.

EventBus é uma interface que abstrai a infraestrutura de publicação de eventos, por isso deve estar no pacote de domínio do seu projeto. Assim como PesquisadorRepository, essa interface poderia estar em um arquivo chamado event_bus.go.

ServicoProcessamento é um serviço de aplicação que orquestra a lógica de negócio, portanto, deve estar no pacote de aplicação do seu projeto. Você pode criar um novo arquivo chamado servico_processamento.go no pacote application ou similar.


# Frontend simplificado
Para integrar a interface de usuário criada pelo `main.js` e o `main.go` na pasta raiz do projeto, você pode seguir as seguintes etapas:

1. Mova os arquivos HTML e CSS necessários para a pasta `static/html` no diretório raiz do seu projeto.

2. Crie um novo arquivo chamado `main.go` na pasta raiz do projeto, onde você implementará o servidor HTTP em Go.

3. No arquivo `main.go`, importe os pacotes necessários:

```go
package main

import (
	"fmt"
	"log"
	"net/http"
)
```

4. Adicione um manipulador de rota para servir os arquivos estáticos, como HTML, CSS e JavaScript:

```go
func main() {
	fs := http.FileServer(http.Dir("static"))
	http.Handle("/static/", http.StripPrefix("/static/", fs))

	http.HandleFunc("/", indexHandler)

	fmt.Println("Servidor rodando em http://localhost:8080")
	log.Fatal(http.ListenAndServe(":8080", nil))
}
```

Neste exemplo, o servidor será iniciado na porta 8080. Certifique-se de escolher uma porta que não esteja em uso.

5. Implemente o manipulador de rota para a página inicial (`/`), onde você retornará o arquivo HTML principal:

```go
func indexHandler(w http.ResponseWriter, r *http.Request) {
	http.ServeFile(w, r, "static/html/index.html")
}
```

6. No arquivo `index.html`, adicione o seguinte código no cabeçalho para carregar o arquivo JavaScript `main.js`:

```html
<script src="static/js/main.js"></script>
```

Certifique-se de ajustar o caminho do arquivo `main.js` de acordo com a estrutura de pastas atual.

7. Execute o servidor Go executando o comando `go run main.go` no diretório raiz do projeto.

Após seguir essas etapas, você poderá acessar a interface de usuário em um navegador da web usando o endereço `http://localhost:8080` (ou a porta que você especificou). O arquivo HTML principal será servido pelo Go e o código JavaScript do `main.js` será carregado e executado na página.

Certifique-se de que o servidor Go esteja sendo executado enquanto você acessa a página para que as requisições HTTP funcionem corretamente.

## Frontend simples com Node.js e Express
1. Instalar o node.js com yarn
```bash
yarn add express
```
2. Criar o server.js
Contém todas as rotas para as páginas estáticas

3. Criar pasta static, com o index.html e index.js


## Messageria
yarn add amqplib
yarn add neo4j

Para instalar o RabbitMQ a partir do Terminal Integrado do VSCode usando o Yarn, siga as etapas abaixo:

1. Certifique-se de ter o Yarn instalado em seu ambiente. Se você ainda não o tiver instalado, consulte a documentação oficial do Yarn para obter instruções sobre como instalá-lo: [Yarn Installation](https://yarnpkg.com/getting-started/install).

2. Abra o Terminal Integrado no VSCode.

3. Navegue até a raiz do seu projeto ou para o diretório onde deseja instalar o RabbitMQ.

4. Execute o seguinte comando para instalar o RabbitMQ usando o Yarn:

   ```bash
   yarn add amqplib
   ```

   Este comando irá baixar e instalar a biblioteca `amqplib`, que é uma biblioteca popular do Node.js para trabalhar com o RabbitMQ.

Após a conclusão bem-sucedida da instalação, você poderá importar e usar a biblioteca `amqplib` em seu código para se conectar e interagir com o RabbitMQ.

Certifique-se de ajustar o código conforme necessário para sua configuração específica do RabbitMQ. Além disso, verifique se você tem o RabbitMQ instalado e configurado corretamente em seu ambiente antes de usar a biblioteca `amqplib`.

## Atualização do Node
Para atualizar o Node.js para a versão mais recente usando o Yarn no Terminal Integrado do VSCode, você pode seguir as etapas abaixo:

1. Verifique a versão atual do Node.js instalada em seu ambiente. No Terminal Integrado, execute o seguinte comando:

   ```bash
   node --version
   ```

   Isso mostrará a versão atual do Node.js instalada.

2. No Terminal Integrado, execute o seguinte comando para atualizar o Yarn para a versão mais estável:

   ```bash
   yarn set version latest
   ```

   Isso garantirá que você tenha a versão mais estável do Yarn instalada.

3. Em seguida, você precisará atualizar o Node.js usando o Yarn. No Terminal Integrado, execute o seguinte comando:

   ```bash
   yarn dlx n stable
   ```

   Isso usará o utilitário `n` do Yarn para baixar e instalar a versão mais estável do Node.js.

4. Após a conclusão bem-sucedida, verifique se o Node.js foi atualizado. No Terminal Integrado, execute novamente o comando:

   ```bash
   node --version
   ```

   Ele deve exibir a versão mais recente do Node.js que você instalou.

Lembre-se de que a atualização do Node.js pode levar algum tempo, pois envolve a instalação de uma nova versão. Certifique-se de estar conectado à Internet durante o processo de atualização.

Após atualizar o Node.js, você pode continuar usando o Yarn normalmente para instalar suas dependências do projeto.


## instalar o n
Pode-se usar o `npx` em vez do `yarn dlx`. O `npx` permite executar pacotes executáveis diretamente sem a necessidade de instalação global.

No Terminal Integrado do VSCode, execute o seguinte comando:

```bash
npx n latest
```

Isso irá baixar e instalar a versão mais recente do Node.js usando o pacote `n`.

Após a instalação ser concluída, você pode verificar se o Node.js foi atualizado corretamente executando o seguinte comando:

```bash
node --version
```


## Persistência no MongoDB
1. Execute o seguinte comando para instalar o pacote oficial do MongoDB para Node.js usando o Yarn:

   ```bash
   yarn add mongodb
   ```

   Este comando irá baixar e instalar o pacote `mongodb`, que é uma biblioteca do Node.js para interagir com o MongoDB.

Após a conclusão bem-sucedida da instalação, você poderá importar e usar a biblioteca `mongodb` em seu código para se conectar e interagir com o banco de dados MongoDB.

Lembre-se de configurar corretamente a conexão com o MongoDB em seu código, fornecendo as informações de host, porta, nome do banco de dados e credenciais (se aplicável) adequadas para se conectar ao banco de dados.

Certifique-se de ajustar o código conforme necessário para sua configuração específica do MongoDB. Além disso, verifique se você tem o MongoDB instalado e em execução em seu ambiente antes de usar a biblioteca `mongodb`.

## Persistência no Neo4j
Para instalar o Neo4j mais atualizado a partir do Terminal Integrado do VSCode usando o Yarn, siga as etapas abaixo:

1. Execute o seguinte comando para instalar o pacote oficial do Neo4j usando o Yarn:

   ```bash
   yarn add neo4j
   ```

   Este comando irá baixar e instalar o pacote `neo4j`, que é uma biblioteca do Node.js para interagir com o Neo4j.

Após a conclusão bem-sucedida da instalação, você poderá importar e usar a biblioteca `neo4j` em seu código para se conectar e interagir com o banco de dados Neo4j.

Lembre-se de configurar corretamente a conexão com o Neo4j em seu código, fornecendo as informações de host, porta, usuário e senha adequadas para se conectar ao banco de dados.

Certifique-se de ajustar o código conforme necessário para sua configuração específica do Neo4j. Além disso, verifique se você tem o Neo4j instalado e em execução em seu ambiente antes de usar a biblioteca `neo4j`.


## Verificação das dependências
Para verificar se todas as versões e dependências do projeto estão corretamente instaladas, você pode seguir as etapas abaixo:

1. Certifique-se de estar no diretório raiz do projeto no Terminal Integrado do VSCode.

2. Execute o comando a seguir para listar todas as dependências instaladas:

   ```bash
   yarn list --depth=0
   ```

   Esse comando exibirá todas as dependências do projeto junto com suas versões instaladas. Verifique se todas as dependências necessárias estão presentes e se suas versões estão corretas.

3. Verifique se todas as dependências estão atualizadas executando o seguinte comando:

   ```bash
   yarn outdated
   ```

   Esse comando exibirá uma lista das dependências que têm versões mais recentes disponíveis. Verifique se há alguma dependência desatualizada e considere atualizá-las, se necessário.

4. Verifique a versão do Node.js instalada executando o seguinte comando:

   ```bash
   node --version
   ```

   Verifique se a versão exibida corresponde à versão desejada do Node.js para o projeto.

5. Verifique a versão do MongoDB instalada executando o seguinte comando:

   ```bash
   mongod --version
   ```

   Verifique se a versão exibida corresponde à versão desejada do MongoDB para o projeto.

6. Verifique a versão do RabbitMQ instalada executando o seguinte comando:

   ```bash
   rabbitmqctl status
   ```

   Verifique se as informações exibidas correspondem à instalação correta e à versão desejada do RabbitMQ.

Essas etapas permitirão que você verifique se todas as versões e dependências do projeto estão corretamente instaladas. Se alguma dependência estiver faltando ou desatualizada, você pode atualizá-la ou instalá-la usando o Yarn.