//go:build sockets
// +build sockets

package main

import (
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"sync"

	// _ "net/http/pprof"
	"encoding/json"
	"strconv"
	"strings"

	"github.com/PuerkitoBio/goquery"
)

// Definir estruturas para representar os dados.
type CurriculumSockets struct {
	Labels          string                                  `json:"labels"`
	Name            string                                  `json:"name"`
	InfPes          []string                                `json:"InfPes"`
	Resumo          map[string]string                       `json:"Resumo"`
	Identificacao   map[string]string                       `json:"Identificação"`
	Endereco        map[string]string                       `json:"Endereço"`
	Formacao        map[string]string                       `json:"Formação acadêmica/titulação"`
	Complementar    map[string]string                       `json:"Formação Complementar"`
	Atuacao         map[string]map[string]map[string]string `json:"Atuação Profissional"`
	Pesquisa        map[string]string                       `json:"Projetos de pesquisa"`
	Desenvolvimento map[string]string                       `json:"Projetos de desenvolvimento"`
	AtuacaoAreas    map[string]string                       `json:"Áreas de atuação"`
	Idiomas         map[string]string                       `json:"Idiomas"`
	Inovacao        map[string]string                       `json:"Inovação"`
	Producoes       map[string]map[string]interface{}       `json:"Produções"`
	JCR2            map[string]JCR2InfoSockets              `json:"JCR2"`
	Bancas          map[string]map[string]map[string]string `json:"Bancas"`
	// Eventos        map[string]map[string]map[string]string `json:"Eventos"`
}

type JCR2InfoSockets struct {
	Titulo       string `json:"titulo"`
	DOI          string `json:"doi"`
	ImpactFactor string `json:"impact-factor"`
}

type ProcessedCurriculumSockets struct {
	IDLattes string                    `json:"idLattes"`
	Name     string                    `json:"name"`
	Articles []ProcessedArticleSockets `json:"articles"`
}

type ProcessedArticleSockets struct {
	Year         int    `json:"year"`
	Title        string `json:"title"`
	DOI          string `json:"doi"`
	ImpactFactor string `json:"impactFactor"`
	Abstract     string `json:"abstract"`
}

func extrairIDLattesSockets(infPes []string) (string, error) {
	for _, str := range infPes {
		if strings.Contains(str, "ID Lattes:") {
			parts := strings.Split(str, ":")
			if len(parts) > 1 {
				return strings.TrimSpace(parts[1]), nil
			}
		}
	}
	return "", fmt.Errorf("ID Lattes não encontrado")
}

func fetchCrossRefInfoSockets(doi string) (string, string, error) {
	url := "https://api.crossref.org/works/" + doi
	resp, err := http.Get(url)
	if err != nil {
		return "", "", err
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		return "", "", fmt.Errorf("falha ao acessar crossref")
	}

	var result struct {
		Message struct {
			Title    []string `json:"title"`
			Abstract string   `json:"abstract"`
		} `json:"message"`
	}

	err = json.NewDecoder(resp.Body).Decode(&result)
	if err != nil {
		return "", "", err
	}

	var title string
	if len(result.Message.Title) > 0 {
		title = result.Message.Title[0]
	} else {
		title = "Título não encontrado"
	}

	return title, result.Message.Abstract, nil
}

func scrapeArticleInfoSockets(doi string) (string, string, error) {
	url := "https://doi.org/" + doi
	resp, err := http.Get(url)
	if err != nil {
		return "", "", err
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		return "", "", fmt.Errorf("falha ao acessar a página do doi")
	}

	doc, err := goquery.NewDocumentFromReader(resp.Body)
	if err != nil {
		return "", "", err
	}

	var title, abstract string
	doc.Find("TITLE, Title, title, TÍTULO, Título, título").Each(func(i int, s *goquery.Selection) {
		if title == "" {
			title = s.Text()
		}
	})
	doc.Find("ABSTRACT, Abstract, abstract, RESUMO, Resumo, resumo").Each(func(i int, s *goquery.Selection) {
		if abstract == "" {
			abstract = s.Text()
		}
	})

	return title, abstract, nil
}

// // Função para processar os dados de cada currículo.
// func processCurriculumSockets(curriculum Curriculum) ProcessedCurriculum {
// 	var processedCurriculum ProcessedCurriculum
// 	idLattes, err := extrairIDLattes(curriculum.InfPes)
// 	if err != nil {
// 		fmt.Printf("Erro ao extrair ID Lattes: %v\n", err)
// 		return ProcessedCurriculum{}
// 	}

// 	processedCurriculum.IDLattes = idLattes
// 	processedCurriculum.Name = curriculum.Name

// 	// Expressão regular para encontrar o ano
// 	re := regexp.MustCompile(`, (\d{4})\.`)

// 	if producaoBibliografica, ok := curriculum.Producoes["Produção bibliográfica"]; ok {
// 		if artigosInterface, ok := producaoBibliografica["Artigos completos publicados em periódicos"]; ok {
// 			artigosPublicados, ok := artigosInterface.(map[string]interface{})
// 			if !ok {
// 				fmt.Println("Erro: tipo inesperado para artigos publicados")
// 				return ProcessedCurriculum{}
// 			}

// 			for chave, tituloInterface := range artigosPublicados {
// 				titulo, ok := tituloInterface.(string)
// 				if !ok {
// 					fmt.Println("Erro: tipo inesperado para título do artigo")
// 					continue
// 				}

// 				var processedArticle ProcessedArticle

// 				// Encontrar o ano usando a expressão regular
// 				matches := re.FindStringSubmatch(titulo)
// 				if len(matches) > 1 {
// 					ano, err := strconv.Atoi(matches[1])
// 					if err != nil {
// 						fmt.Printf("Erro ao converter ano para int: %v\n", err)
// 						continue
// 					}
// 					processedArticle.Year = ano
// 				}

// 				processedArticle.Title = titulo

// 				// Remover o ponto e subtrair 1 para obter a chave de JCR2 correspondente
// 				chaveSeq, err := strconv.Atoi(strings.TrimSuffix(chave, "."))
// 				if err != nil {
// 					fmt.Printf("Erro ao converter chave para int: %v\n", err)
// 					continue
// 				}
// 				chaveJCR2 := strconv.Itoa(chaveSeq - 1)

// 				if jcr2Info, ok := curriculum.JCR2[chaveJCR2]; ok {
// 					processedArticle.DOI = jcr2Info.DOI
// 					processedArticle.ImpactFactor = jcr2Info.ImpactFactor

// 					title, abstract, err := fetchCrossRefInfo(jcr2Info.DOI)
// 					if err == nil && title != "Título não encontrado" {
// 						processedArticle.Title = title
// 						processedArticle.Abstract = abstract
// 					} else {
// 						scrapedTitle, scrapedAbstract, err := scrapeArticleInfo(jcr2Info.DOI)
// 						if err == nil {
// 							processedArticle.Title = scrapedTitle
// 							processedArticle.Abstract = scrapedAbstract
// 						}
// 					}
// 				}

// 				processedCurriculum.Articles = append(processedCurriculum.Articles, processedArticle)
// 			}
// 		}
// 	}

// 	return processedCurriculum
// }

// Função para processar os dados de cada currículo com monitoramento por cada Goroutine.
func processCurriculumSockets(curriculum CurriculumSockets, index int, total int, progressFilePath string) ProcessedCurriculumSockets {
	var processedCurriculum ProcessedCurriculumSockets
	idLattes, err := extrairIDLattesSockets(curriculum.InfPes)
	if err != nil {
		fmt.Printf("Erro ao extrair ID Lattes: %v\n", err)
		return ProcessedCurriculumSockets{}
	}

	processedCurriculum.IDLattes = idLattes
	processedCurriculum.Name = curriculum.Name

	if producaoBibliografica, ok := curriculum.Producoes["Produção bibliográfica"]; ok {
		if artigosInterface, ok := producaoBibliografica["Artigos completos publicados em periódicos"]; ok {
			artigosPublicados, ok := artigosInterface.(map[string]interface{})
			if !ok {
				fmt.Println("Erro: tipo inesperado para artigos publicados")
				return ProcessedCurriculumSockets{}
			}

			for chave, tituloInterface := range artigosPublicados {
				titulo, ok := tituloInterface.(string)
				if !ok {
					fmt.Println("Erro: tipo inesperado para título do artigo")
					continue
				}

				var processedArticle ProcessedArticleSockets
				processedArticle.Title = titulo

				// Remover o ponto e subtrair 1 para obter a chave de JCR2 correspondente
				chaveSeq, err := strconv.Atoi(strings.TrimSuffix(chave, "."))
				if err != nil {
					fmt.Printf("Erro ao converter chave para int: %v\n", err)
					continue
				}
				chaveJCR2 := strconv.Itoa(chaveSeq - 1)

				if jcr2Info, ok := curriculum.JCR2[chaveJCR2]; ok {
					processedArticle.DOI = jcr2Info.DOI
					processedArticle.ImpactFactor = jcr2Info.ImpactFactor

					title, abstract, err := fetchCrossRefInfoSockets(jcr2Info.DOI)
					if err == nil && title != "Título não encontrado" {
						processedArticle.Title = title
						processedArticle.Abstract = abstract
					} else {
						scrapedTitle, scrapedAbstract, err := scrapeArticleInfoSockets(jcr2Info.DOI)
						if err == nil {
							processedArticle.Title = scrapedTitle
							processedArticle.Abstract = scrapedAbstract
						}
					}
				}

				processedCurriculum.Articles = append(processedCurriculum.Articles, processedArticle)
			}
		}
	}

	// Atualizar progresso após processamento de cada currículo
	updateProgressSockets(index+1, total, progressFilePath)

	return processedCurriculum
}

// Função para processar dados de currículo sem monitorar tempo
func main() {

	go func() {
		fmt.Println("Starting pprof server at http://localhost:6060")
		if err := http.ListenAndServe("localhost:6060", nil); err != nil {
			fmt.Printf("Error starting pprof server: %v\n", err)
		}
	}()

	var baseRepoDir string

	// Detectar o sistema operacional
	if runtime.GOOS == "windows" {
		// Definir caminho para Windows
		baseRepoDir = "C:\\Users\\marcos.aires\\gml_classifier-1"
	} else {
		// Definir caminho padrão para Unix/Linux/MacOS
		baseRepoDir = "/home/mak/gml_classifier-1"
	}

	// Construir os caminhos usando filepath.Join
	inputFilename := filepath.Join(baseRepoDir, "data", "input", "normalized_dict_list.json")
	outputFilename := filepath.Join(baseRepoDir, "data", "output", "output_go_cpu_multithreads.json")
	progressFilePath := filepath.Join(baseRepoDir, "data", "input", "progress.txt")

	file, err := os.ReadFile(inputFilename)
	if err != nil {
		fmt.Printf("Erro ao ler arquivo: %v\n", err)
		os.Exit(1)
	}

	var curriculums []CurriculumSockets
	err = json.Unmarshal(file, &curriculums)
	if err != nil {
		fmt.Printf("Erro ao decodificar JSON: %v\n", err)
		os.Exit(1)
	}

	var wg sync.WaitGroup
	processedCurriculums := make([]ProcessedCurriculumSockets, len(curriculums))

	for i, curriculum := range curriculums {
		wg.Add(1)
		go func(i int, currSockets CurriculumSockets) {
			defer wg.Done()
			processedCurriculums[i] = processCurriculumSockets(currSockets, i, len(curriculums), progressFilePath)
		}(i, curriculum)
	}

	wg.Wait()

	processedData, err := json.Marshal(processedCurriculums)
	if err != nil {
		fmt.Printf("Erro ao codificar para JSON: %v\n", err)
		os.Exit(1)
	}

	err = os.WriteFile(outputFilename, processedData, 0644)
	if err != nil {
		fmt.Printf("Erro ao escrever arquivo: %v\n", err)
		os.Exit(1)
	}
}

func updateProgressSockets(current, total int, filePath string) {
	err := os.WriteFile(filePath, []byte(fmt.Sprintf("%d", current)), 0644)
	if err != nil {
		fmt.Printf("Erro ao escrever no arquivo de progresso: %v\n", err)
	}
}

/*
compilar com os comandos:
cd gml_classifier-1/source/domain
go build -tags sockets -o dataset_articles_generator_sockets


Características do Processamento Paralelo:
 Concorrência: O uso de goroutines permite o processamento concorrente de cada Curriculum. Isso pode acelerar significativamente o processamento, especialmente em sistemas com múltiplos núcleos de CPU.

 Sincronização: O uso de sync.WaitGroup garante que o programa principal espere que todas as tarefas paralelas sejam concluídas antes de prosseguir para a serialização e gravação dos dados processados.

 Escalabilidade: Esta abordagem é escalável, pois pode processar um grande número de Curriculum em paralelo, limitado principalmente pela capacidade do hardware.

A função a seguir é um exemplo, que utiliza goroutines para o processamento em paralelo junto com sincronização de tarefas usando sync.WaitGroup. Detalhes da implmentação:

 Servidor pprof: A função cria goroutine para executar um servidor pprof, ferramenta para visualizar e analisar perfis de desempenho de aplicativos Go, que permite inspeção do desempenho do programa em tempo real.

 Leitura e Deserialização de Dados: O programa lê um arquivo JSON de entrada e deserializa seu conteúdo em uma slice de Curriculum.

 Processamento Paralelo com Goroutines: Para cada Curriculum na slice, a função main dispara uma goroutine para processá-lo de forma independente. Isso é feito invocando a função processCurriculum dentro de uma goroutine anônima. O sync.WaitGroup (wg) é usado para sincronizar todas essas goroutines.

 wg.Add(1): Incrementa o contador do WaitGroup antes de iniciar cada goroutine.

 defer wg.Done(): Decrementa o contador do WaitGroup quando a goroutine termina.

 Esperando pelas Goroutines Terminarem: wg.Wait() bloqueia até que todas as goroutines disparadas tenham terminado (ou seja, até que o contador do WaitGroup volte a zero).

 Serialização e Escrita de Dados de Saída: Após o processamento, os dados processados são serializados para JSON e salvos em um arquivo de saída.

 Função processCurriculum: Esta função, chamada dentro de cada goroutine, processa individualmente cada Curriculum. Após o processamento de cada Curriculum, a função updateProgress é chamada para atualizar o progresso do processamento.

 Função updateProgress: Esta função escreve o progresso atual em um arquivo. Isso é feito para cada Curriculum processado.

*/
