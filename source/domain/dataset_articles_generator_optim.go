//go:build opt
// +build opt

package main

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"runtime"
	"strconv"
	"strings"
	"sync"

	"github.com/PuerkitoBio/goquery"
)

// Definir um mutex e um contador global para o progresso
var mutex = &sync.Mutex{}
var progressCounter int
var (
	fetchCrossRefCount int
	scrapeArticleCount int
)

// Definir estruturas que representam o currículo e as informações processadas
// Aqui renomeadas para evitar conflito com estruturas do outro arquivo com main.
type CurriculumOptimized struct {
	Labels          string                                  `json:"Labels"`
	Name            string                                  `json:"Name"`
	InfPes          interface{}                             `json:"InfPes"`
	Resumo          map[string]string                       `json:"Resumo"`
	Identificacao   map[string]string                       `json:"Identificacao"`
	Endereco        map[string]string                       `json:"Endereco"`
	Formacao        map[string]string                       `json:"Formacao"`
	Complementar    map[string]string                       `json:"Complementar"`
	Atuacao         map[string][]AtuacaoItemOptimized       `json:"Atuacao"`
	Pesquisa        map[string]string                       `json:"Pesquisa"`
	Desenvolvimento map[string]string                       `json:"Desenvolvimento"`
	AtuacaoAreas    map[string]string                       `json:"AtuacaoAreas"`
	Idiomas         map[string]string                       `json:"Idiomas"`
	Inovacao        map[string]string                       `json:"Inovacao"`
	Producoes       map[string]map[string]interface{}       `json:"Producoes"`
	JCR2            map[string]JCR2InfoOptimized            `json:"JCR2"`
	Bancas          map[string]map[string]map[string]string `json:"Bancas"`
}

func normalizeInfPes(data interface{}) []string {
	switch v := data.(type) {
	case string:
		// Se for string, transformar em slice de string
		return []string{v}
	case []interface{}:
		// Se for slice, converter cada elemento em string
		var result []string
		for _, item := range v {
			if str, ok := item.(string); ok {
				result = append(result, str)
			}
		}
		return result
	case map[string]interface{}:
		// Se for um mapa, converter valores em strings
		// ou extrair informações específicas do mapa
		var result []string
		for _, value := range v {
			if str, ok := value.(string); ok {
				result = append(result, str)
			}
		}
		return result
	}
	return nil // Retornar slice vazio ou algum comportamento padrão em caso de tipo inesperado
}

type AtuacaoItemOptimized struct {
	VinculoInstitucional string `json:"Vínculo institucional"`
	// Outros campos conforme necessário
}

type JCR2InfoOptimized struct {
	Titulo       string `json:"titulo"`
	DOI          string `json:"doi"`
	ImpactFactor string `json:"impact-factor"`
}

type ProcessedCurriculumOptimized struct {
	IDLattes string                      `json:"idLattes"`
	Name     string                      `json:"name"`
	Articles []ProcessedArticleOptimized `json:"articles"`
}

type ProcessedArticleOptimized struct {
	Year         int    `json:"year"`
	Title        string `json:"title"`
	DOI          string `json:"doi"`
	ImpactFactor string `json:"impactFactor"`
	Abstract     string `json:"abstract"`
}

// Variáveis globais para controlar a concorrência
var (
	maxGoroutines = 10 // Limite máximo de goroutines
	semaphore     chan struct{}
)

// Funções auxiliares adaptadas para evitar conflitos de nomeação
// Função de validação de DOI adaptada
func extractDOIFromURL(url string) string {
	re := regexp.MustCompile(`10.\d{4,9}/[-._;()/:A-Z0-9]+$`)
	if matches := re.FindStringSubmatch(url); len(matches) > 0 {
		return matches[0]
	}
	return ""
}

func validarDOI(doi string) bool {
	if strings.HasPrefix(doi, "http://dx.doi.org/") || strings.HasPrefix(doi, "https://doi.org/") {
		doi = extractDOIFromURL(doi)
	}
	re := regexp.MustCompile(`^10.\d{4,9}/[-._;()/:A-Z0-9]+$`)
	return re.MatchString(doi)
}

// Extrair o ID Lattes de um currículo
func extrairIDLattesOptimized(data interface{}) (string, error) {
	var infPes []string

	switch v := data.(type) {
	case string:
		infPes = []string{v}
	case []interface{}:
		for _, item := range v {
			if str, ok := item.(string); ok {
				infPes = append(infPes, str)
			}
		}
	case []string:
		infPes = v
	default:
		return "", fmt.Errorf("tipo inesperado para InfPes")
	}

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

// Trazer informações do CrossRef através do endpoint works
// func fetchCrossRefInfoOptimized(doi string) (string, string, error) {
func fetchCrossRefInfoOptimized(doi string) (string, error) {
	if !validarDOI(doi) {
		log.Printf("DOI inválido: %s", doi)
		return "", fmt.Errorf("DOI inválido: %s", doi)
	}

	url := "https://api.crossref.org/works/" + doi
	resp, err := http.Get(url)
	if err != nil {
		// return "", "", err
		return "", err
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		// return "", "", fmt.Errorf("falha ao acessar crossref")
		return "", fmt.Errorf("falha ao acessar crossref")
	}

	// Definir variáveis que serão servidas na saída da função
	var result struct {
		Message struct {
			Title []string `json:"title"`
			// Abstract string   `json:"abstract"`
		} `json:"message"`
	}

	// Incrementar contagem de requisições com sucesso
	err = json.NewDecoder(resp.Body).Decode(&result)
	if err == nil {
		mutex.Lock()
		fetchCrossRefCount++
		mutex.Unlock()
	}

	// Atualizar o conteúdo do título encontrado
	var title string
	if len(result.Message.Title) > 0 {
		title = result.Message.Title[0]
	} else {
		title = "Título não encontrado"
	}

	// return title, result.Message.Abstract, nil
	return title, nil
}

// Trazer informações de um artigo da página HTML do DOI
// func scrapeArticleInfoOptimized(doi string) (string, string, error) {
func scrapeArticleInfoOptimized(doi string) (string, error) {
	if !validarDOI(doi) {
		log.Printf("DOI inválido: %s", doi)
		return "", fmt.Errorf("DOI inválido: %s", doi)
	}

	url := "https://doi.org/" + doi
	resp, err := http.Get(url)
	if err != nil {
		// return "", "", err
		return "", err
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		// return "", "", fmt.Errorf("falha ao acessar a página do doi")
		return "", fmt.Errorf("falha ao acessar a página do doi")
	}

	doc, err := goquery.NewDocumentFromReader(resp.Body)
	if err == nil {
		mutex.Lock()
		scrapeArticleCount++
		mutex.Unlock()
	}

	// var title, abstract string
	var title string
	doc.Find("TITLE, Title, title, TÍTULO, Título, título").Each(func(i int, s *goquery.Selection) {
		if title == "" {
			title = s.Text()
		}
	})
	// doc.Find("ABSTRACT, Abstract, abstract, RESUMO, Resumo, resumo").Each(func(i int, s *goquery.Selection) {
	// 	if abstract == "" {
	// 		abstract = s.Text()
	// 	}
	// })

	// return title, abstract, nil
	return title, nil
}

// Processar um único currículo e retorna um ProcessedCurriculum
func processCurriculumOptimized(curriculum CurriculumOptimized, infPes []string) ProcessedCurriculumOptimized {
	var processed ProcessedCurriculumOptimized

	idLattes, err := extrairIDLattesOptimized(infPes)
	if err != nil {
		fmt.Printf("Erro ao extrair ID Lattes: %v\n", err)
		return ProcessedCurriculumOptimized{}
	}

	processed.IDLattes = idLattes
	processed.Name = curriculum.Name

	re := regexp.MustCompile(`, (\d{4})\.`) // Compilando a expressão regular fora do loop
	if producaoBibliografica, ok := curriculum.Producoes["Produção bibliográfica"]; ok {
		if artigos, ok := producaoBibliografica["Artigos completos publicados em periódicos"]; ok {
			for chave, artigo := range artigos.(map[string]interface{}) {
				titulo := artigo.(string)
				var artigoProcessado ProcessedArticleOptimized

				matches := re.FindStringSubmatch(titulo)
				if len(matches) > 1 {
					ano, err := strconv.Atoi(matches[1])
					if err != nil {
						fmt.Printf("Erro ao converter ano: %v\n", err)
						continue
					}
					artigoProcessado.Year = ano
				}

				artigoProcessado.Title = titulo

				// Corrigindo a chave para alinhar com o índice de JCR2
				chaveJCR2 := strings.TrimRight(chave, ".")
				indiceJCR2, err := strconv.Atoi(chaveJCR2)
				if err != nil {
					fmt.Printf("Erro ao converter chave para int: %v\n", err)
					continue
				}
				chaveJCR2 = strconv.Itoa(indiceJCR2 - 1)

				if jcr2Info, ok := curriculum.JCR2[chaveJCR2]; ok {
					artigoProcessado.DOI = jcr2Info.DOI
					artigoProcessado.ImpactFactor = jcr2Info.ImpactFactor

					// Utilizando fetchCrossRefInfo e scrapeArticleInfo
					// title, abstract, err := fetchCrossRefInfoOptimized(jcr2Info.DOI)
					title, err := fetchCrossRefInfoOptimized(jcr2Info.DOI)
					if err == nil && title != "Título não encontrado" {
						artigoProcessado.Title = title
						// artigoProcessado.Abstract = abstract
					} else {
						// scrapedTitle, scrapedAbstract, err := scrapeArticleInfoOptimized(jcr2Info.DOI)
						scrapedTitle, err := scrapeArticleInfoOptimized(jcr2Info.DOI)
						if err == nil {
							artigoProcessado.Title = scrapedTitle
							// artigoProcessado.Abstract = scrapedAbstract
						}
					}
				}

				processed.Articles = append(processed.Articles, artigoProcessado)
			}
		}
	}

	return processed
}

// Ler o arquivo JSON e retorna um slice de CurriculumOptimized
func readJSONOptimized(filename string) ([]CurriculumOptimized, error) {
	bytes, err := os.ReadFile(filename)
	if err != nil {
		return nil, err
	}

	var curriculums []CurriculumOptimized
	err = json.Unmarshal(bytes, &curriculums)
	if err != nil {
		return nil, err
	}

	return curriculums, nil
}

// saveToJSONOptimized salva dados processados em JSON
func saveToJSONOptimized(filename string, data []ProcessedCurriculumOptimized) error {
	file, err := json.MarshalIndent(data, "", "  ")
	if err != nil {
		return err
	}

	return os.WriteFile(filename, file, 0644)
}

// Atualizar o progresso para alimentar o monitoramento em tempo real
func updateProgressOptimized(filePath string, total int) {
	mutex.Lock()
	defer mutex.Unlock()

	progressCounter++
	os.WriteFile(filePath, []byte(fmt.Sprintf("%d", progressCounter)), 0644)
}

// Função principal para a versão otimizada
func main() {
	// Configurar o arquivo de log
	logFile, err := os.OpenFile("logs/dataset_generator.log", os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		log.Fatalf("Erro ao abrir o arquivo de log: %v", err)
	}
	defer logFile.Close()

	log.SetOutput(logFile)

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
	outputFilename := filepath.Join(baseRepoDir, "data", "output", "output_go_cpu_mthreadoptim.json")
	progressFilePath := filepath.Join(baseRepoDir, "data", "input", "progress_optimized.txt")

	curriculums, err := readJSONOptimized(inputFilename)
	if err != nil {
		fmt.Printf("Erro ao ler arquivo JSON: %v\n", err)
		os.Exit(1)
	}

	processedCurriculums := make([]ProcessedCurriculumOptimized, len(curriculums))
	var wg sync.WaitGroup
	semaphore = make(chan struct{}, maxGoroutines)

	for i, curriculum := range curriculums {
		wg.Add(1)
		semaphore <- struct{}{} // Adquirindo o semáforo
		go func(i int, curr CurriculumOptimized) {
			defer wg.Done()
			defer func() { <-semaphore }() // Liberando o semáforo

			// Normalize InfPes dentro da goroutine
			normalizedInfPes := normalizeInfPes(curr.InfPes)

			// Passar normalizedInfPes para processCurriculumOptimized
			processedCurriculums[i] = processCurriculumOptimized(curr, normalizedInfPes)

			updateProgressOptimized(progressFilePath, len(curriculums))
		}(i, curriculum)
	}
	wg.Wait()

	err = saveToJSONOptimized(outputFilename, processedCurriculums)
	if err != nil {
		fmt.Printf("Erro ao salvar os currículos processados: %v\n", err)
		os.Exit(1)
	}

	// Imprimir os contadores
	fmt.Printf("Requisições com êxito para CrossRef: %d\n", fetchCrossRefCount)
	fmt.Printf("Requisições por scraping de artigos: %d\n", scrapeArticleCount)
}

/*
locais de desenvolvimento linux e windows
cd gml_classifier-1/source/domain
cd C:\Users\marcos.aires\gml_classifier-1\source\domain

compilar pela tag que define a versão do main usada:
go build -tags opt -o dataset_articles_generator_optim_linux

Em Linux, para setar o SO, compilar com:
GOOS=linux GOARCH=amd64 go build -tags opt -o dataset_articles_generator_optim_linux

GOOS=windows GOARCH=amd64 go build -tags opt -o dataset_articles_generator_optim_windows.exe

Estando em Windows, para setar o SO compilar com:
SET GOOS=linux
SET GOARCH=amd64
go build -tags opt -o dataset_articles_generator_optim_linux

SET GOOS=windows
SET GOARCH=amd64
go build -tags opt -o dataset_articles_generator_optim_windows.exe



Características do Processamento Paralelo:

 Concorrência: O uso de goroutines permite o processamento concorrente de cada Curriculum. Isso pode acelerar significativamente o processamento, especialmente em sistemas com múltiplos núcleos de CPU.

 Sincronização: O uso de sync.WaitGroup garante que o programa principal espere que todas as tarefas paralelas sejam concluídas antes de prosseguir para a serialização e gravação dos dados processados.

 Escalabilidade: Esta abordagem é escalável, pois pode processar um grande número de Curriculum em paralelo, limitado principalmente pela capacidade do hardware.

A função otimizada utiliza goroutines para o processamento em paralelo junto com sincronização de tarefas usando sync.WaitGroup e controle do máximo de threads concorrência com uso de semáforos.

Trechos de código da implementação:

 Servidor pprof: Cria goroutine para executar um servidor pprof, ferramenta para visualizar e analisar perfis de desempenho de aplicativos Go, que permite inspeção do desempenho do programa em tempo real.

 Leitura e Deserialização de Dados: Lê um arquivo JSON de entrada e deserializa seu conteúdo em uma slice de Curriculum.

 Processamento Paralelo com Goroutines: Para cada Curriculum na slice, a função main dispara uma goroutine para processá-lo de forma independente. Isso é feito invocando a função processCurriculum dentro de uma goroutine anônima. O sync.WaitGroup (wg) é usado para sincronizar todas essas goroutines.

 wg.Add(1): Incrementa o contador do WaitGroup antes de iniciar cada goroutine.

 defer wg.Done(): Decrementa o contador do WaitGroup quando a goroutine termina.

 wg.Wait(): Espera pelas Goroutines Terminarem, bloqueia até que todas as goroutines disparadas tenham terminado (ou seja, até que o contador do WaitGroup volte a zero).

 Função saveToJSONOptimized: Serializa para JSON e escreve os dados de saída após o processamento e salva em arquivo de saída.

 Função processCurriculum: Esta função, chamada dentro de cada goroutine, processa individualmente cada Curriculum. Após o processamento de cada Curriculum, a função updateProgress é chamada para atualizar o progresso do processamento.

 Função updateProgress: Esta função escreve o progresso atual em um arquivo. Isso é feito para cada Curriculum processado.

*/
