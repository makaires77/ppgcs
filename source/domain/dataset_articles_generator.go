//go:build generator
// +build generator

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
)

// Definir um mutex e um contador global para o progresso
var mutex = &sync.Mutex{}
var progressCounter int
var (
	fetchCrossRefCount int
	scrapeArticleCount int
)

// Definir estruturas que representam o currículo e as informações processadas
type Curriculum struct {
	Labels          string                                  `json:"Labels"`
	Name            string                                  `json:"Name"`
	InfPes          interface{}                             `json:"InfPes"`
	Resumo          map[string]string                       `json:"Resumo"`
	Identificacao   map[string]string                       `json:"Identificacao"`
	Endereco        map[string]string                       `json:"Endereco"`
	Formacao        map[string]string                       `json:"Formacao"`
	Complementar    map[string]string                       `json:"Complementar"`
	Atuacao         map[string][]AtuacaoItem                `json:"Atuacao"`
	Pesquisa        map[string]string                       `json:"Pesquisa"`
	Desenvolvimento map[string]string                       `json:"Desenvolvimento"`
	AtuacaoAreas    map[string]string                       `json:"AtuacaoAreas"`
	Idiomas         map[string]string                       `json:"Idiomas"`
	Inovacao        map[string]string                       `json:"Inovacao"`
	Producoes       map[string]map[string]interface{}       `json:"Producoes"`
	JCR2            map[string]JCR2Info                     `json:"JCR2"`
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

type AtuacaoItem struct {
	VinculoInstitucional string `json:"Vínculo institucional"`
	// Outros campos conforme necessário
}

type JCR2Info struct {
	Titulo       string `json:"titulo"`
	DOI          string `json:"doi"`
	ImpactFactor string `json:"impact-factor"`
}

type ProcessedCurriculum struct {
	IDLattes string             `json:"idLattes"`
	Name     string             `json:"name"`
	Articles []ProcessedArticle `json:"articles"`
}

type ProcessedArticle struct {
	Year         int    `json:"year"`
	Title        string `json:"title"`
	DOI          string `json:"doi"`
	ImpactFactor string `json:"impactFactor"`
	Abstract     string `json:"abstract"`
}

// Funções auxiliares
// Extrair o ID Lattes de um currículo
func extrairIDLattes(infPes []string) (string, error) {
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

// Buscar informações somente do Título de um artigo na página HTML do DOI
func scrapeArticleTitle(doi string) (string, error) {
	if !validarDOI(doi) {
		return "", fmt.Errorf("DOI inválido")
	}

	url := "https://doi.org/" + doi

	// Realiza a requisição ao serviço de scraping
	resp, err := http.Get(url)
	if err != nil {
		log.Printf("Erro na requisição de scraping: %v", err)
		return "", err
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		log.Printf("Falha na resposta do serviço de scraping: StatusCode=%d", resp.StatusCode)
		return "", fmt.Errorf("falha ao acessar serviço de scraping, StatusCode=%d", resp.StatusCode)
	}

	// Decodificar resposta para extrair as informações necessárias
	var result struct {
		ScrapedTitle string `json:"title"`
		// Outros campos podem ser adicionados conforme necessário
	}

	err = json.NewDecoder(resp.Body).Decode(&result)
	if err != nil {
		log.Printf("Erro ao decodificar resposta do scraping: %v", err)
		return "", err
	}

	if result.ScrapedTitle == "" {
		log.Printf("Título não encontrado ou inválido no scraping para DOI: %s", doi)
		return "", fmt.Errorf("título não encontrado no scraping")
	}

	return result.ScrapedTitle, nil
}

// Requisitar Título e Resumo do CrossRef através do endpoint works
func fetchCrossRefInfo(doi string) (string, string, error) {
	log.Printf("Iniciando fetchCrossRefInfo para DOI: %s", doi)
	if !validarDOI(doi) {
		return "", "", fmt.Errorf("DOI inválido: %s", doi)
	}

	url := "https://api.crossref.org/works/" + doi
	resp, err := http.Get(url) // Implementar timeout
	if err != nil {
		log.Printf("Erro na requisição HTTP para DOI %s: %v", doi, err)
		return "", "", err
	}
	defer resp.Body.Close()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		log.Printf("Falha na resposta do CrossRef para DOI %s: StatusCode=%d", doi, resp.StatusCode)
		return "", "", fmt.Errorf("falha ao acessar crossref, StatusCode=%d", resp.StatusCode)
	}

	var result struct {
		Message struct {
			Title    []string `json:"title"`
			Abstract string   `json:"abstract"`
		} `json:"message"`
	}

	err = json.NewDecoder(resp.Body).Decode(&result)
	if err != nil {
		log.Printf("Erro ao decodificar JSON para DOI %s: %v", doi, err)
		return "", "", err
	}

	mutex.Lock()
	fetchCrossRefCount++
	mutex.Unlock()
	log.Printf("fetchCrossRefInfo completado para DOI %s, contador atualizado: %d", doi, fetchCrossRefCount)

	title := "Título não encontrado"
	if len(result.Message.Title) > 0 && result.Message.Title[0] != "" {
		title = result.Message.Title[0]
	} else {
		log.Printf("Título não encontrado ou inválido para DOI: %s", doi)
	}

	abstract := result.Message.Abstract
	if abstract == "" {
		log.Printf("Resumo não encontrado ou inválido para DOI: %s", doi)
	}

	return title, abstract, nil
}

func validarDOI(doi string) bool {
	// Expressão regular para validar um DOI
	re := regexp.MustCompile(`^10.\d{4,9}/[-._;()/:A-Z0-9]+$`)

	// Verificar se o input é uma URL e extrair apenas a parte do DOI
	if strings.Contains(doi, "http://dx.doi.org/") {
		doi = strings.TrimPrefix(doi, "http://dx.doi.org/")
	} else if strings.Contains(doi, "https://doi.org/") {
		doi = strings.TrimPrefix(doi, "https://doi.org/")
	}

	// Validar o DOI com a expressão regular
	return re.MatchString(doi)
}

// Buscar informações de título e resumo na página do DOI
func scrapeArticleInfo(doi string) (string, string, error) {
	log.Printf("Iniciando scrapeArticleInfo para DOI: %s", doi)
	if !validarDOI(doi) {
		log.Printf("DOI inválido: %s", doi)
		return "", "", fmt.Errorf("DOI inválido: %s", doi)
	}

	url := "https://doi.org/" + doi

	resp, err := http.Get(url)
	if err != nil {
		log.Printf("Erro na requisição de scraping para DOI %s: %v", doi, err)
		return "", "", err
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		log.Printf("Falha na resposta do serviço de scraping para DOI %s: StatusCode=%d", doi, resp.StatusCode)
		return "", "", fmt.Errorf("falha ao acessar serviço de scraping, StatusCode=%d", resp.StatusCode)
	}

	var result struct {
		ScrapedTitle    string `json:"title"`
		ScrapedAbstract string `json:"abstract"`
	}

	err = json.NewDecoder(resp.Body).Decode(&result)
	if err != nil {
		log.Printf("Erro ao decodificar resposta do scraping para DOI %s: %v", doi, err)
		return "", "", err
	}

	if result.ScrapedTitle == "" {
		log.Printf("Título não encontrado ou inválido no scraping para DOI: %s", doi)
		return "", "", fmt.Errorf("título não encontrado no scraping")
	}

	mutex.Lock()
	scrapeArticleCount++
	mutex.Unlock()
	log.Printf("scrapeArticleInfo completado para DOI %s, contador atualizado: %d", doi, scrapeArticleCount)

	return result.ScrapedTitle, result.ScrapedAbstract, nil
}

// Processar um único currículo e retorna um ProcessedCurriculum
func processCurriculum(curriculum Curriculum) ProcessedCurriculum {
	log.Printf("Iniciando processamento de currículo: %s", curriculum.Name)

	var processed ProcessedCurriculum

	// Normalizar InfPes dentro da função
	normalizedInfPes := normalizeInfPes(curriculum.InfPes)

	idLattes, err := extrairIDLattes(normalizedInfPes)
	if err != nil {
		log.Printf("Erro ao extrair ID Lattes de %s: %v", curriculum.Name, err)
		return ProcessedCurriculum{}
	}

	processed.IDLattes = idLattes
	processed.Name = curriculum.Name

	re := regexp.MustCompile(`, (\d{4})\.`)
	if producaoBibliografica, ok := curriculum.Producoes["Produção bibliográfica"]; ok {
		if artigos, ok := producaoBibliografica["Artigos completos publicados em periódicos"]; ok {
			for chave, artigo := range artigos.(map[string]interface{}) {
				titulo := artigo.(string)
				var artigoProcessado ProcessedArticle

				matches := re.FindStringSubmatch(titulo)
				if len(matches) > 1 {
					ano, err := strconv.Atoi(matches[1])
					if err != nil {
						log.Printf("Erro ao converter ano em artigo de %s: %v", curriculum.Name, err)
						continue
					}
					artigoProcessado.Year = ano
				}

				artigoProcessado.Title = titulo

				chaveJCR2 := strings.TrimRight(chave, ".")
				indiceJCR2, err := strconv.Atoi(chaveJCR2)
				if err != nil {
					log.Printf("Erro ao converter chave para int em artigo de %s: %v", curriculum.Name, err)
					continue
				}
				chaveJCR2 = strconv.Itoa(indiceJCR2 - 1)

				if jcr2Info, ok := curriculum.JCR2[chaveJCR2]; ok {
					artigoProcessado.DOI = jcr2Info.DOI
					artigoProcessado.ImpactFactor = jcr2Info.ImpactFactor

					title, abstract, err := fetchCrossRefInfo(jcr2Info.DOI)
					if err == nil && title != "Título não encontrado" {
						artigoProcessado.Title = title
						artigoProcessado.Abstract = abstract
					} else {
						scrapedTitle, scrapedAbstract, err := scrapeArticleInfo(jcr2Info.DOI)
						if err == nil {
							artigoProcessado.Title = scrapedTitle
							artigoProcessado.Abstract = scrapedAbstract
						}
					}
				}

				processed.Articles = append(processed.Articles, artigoProcessado)
			}
		}
	}

	log.Printf("Processamento de currículo completo: %s", curriculum.Name)
	return processed
}

// Ler o arquivo JSON e retorna um slice de Curriculum
func readJSON(filename string) ([]Curriculum, error) {
	bytes, err := os.ReadFile(filename)
	if err != nil {
		return nil, err
	}

	var curriculums []Curriculum
	err = json.Unmarshal(bytes, &curriculums)
	if err != nil {
		return nil, err
	}

	return curriculums, nil
}

// Salvar os dados processados em um arquivo JSON
func saveToJSON(filename string, data []ProcessedCurriculum) error {
	file, err := json.MarshalIndent(data, "", "  ")
	if err != nil {
		return err
	}

	return os.WriteFile(filename, file, 0644)
}

// Atualizar o progresso para alimentar o monitoramento em tempo real
func updateProgress(filePath string, total int) {
	mutex.Lock()
	defer mutex.Unlock()

	progressCounter++
	os.WriteFile(filePath, []byte(fmt.Sprintf("%d", progressCounter)), 0644)
}

// Função principal
func main() {
	// Configurar o arquivo de log
	logFile, err := os.OpenFile("logs/dataset_generator.log", os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		log.Fatalf("Erro ao abrir o arquivo de log: %v", err)
	}
	defer logFile.Close()

	// Definir o logger para escrever no arquivo
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
	outputFilename := filepath.Join(baseRepoDir, "data", "output", "output_go_cpu_multithreads.json")
	progressFilePath := filepath.Join(baseRepoDir, "data", "input", "progress.txt")

	curriculums, err := readJSON(inputFilename)
	if err != nil {
		fmt.Printf("Erro ao ler arquivo JSON: %v\n", err)
		os.Exit(1)
	}

	processedCurriculums := make([]ProcessedCurriculum, len(curriculums))
	var wg sync.WaitGroup

	for i, curriculum := range curriculums {
		wg.Add(1)

		go func(i int, curr Curriculum) {
			defer wg.Done()

			processedCurriculums[i] = processCurriculum(curr)

			updateProgress(progressFilePath, len(curriculums))
		}(i, curriculum)
	}

	wg.Wait()

	err = saveToJSON(outputFilename, processedCurriculums)
	if err != nil {
		fmt.Printf("Erro ao salvar os currículos processados: %v\n", err)
		os.Exit(1)
	}

	// Imprimir contadores de requisições com sucesso ao CrossRef e à alternativa
	fmt.Printf("Requisições com êxito para CrossRef: %d\n", fetchCrossRefCount)
	fmt.Printf("Requisições por scraping de artigos: %d\n", scrapeArticleCount)
}

/*
compilar pela tag que define a versão do main usada:
cd gml_classifier-1/source/domain
go build -tags generator -o dataset_articles_generator_linux
*/
