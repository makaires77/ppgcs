package main

import (
	"encoding/csv"
	"fmt"
	"log"
	"os"
	"sync"
)

type Author struct {
	// Defina a estrutura do autor aqui
}

type Student struct {
	// Defina a estrutura do discente aqui
}

// Função de comparação de nomes
func compareNames(authorName, studentName string, wg *sync.WaitGroup, progress chan<- string, results chan<- bool) {
	defer wg.Done()

	// Lógica de comparação de nomes aqui
	// ...
	// Exemplo de uso:
	// if similarity > 0.86 {
	//     progress <- fmt.Sprintf("Similaridade encontrada! Autor: %s | Discente: %s", authorName, studentName)
	//     results <- true
	// }
}

func main() {
	// Abrir o arquivo CSV dos autores
	fileAuthors, err := os.Open("caminho_do_arquivo/autores.csv")
	if err != nil {
		log.Fatalf("Falha ao abrir o arquivo CSV dos autores: %v", err)
	}
	defer fileAuthors.Close()

	fmt.Println("Lendo o arquivo CSV dos autores...")

	// Ler os registros do arquivo CSV dos autores
	readerAuthors := csv.NewReader(fileAuthors)
	readerAuthors.Comma = ';'
	readerAuthors.LazyQuotes = true

	authorRecords, err := readerAuthors.ReadAll()
	if err != nil {
		log.Fatalf("Falha ao ler o arquivo CSV dos autores: %v", err)
	}
	fmt.Printf("Total de registros de autores: %d\n", len(authorRecords))

	// Abrir o arquivo CSV dos discentes
	fileStudents, err := os.Open("caminho_do_arquivo/discentes.csv")
	if err != nil {
		log.Fatalf("Falha ao abrir o arquivo CSV dos discentes: %v", err)
	}
	defer fileStudents.Close()

	// Ler os registros do arquivo CSV dos discentes
	readerStudents := csv.NewReader(fileStudents)
	readerStudents.Comma = ';'
	readerStudents.LazyQuotes = true

	studentRecords, err := readerStudents.ReadAll()
	if err != nil {
		log.Fatalf("Falha ao ler o arquivo CSV dos discentes: %v", err)
	}
	fmt.Printf("Total de registros de discentes: %d\n", len(studentRecords))

	// Criar um canal para enviar atualizações de progresso
	progress := make(chan string)

	// Criar um canal para enviar os resultados da comparação de nomes (true se houver similaridade, false caso contrário)
	results := make(chan bool)

	// Criar uma WaitGroup para sincronizar as goroutines
	var wg sync.WaitGroup

	fmt.Println("Comparando nomes de autores com nomes de discentes...")

	// Chamar a função compareNames para cada combinação de autor e discente
	for _, authorRecord := range authorRecords {
		for _, studentRecord := range studentRecords {
			authorName := authorRecord[1]
			studentName := studentRecord[1]

			wg.Add(1)
			go compareNames(authorName, studentName, &wg, progress, results)
		}
	}

	// Goroutine para monitorar o canal de progresso e exibir informações
	go func() {
		for msg := range progress {
			fmt.Println(msg)
		}
	}()

	// Goroutine para coletar os resultados da comparação de nomes e contar a colaboração por docente
	go func() {
		docenteColaboracao := make(map[string]int)
		totalSimilarities := 0

		for res := range results {
			if res {
				totalSimilarities++
				// Atualizar a contagem de colaboração para o docente
<<<<<<< HEAD
				authorName := authorRecord[0]
				docenteColaboracao[authorName]++
=======
				for _, authorRecord := range authorRecords {
					authorName := authorRecord[1]
					docenteColaboracao[authorName]++
				}
>>>>>>> c51253137853d9681efc37ff1382c4b6f7ed1174
			}
		}

		// Calcular o percentual de colaboração por docente
		fmt.Println("\nTotal de colaboração por docente:")
		for docente, colaboracao := range docenteColaboracao {
			percentual := float64(colaboracao) / float64(len(studentRecords)) * 100
			fmt.Printf("%s: %.2f%%\n", docente, percentual)
		}

		// Calcular o percentual de colaboração no programa
		collaborationPercentage := float64(totalSimilarities) / float64(len(authorRecords)*len(studentRecords)) * 100

		fmt.Printf("\nResumo da comparação:\n")
		fmt.Printf("Artigos com similaridades: %d\n", totalSimilarities)
		fmt.Printf("Total de artigos: %d\n", len(authorRecords)*len(studentRecords))
		fmt.Printf("Total de discentes: %d\n", len(studentRecords))
		fmt.Printf("Percentual de colaboração no programa: %.2f%%\n", collaborationPercentage)

		// Fechar os canais de progresso e resultados
		close(progress)
		close(results)
	}()

	// Aguardar a conclusão de todas as goroutines
	wg.Wait()

	fmt.Println("Leitura dos arquivos CSV concluída com sucesso.")
}
