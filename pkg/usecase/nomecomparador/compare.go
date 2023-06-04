package nomecomparador

import (
	"fmt"
	"sync"
	"time"
)

// CompareNames compara cada nome de autor com cada nome de discente
// authors: slice de slices de strings representando os nomes dos autores
// students: slice de slices de strings representando os nomes dos discentes
// CompareNames compara cada nome de autor com cada nome de discente
func CompareNames(authors [][]string, students [][]string, wg *sync.WaitGroup, progress chan<- string) {
	defer wg.Done()

	startTime := time.Now()

	totalComparisons := len(authors) * len(students)
	completedComparisons := 0

	for _, authorGroup := range authors {
		for _, author := range authorGroup {
			for _, studentGroup := range students {
				for _, student := range studentGroup {
					similarity := JaccardSimilarity(author, student)
					if similarity > 0.7 {
						msg := fmt.Sprintf("Similar names: %s and %s\n", author, student)
						progress <- msg
					}

					completedComparisons++

					// Atualizar o progresso a cada 10% completado
					if completedComparisons%int(0.1*float64(totalComparisons)) == 0 {
						progress <- fmt.Sprintf("Progresso: %.0f%% concluído", float64(completedComparisons)/float64(totalComparisons)*100)
					}
				}
			}
		}
	}

	elapsedTime := time.Since(startTime)
	progress <- fmt.Sprintf("Tempo total de execução: %s", elapsedTime.String())
}

// JaccardSimilarity calcula a similaridade de Jaccard entre duas strings
func JaccardSimilarity(str1, str2 string) float64 {
	set1 := stringToSet(str1)
	set2 := stringToSet(str2)

	intersection := intersection(set1, set2)
	union := union(set1, set2)

	return float64(len(intersection)) / float64(len(union))
}

// Funções auxiliares
// Converte uma string em um conjunto de caracteres
func stringToSet(str string) map[rune]bool {
	set := make(map[rune]bool)
	for _, char := range str {
		if !isSpace(char) {
			set[char] = true
		}
	}
	return set
}

// Calcula a interseção entre dois conjuntos de caracteres
func intersection(set1, set2 map[rune]bool) map[rune]bool {
	intersection := make(map[rune]bool)
	for char := range set1 {
		if set2[char] {
			intersection[char] = true
		}
	}
	return intersection
}

// Calcula a união entre dois conjuntos de caracteres
func union(set1, set2 map[rune]bool) map[rune]bool {
	union := make(map[rune]bool)
	for char := range set1 {
		union[char] = true
	}
	for char := range set2 {
		union[char] = true
	}
	return union
}

// Verifica se o caractere é um espaço
func isSpace(char rune) bool {
	return char == ' ' || char == '\t' || char == '\n' || char == '\r'
}
