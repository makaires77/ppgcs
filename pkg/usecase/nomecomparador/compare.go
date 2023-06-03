package nomecomparador

import (
	"strings"
	"unicode"
)

// JaccardSimilarity calcula a similaridade de Jaccard entre duas strings
func JaccardSimilarity(str1, str2 string) float64 {
	set1 := stringToSet(str1)
	set2 := stringToSet(str2)

	intersection := intersection(set1, set2)
	union := union(set1, set2)

	return float64(len(intersection)) / float64(len(union))
}

// LevenshteinDistance calcula a distância de Levenshtein entre duas strings
func LevenshteinDistance(str1, str2 string) int {
	len1 := len(str1)
	len2 := len(str2)

	// Criar uma matriz para armazenar as distâncias
	matrix := make([][]int, len1+1)
	for i := range matrix {
		matrix[i] = make([]int, len2+1)
	}

	// Inicializar a primeira linha e a primeira coluna da matriz
	for i := 0; i <= len1; i++ {
		matrix[i][0] = i
	}
	for j := 0; j <= len2; j++ {
		matrix[0][j] = j
	}

	// Preencher a matriz com as distâncias
	for i := 1; i <= len1; i++ {
		for j := 1; j <= len2; j++ {
			cost := 1
			if str1[i-1] == str2[j-1] {
				cost = 0
			}

			matrix[i][j] = min(matrix[i-1][j]+1, matrix[i][j-1]+1, matrix[i-1][j-1]+cost)
		}
	}

	return matrix[len1][len2]
}

// Soundex converte uma string em seu código Soundex
func Soundex(str string) string {
	if len(str) == 0 {
		return ""
	}

	str = strings.ToUpper(str)
	soundex := string(str[0])
	prevCode := getCode(str[0])

	for i := 1; i < len(str) && len(soundex) < 4; i++ {
		code := getCode(str[i])
		if code != prevCode && code != 0 {
			soundex += string(code)
		}
		prevCode = code
	}

	// Preencher com zeros se necessário
	for len(soundex) < 4 {
		soundex += "0"
	}

	return soundex
}

// Funções auxiliares

// Converte uma string em um conjunto de caracteres
func stringToSet(str string) map[rune]bool {
	set := make(map[rune]bool)
	for _, char := range str {
		if !unicode.IsSpace(char) {
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

// Retorna o código Soundex de um caractere
func getCode(char byte) byte {
	switch char {
	case 'B', 'F', 'P', 'V':
		return '1'
	case 'C', 'G', 'J', 'K', 'Q', 'S', 'X', 'Z':
		return '2'
	case 'D', 'T':
		return '3'
	case 'L':
		return '4'
	case 'M', 'N':
		return '5'
	case 'R':
		return '6'
	default:
		return 0
	}
}

// Retorna o mínimo entre três inteiros
func min(a, b, c int) int {
	if a < b && a < c {
		return a
	}
	if b < a && b < c {
		return b
	}
	return c
}
