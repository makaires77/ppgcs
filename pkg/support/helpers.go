package support

import (
	"regexp"
	"strings"
	"sync"
	"unicode"

	"golang.org/x/text/runes"
	"golang.org/x/text/transform"
	"golang.org/x/text/unicode/norm"
)

// Criar um Mutex para proteger a variável totalSimilarities
var mu sync.Mutex

func normalizeName(name string) string {
	name = removePrepositions(name)
	name = normalizeString(name)

	// Verificar se o nome já contém vírgula
	if strings.Contains(name, ",") {
		// Se o nome já contém vírgula, realizar apenas a remoção de acentuação e preposições
		name = convertToInitials(name)
	} else {
		// Se o nome não contém vírgula, trazer o sobrenome para o início e adicionar iniciais
		name = bringLastNameToFront(name)
		name = convertToInitials(name)
	}

	return name
}

func bringLastNameToFront(name string) string {
	names := strings.Fields(name)
	if len(names) > 1 {
		lastName := names[len(names)-1]
		initials := ""
		for i, n := range names[:len(names)-1] {
			if i == 0 {
				// Manter o primeiro nome completo
				initials += n + " "
			} else {
				// Converter os demais nomes em iniciais
				initials += string(n[0]) + ". "
			}
		}
		name = lastName + ", " + strings.TrimSpace(initials)
	}
	return name
}

func removePrepositions(name string) string {
	prepositions := []string{"de", "da", "do", "das", "dos"}
	for _, prep := range prepositions {
		name = strings.ReplaceAll(name, " "+prep+" ", " ")
	}
	return name
}

func convertToInitials(name string) string {
	names := strings.Fields(name)
	initials := ""
	for i, n := range names {
		// Verificar se é um sobrenome
		if i == 0 || strings.Contains(n, "-") {
			initials += n + " "
		} else {
			initials += string(n[0]) + ". "
		}
	}
	return strings.TrimSpace(initials)
}

func normalizeString(s string) string {
	t := transform.Chain(norm.NFD, runes.Remove(runes.In(unicode.Mn)), norm.NFC)
	name, _, _ := transform.String(t, s)
	name = removeAccentRunes(name)
	name = strings.ToUpper(name)
	return name
}

func removeAccentRunes(s string) string {
	reg := regexp.MustCompile("[ÀÁÂÃÄÅàáâãäåÈÉÊËèéêëÌÍÎÏìíîïÒÓÔÕÖØòóôõöøÙÚÛÜùúûüÇç]")
	return reg.ReplaceAllString(s, "")
}
