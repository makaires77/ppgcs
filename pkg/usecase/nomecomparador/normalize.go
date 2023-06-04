// normalize.go
package nomecomparador

import (
	"regexp"
	"strings"
	"unicode"

	"golang.org/x/text/runes"
	"golang.org/x/text/transform"
	"golang.org/x/text/unicode/norm"
)

// normalizeName normaliza o nome do discente
func normalizeName(name string) (string, error) {
	var err error
	t := transform.Chain(norm.NFD, runes.Remove(runes.In(unicode.Mn)), norm.NFC)
	name, _, err = transform.String(t, name)
	if err != nil {
		return "", err
	}

	// Remover pontuação
	reg, err := regexp.Compile("[^a-zA-Z0-9]+")
	if err != nil {
		return "", err
	}
	name = reg.ReplaceAllString(name, " ")

	// Transformar o nome em letras minúsculas
	name = strings.ToLower(name)

	// Mover o sobrenome para o início e adicionar iniciais
	names := strings.Fields(name)
	if len(names) > 1 {
		name = names[len(names)-1] + " " + strings.Join(names[:len(names)-1], " ")
	}

	return name, nil
}
