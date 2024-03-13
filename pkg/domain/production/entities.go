// Em pkg/domain/production/entities.go
package production

import "time"

// Production é a interface que define os métodos comuns a todas as produções.
type Production interface {
	GetTitle() string
	GetDate() time.Time
	// Outros métodos comuns a todas as produções...
}

// Publication é uma estrutura que representa uma publicação específica.
type Publication struct {
	Title   string
	Date    time.Time
	Authors []string
	// Outros campos específicos de uma publicação...
}

// GetTitle retorna o título da publicação.
func (p *Publication) GetTitle() string {
	return p.Title
}

// GetDate retorna a data da publicação.
func (p *Publication) GetDate() time.Time {
	return p.Date
}

// Outros métodos específicos de uma publicação podem ser implementados aqui.
// Usamos a interface Production para representar todos os tipos de produções e a estrutura Publication para representar publicações específicas. Para outros tipos de produções, pode-se criar estruturas adicionais que também implementam a interface Production, cada uma com seus próprios campos e métodos específicos. Essa abordagem permite que trabalhar com produções de diferentes tipos de maneira uniforme, aproveitando os métodos definidos na interface Production, ao mesmo tempo em que pode acessar os campos e métodos específicos de cada tipo de produção individualmente.
