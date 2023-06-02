package domain

// ErroPersonalizado é uma struct que implementa a interface de erro
type ErroPersonalizado struct {
	Erro error
	Msg  string
}
