import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import ipywidgets as widgets
from IPython.display import display, clear_output

class UserIdentification:
    def __init__(self):
        self.student_name = ""
        self.institution_name = ""
        self.program_name = ""
        self.advisor_name = ""

    def collect_user_info(self):
        """ Coleta informações de identificação do usuário através de widgets. """
        self.institution_name = widgets.Text(description="  Instituição:")
        self.student_name = widgets.Text(description="Nome do Aluno:")
        self.program_name = widgets.Text(description="     Programa:")
        self.advisor_name = widgets.Text(description="   Orientador:")

        submit_button = widgets.Button(description="Enviar")
        submit_button.on_click(self._on_submit)

        display(self.student_name, self.institution_name, self.program_name, self.advisor_name, submit_button)

    def _on_submit(self, b):
        """ Ação executada quando o botão Enviar é pressionado. """
        clear_output()
        print("Informações Coletadas:")
        print("Instituição de Pesquisa:", self.institution_name.value)
        print("          Nome do Aluno:", self.student_name.value)
        print("       Nome do Programa:", self.program_name.value)
        print("     Nome do Orientador:", self.advisor_name.value)

class QuestionFormulation:
    def __init__(self):
        self._initialize_nltk_resources()
        self.subject = ""
        self.action = ""
        self.object = ""
        self.additional_context = ""
        self.question = ""

    def _initialize_nltk_resources(self):
        """ Baixa e instala os recursos do NLTK necessários, se ainda não estiverem instalados. """
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)

    def input_ideas(self):
        """ Coleta informações detalhadas para formular a pergunta de pesquisa usando widgets. """
        self.error_label = widgets.Label()  # Label para mostrar mensagens de erro

        style = {'description_width': 'initial'}
        full_width_layout = widgets.Layout(width='100%')  # Define a largura para 100%

        self.subject_widget = widgets.Text(description="Sujeito/Tema:", style=style, layout=full_width_layout)
        self.action_widget = widgets.Text(description="Relação/Efeito:", style=style, layout=full_width_layout)
        self.object_widget = widgets.Text(description="Objeto de Impacto:", style=style, layout=full_width_layout)
        self.additional_context_widget = widgets.Text(description="Contexto Adicional (opcional):", style=style, layout=full_width_layout)

        submit_button = widgets.Button(description="Gerar Pergunta")
        submit_button.on_click(self._on_submit)

        display(self.error_label, self.subject_widget, self.action_widget, self.object_widget, self.additional_context_widget, submit_button)

    def _on_submit(self, b):
        """ Ação executada quando o botão 'Gerar Pergunta' é pressionado. """
        clear_output()
        self.subject = self.subject_widget.value
        self.action = self.action_widget.value
        self.object = self.object_widget.value
        self.additional_context = self.additional_context_widget.value

        # Verificar se todos os campos obrigatórios estão preenchidos
        if not self.subject or not self.action or not self.object:
            self.error_label.value = "Preencher todos os campos abaixo:"
            self.input_ideas()  # Chama novamente a função para reiniciar o processo de coleta
            return

        self.question = self._synthesize_question()

        # Exibir a pergunta gerada em um widget
        question_output = widgets.Textarea(
            value=self.question,
            description='Pergunta de Pesquisa Gerada:',
            disabled=True,
            layout=widgets.Layout(width='100%')
        )
        display(question_output)

    def generate_question(self):
        """ Gera uma pergunta de pesquisa com base nas informações coletadas. """
        # Verificar se os campos estão preenchidos
        if hasattr(self, 'subject') and self.subject and \
           hasattr(self, 'action') and self.action and \
           hasattr(self, 'object') and self.object:
            # Chamar _synthesize_question diretamente com as informações coletadas
            self.question = self._synthesize_question()
            return self.question
        else:
            return "Preencha todos os campos do contexto (sujeito, ação, objeto) antes de gerar a pergunta."

    def _synthesize_question(self):
        """ Sintetiza uma pergunta de pesquisa a partir das informações coletadas. """
        # Usar diretamente os valores coletados pelos widgets
        basic_question = f"Qual é o {self.action} do {self.subject} sobre o {self.object}?"
        if self.additional_context:
            return f"{basic_question} Considerando {self.additional_context}."
        else:
            return basic_question
    
    def _process_ideas(self):
        """ Processar as ideias para extrair conceitos chave de interesse."""
        words = word_tokenize(self.ideas)
        lemmatizer = WordNetLemmatizer()
        filtered_words = [lemmatizer.lemmatize(word) for word in words if word not in stopwords.words('english') and word.isalpha()]
        return filtered_words

    def _identify_key_elements(self, concepts):
        """
        Identificar elementos-chave como sujeito, objeto e ação nos conceitos.
        Retorna um dicionário com estes elementos.
        """
        elements = {'subject': None, 'action': None, 'object': None}
        for i, word in enumerate(concepts):
            if word in ['effect', 'impact', 'role', 'influence']:
                elements['action'] = word
                if i > 0:
                    elements['subject'] = concepts[i - 1]
                if i + 1 < len(concepts):
                    elements['object'] = concepts[i + 1]
        return elements

    def evaluate_question(self, question):
        # Implementar a lógica para avaliar a pergunta com base nos critérios
        evaluation_results = {}
        # ...
        return evaluation_results

class QuestionCriteriaEvaluator:
    def __init__(self):
        self.criteria = {
            "Clareza": None,
            "Relevância": None,
            "Viabilidade": None,
            "Originalidade": None,
            "Importância": None
        }

    def evaluate_question(self, question):
        # Avaliar cada critério com uma pontuação de 1 a 5
        # A avaliação é por análises específicas ou inputs do usuário validando na literatura
        for criterion in self.criteria:
            print(f"Avalie o critério '{criterion}' de 1 (Insuficiente) a 5 (Muito Bom): ")
            self.criteria[criterion] = int(input())  # Simples entrada do usuário para exemplo
        return self.criteria

class InteractiveFeedback:
    def __init__(self, evaluator: QuestionCriteriaEvaluator):
        self.evaluator = evaluator
        self.modified_question = ""

    def collect_user_feedback(self, original_question):
        print(f"Pergunta original: {original_question}")
        evaluation = self.evaluator.evaluate_question(original_question)

        for criterion, score in evaluation.items():
            if score < 3:  # Limiar para feedback
                print(f"O critério '{criterion}' teve uma pontuação baixa ({score}). Como podemos melhorar?")
                suggestion = input("Sugestão: ")
                self.modified_question += suggestion + " "  # Simples concatenação para exemplo

        if not self.modified_question:
            self.modified_question = original_question

        return self.modified_question

class LiteratureDataIntegration:
    def __init__(self, apis):
        self.apis = apis  # Em computação: [SpringerNatureAPI(), IEEEXploreAPI(), COREAPI(), CrossRefAPI()]

    def search_literature(self, query):
        results = {}
        for api in self.apis:
            results[api.name] = api.search(query)
        return results

class SpringerNatureAPI:
    def __init__(self):
        self.name = "SpringerNature"
        # Outras configurações da API

    def search(self, query):
        # Implementar a lógica de busca
        search_results = {}
        # ...
        return search_results


