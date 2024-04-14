from bs4 import BeautifulSoup, NavigableString, Comment, Tag
import json

class HTMLParser:
    def __init__(self, html):
        self.soup = BeautifulSoup(html, 'lxml')
        self.ignore_classes = ["header", "sub_tit_form", "footer", "rodape-cv", "menucent", "menu-header", "menuPrincipal", "to-top-bar", "header-content max-width", "control-bar-wrapper", "megamenu"]
        self.ignore_elements = ['script', 'style']

        # Estrutura base do currículo
        # self.estrutura = {
        #     "Dados Gerais": {"Nome": "", "ID Lattes": "", "Última atualização": ""},
        #     "Formação Acadêmica/Titulação": [],
        #     "Formação Complementar": [],
        #     "Atuação Profissional": [],
        #     "Linhas de Pesquisa": [],
        #     "Áreas de Atuação": [],
        #     "Idiomas": [],
        #     "Produções": {"Bibliográfica": [], "Técnica": [], "Artística/Cultural": []},
        #     "Bancas": [],
        #     "Eventos": [],
        #     "Orientações": [],
        #     "Inovação": [],
        # }

        self.estrutura = {
            "Label": "Person",
            "id_lattes": "",
            "Dados gerais": {"Nome": "", "Lattes iD": "", "Última atualização": "", "Nome em citações bibliográficas": []},
            "Endereço": {"Endereço Profissional": []},
            "Formação acadêmica/titulação": [{"": []}],
            "Pós-doutorado": [{"": []}],
            "Formação complementar": [{"": []}],
            "Linhas de pesquisa": [],
            "Membro de corpo editorial": [],
            "Membro de comitê de assessoramento": [],
            "Revisor de periódico": [],
            "Revisor de projeto de fomento": [],
            "Atuação profissional": [{"": []}],
            "Linhas de pesquisa": [],
            "Áreas de atuação": [],
            "Membro de corpo editorial": [],
            "Revisor de periódico": [],
            "Área de atuação": [],
            "Idiomas": [],
            "Prêmios e títulos": [],
            "Produções": [
                {
                    "Produção Bibliográfica": [
                        {"Citações": []},
                        {"Artigos completos publicados em periódicos": []},
                        {"Livros publicados": []},
                        {"Capítulos de livros publicados": []},
                        {"Textos em jornais ou revistas (magazine)": []},
                        {"Trabalhos publicados em anais de congressos": []},
                        {"Resumos publicados em anais de congressos": []},
                        {"Resumos publicados em anais de congressos (artigos)": []},
                        {"Artigos aceitos para publicação": []},
                        {"Apresentações de trabalho": []},
                        {"Partitura musical": []},
                        {"Tradução": []},
                        {"Prefácio, pósfacio": []},
                        {"Outras produções bibliográficas": []}
                    ],
                    "Produção Técnica": [
                        {"Assessoria e consultoria": []},
                        {"Programas de computador sem registro": []},
                        {"Produtos tecnológicos": []},
                        {"Processos e técnicas": []},
                        {"Trabalhos técnicos": []},
                        {"Cartas, mapas ou similares": []},
                        {"Curso de curta duração ministrado": []},
                        {"Desenvolvimento de material didático ou instrucional": []},
                        {"Editoração": []},
                        {"Manutenção de obra artística": []},
                        {"Maquete": []},
                        {"Entrevistas, mesas redondas, programas e comentários na mídia": []},
                        {"Relatório de pesquisa": []},
                        {"Redes sociais, websites e blogs": []},
                        {"Outras produções técnicas": []}
                    ],
                    "Produção Artística/Cultural": [
                        {"Artes cênicas": []},
                        {"Música": []},
                        {"Artes visuais": []},
                        {"Outras produções artísticas/culturais": []}
                    ]
                }
            ],
            "Patentes e registros": [
                {"Patente": []},
                {"Programa de computador registrado": []},
                {"Cultivar protegida": []},
                {"Cultivar registrada": []},
                {"Desenho industrial registrado": []},
                {"Marca registrada": []},
                {"Topografia de circuito integrado registrada": []},
                {"Programa de computador sem registro": []},
                {"Produtos": []},
                {"Processos ou técnicas": []},
                {"Projetos de pesquisa": []},
                {"Projeto de desenvolvimento tecnológico": []},
                {"Projeto de extensão": []},
                {"Outros projetos": []}
            ],
            "Inovação": [
                {"Patente": []},
                {"Programa de computador registrado": []},
                {"Cultivar protegida": []},
                {"Cultivar registrada": []},
                {"Desenho industrial registrado": []},
                {"Marca registrada": []},
                {"Topografia de circuito integrado registrada": []},
                {"Programa de computador sem registro": []},
                {"Produtos": []},
                {"Processos ou técnicas": []},
                {"Projetos de pesquisa": []},
                {"Projeto de desenvolvimento tecnológico": []},
                {"Projeto de extensão": []},
                {"Outros projetos": []}
            ],
            "Educação e Popularização de C & T": [
                {"Livros e capítulos": []},
                {"Organização de eventos, congressos, exposições e feiras": []}
            ],
            "Bancas": [
                {
                    "Participação em bancas": [
                        {
                            "Participação em bancas de trabalhos de conclusão": [
                                {"Mestrado": []},
                                {"Teses de doutorado": []},
                                {"Trabalhos de conclusão de curso de graduação": []},
                                {"Outros tipos": []}
                            ],
                            "Participação em bancas de comissões julgadoras": [
                                {"Concurso público": []},
                                {"Livre docência": []},
                                {"Avaliação de cursos": []},
                                {"Outras participações": []}
                            ]
                        }
                    ]
                }
            ],
            "Eventos": [
                {"Participação em eventos, congressos, exposições e feiras": []},
                {"Organização de eventos, congressos, exposições e feiras": []}
            ],
            "Orientações": [
                {
                    "Orientações e supervisões concluídas": [
                        {"Dissertação de mestrado": []},
                        {"Tese de doutorado": []},
                        {"Supervisão de pós-doutorado": []},
                        {"Monografia de conclusão de curso de aperfeiçoamento/especialização": []},
                        {"Trabalho de conclusão de curso de graduação": []},
                        {"Iniciação científica": []},
                        {"Orientações de outra natureza": []}
                    ],
                    "Orientações e supervisões em andamento": [
                        {"Dissertação de mestrado": []},
                        {"Tese de doutorado": []},
                        {"Supervisão de pós-doutorado": []},
                        {"Monografia de conclusão de curso de aperfeiçoamento/especialização": []},
                        {"Trabalho de conclusão de curso de graduação": []},
                        {"Iniciação científica": []},
                        {"Orientações de outra natureza": []}
                    ]
                }
            ],
            "Atividades de Extensão Universitária": [
                {"Projetos de extensão": []},
                {"Cursos e oficinas ministradas": []}
            ],
            "Outras informações relevantes": [],
            "Projetos": [
                {
                    "Projetos de pesquisa em andamento e concluídos": [],
                    "Projetos de desenvolvimento": []
                }
            ]
        }

    def should_ignore(self, tag):
        if isinstance(tag, Tag):
            if any(cls in self.ignore_classes for cls in tag.get('class', [])) or tag.name in self.ignore_elements:
                return True
        return False

    def extract_key_value_pairs(self):
        def recursive_extract(tag):
            if isinstance(tag, Comment) or self.should_ignore(tag):
                return None
            if isinstance(tag, NavigableString):
                return tag.strip()
            content_list = []
            for child in tag.children:
                extracted = recursive_extract(child)
                if extracted:
                    content_list.append(extracted)
            if len(content_list) == 1:
                return content_list[0]
            return content_list if content_list else None
        return recursive_extract(self.soup.body)

    def process_identification(self, data):
        # Exemplo de processamento para 'Identificação'
        for item in data:
            if isinstance(item, list):
                if item[0] == "Nome":
                    self.estrutura["Dados Gerais"]["Nome"] = item[1]
                elif item[0] == "ID Lattes:":
                    self.estrutura["Dados Gerais"]["ID Lattes"] = item[1]
            elif isinstance(item, str) and "Última atualização do currículo" in item:
                self.estrutura["Dados Gerais"]["Última atualização"] = item

    def process_education(self, data):
        education_list = []
        for item in data:
            if isinstance(item, list) and len(item) > 1:
                degree_info = {
                    "Ano": item[0],
                    "Descrição": item[1]
                }
                education_list.append(degree_info)
        self.estrutura["Formação Acadêmica/Titulação"] = education_list

    def process_professional_activities(self, data):
        professional_activities_list = []
        for item in data:
            if isinstance(item, list) and len(item) > 1:
                activity_info = {
                    "Periodo": item[0],
                    "Descrição": item[1]
                }
                professional_activities_list.append(activity_info)
        self.estrutura["Atuação Profissional"] = professional_activities_list

    def process_research_lines(self, data):
        research_lines = []
        for item in data:
            if isinstance(item, list) and len(item) > 1:
                line = {
                    "Número": item[0],
                    "Tema": item[1]
                }
                research_lines.append(line)
        self.estrutura["Linhas de Pesquisa"] = research_lines

    def process_languages(self, data):
        languages = []
        i = 0
        while i < len(data):
            language_info = {"Idioma": data[i]}
            i += 1  # Move para o próximo elemento, que deveria ser a proficiência

            # Verifica se existe um próximo elemento na lista antes de tentar acessá-lo
            if i < len(data):
                language_info["Proficiência"] = data[i]
            else:
                language_info["Proficiência"] = "Informação não fornecida"
            i += 1  # Move para o próximo par de idioma e proficiência

            languages.append(language_info)
        
        self.estrutura["Idiomas"] = languages

    def process_productions(self, data):
        # Verifica se data é uma lista de listas e itera sobre ela
        if isinstance(data, list) and all(isinstance(item, list) and len(item) >= 2 for item in data):
            for item in data:
                category = item[0]  # Primeiro elemento é a categoria
                items = item[1:]    # Os demais elementos são os itens da categoria

                # Processa diferentes categorias de produção
                if category == "Produção bibliográfica":
                    self.estrutura["Produções"]["Bibliográfica"].extend(items)
                elif category == "Produção técnica":
                    self.estrutura["Produções"]["Técnica"].extend(items)
                elif category == "Produção artística/cultural":
                    self.estrutura["Produções"]["Artística/Cultural"].extend(items)
                # Adicione condições para outras categorias conforme necessário
        else:
            print("Formato de dados de produção não suportado ou inválido.")

    def process_committee_participation(self, data):
        committees = []
        for item in data:
            if isinstance(item, list) and len(item) > 1:
                committee_info = {
                    "Tipo": item[0],
                    "Descrição": item[1]
                }
                committees.append(committee_info)
        self.estrutura["Bancas"] = committees

    def allocate_data_to_structure(self, extracted_data):
        for section in extracted_data:
            section_title = section[0]
            section_data = section[1:]

            if section_title == "Identificação":
                self.process_identification(section_data)
            elif section_title == "Formação acadêmica/titulação":
                self.process_education(section_data)
            elif section_title == "Atuação Profissional":
                self.process_professional_activities(section_data)
            elif section_title == "Linhas de pesquisa":
                self.process_research_lines(section_data)
            elif section_title == "Idiomas":
                self.process_languages(section_data)
            elif section_title == "Produções":
                self.process_productions(section_data)
            elif section_title == "Bancas":
                self.process_committee_participation(section_data)
            # Adicione chamadas para processar outras seções conforme necessário

    def to_json_with_all_post_processes(self):
        data = self.extract_key_value_pairs()
        self.allocate_data_to_structure(data)
        return json.dumps(self.estrutura, ensure_ascii=False, indent=4)
