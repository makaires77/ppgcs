import bs4, re, json
from collections import defaultdict
from bs4 import BeautifulSoup, NavigableString, Tag

class HTMLParser:
    def __init__(self, html):
        self.soup = BeautifulSoup(html, 'html.parser')
        self.ignore_classes = ["header", "sub_tit_form", "footer", "rodape-cv", "menucent", "menu-header", "menuPrincipal", "to-top-bar", "header-content max-width", "control-bar-wrapper", "megamenu","layout-cell-6"]
        self.ignore_elements = ['script', 'style', 'head']
        self.visited_structures = defaultdict(int)
        self.estrutura = {}

    def should_ignore(self, node):
        if isinstance(node, bs4.element.Tag):
            if node.name in self.ignore_elements:
                return True
            if any(cls in self.ignore_classes for cls in node.get('class', [])):
                return True
        return False

    # Revelar a estrutura hierárquica do HTML
    def explore_structure(self):
        self.print_node_hierarchy(self.soup)

    def print_node_hierarchy(self, node, level=0, parent_counts=defaultdict(int)):
        if self.should_ignore(node):
            return

        prefix = "  " * level
        if isinstance(node, bs4.element.Tag):
            # Verificar se o nó é parte de uma lista repetitiva
            if node.name == 'b' and node.text.isdigit():
                num = int(node.text)
                if num == 1 or (num - 1) == parent_counts[node.parent.name]:
                    print(f"{prefix}[Estrutura repetida começa aqui]")
                parent_counts[node.parent.name] = num
                
                if num > 1:
                    return

            print(f"{prefix}<{node.name} class='{node.get('class', '')}'>")
            for child in node.children:
                self.print_node_hierarchy(child, level + 1, parent_counts)
        elif isinstance(node, bs4.element.NavigableString) and node.strip():
            print(f"{prefix}{node.strip()}")

    def find_path_to_text(self, node, text, path=None):
        if path is None:
            path = []

        if self.should_ignore(node):
            return None

        if node.string and text.lower() in node.string.lower():
            return path

        if isinstance(node, bs4.element.Tag):
            for i, child in enumerate(node.children):
                child_path = path + [(node.name, node.get('class'), i)]
                found_path = self.find_path_to_text(child, text, child_path)
                if found_path:
                    return found_path

        return None

    def find_element_by_path(self, path):
        """
        Encontra o elemento no BeautifulSoup object baseado no caminho fornecido.
        O caminho é uma lista de tuplas (tag, classe, índice).
        """
        current_element = self.soup
        for tag, classe, index in path:
            try:
                if classe:  # Se uma classe foi especificada
                    current_element = current_element.find_all(tag, class_=classe)[index]
                else:  # Se nenhuma classe foi especificada
                    current_element = current_element.find_all(tag)[index]
            except IndexError:
                return None  # Retorna None se o caminho não levar a um elemento válido
        return current_element

    def extract_data_from_path(self, path):
        """
        Extrai dados de um caminho especificado.
        O caminho é uma lista de direções para navegar na árvore HTML.
        """
        # Inicia no elemento raiz (soup)
        current_element = self.soup

        # Navega pelo caminho
        for tag, classe, index in path:
            # Tenta encontrar o próximo elemento no caminho
            if classe:  # Se classe for especificada
                current_element = current_element.find_all(tag, class_=classe)
            else:  # Se não, apenas pela tag
                current_element = current_element.find_all(tag)

            # Tenta acessar o elemento pelo índice, se falhar retorna None
            try:
                current_element = current_element[index]
            except IndexError:
                return None

        # Retorna o elemento encontrado
        return current_element

    def explore_structure_for_text(self, target_text):
        """
        Encontra o caminho hierárquico até o texto desejado, considerando tags, classes e índices.
        """
        def find_path(element, path=[]):
            if element.name in self.ignore_elements:
                return False
            if isinstance(element, Tag):
                for class_ in self.ignore_classes:
                    if class_ in element.get('class', []):
                        return False
                for child in element.children:
                    new_path = path + [(element.name, ' '.join(element.get('class', [])), element.index(child))]
                    if child.string and target_text in child.string:
                        return new_path
                    else:
                        found_path = find_path(child, new_path)
                        if found_path:
                            return found_path
            return False

        found_path = find_path(self.soup)
        if found_path:
            # Transforma o caminho em uma lista de dicionários.
            path_dicts = [{'Tag': tag, 'Classes': classes, 'Index': index} for tag, classes, index in found_path]
            return path_dicts
        else:
            return f"Caminho para '{target_text}' não encontrado."

    ## Processamentos da extração de dados
    # Identificação OK!!            
    def process_identification(self):
        nome = self.soup.find(class_="nome").text.strip()
        id_lattes = self.soup.find("span", style=lambda value: value and "color: #326C99" in value).text.strip()
        
        ultima_atualizacao_element = self.soup.find(lambda tag: tag.name == "li" and "Última atualização do currículo em" in tag.text)
        if ultima_atualizacao_element:
            ultima_atualizacao = ultima_atualizacao_element.text.split("em")[1].strip()
        else:
            ultima_atualizacao = "Não encontrado"

        self.estrutura["Identificação"] = {
            "Nome": nome,
            "ID Lattes": id_lattes,
            "Última atualização": ultima_atualizacao
        }

    # Idiomas OK!!!
    def process_idiomas(self):
        idiomas = []
        idiomas_header = self.soup.find("h1", text=lambda text: text and "Idiomas" in text)
        
        if idiomas_header:
            idiomas_container = idiomas_header.find_next("div", class_="data-cell")
            
            if not idiomas_container:
                # Se não encontrou usando a classe "data-cell", tenta buscar o próximo container de maneira mais genérica
                idiomas_container = idiomas_header.find_next_sibling()

            if idiomas_container:
                idioma_divs = idiomas_container.find_all("div", recursive=False)
                for idioma_div in idioma_divs:
                    idioma = idioma_div.find("div", class_="layout-cell-pad-5 text-align-right")
                    proficiencia = idioma_div.find_next("div", class_="layout-cell layout-cell-9")
                    
                    if idioma and proficiencia:
                        idioma_text = idioma.text.strip()
                        proficiencia_text = proficiencia.text.strip()
                        idiomas.append({"Idioma": idioma_text, "Proficiência": proficiencia_text})
                    else:
                        continue
            else:
                print("Container de idiomas não encontrado")
        else:
            print("Seção de idiomas não encontrada")

        self.estrutura["Idiomas"] = idiomas

    # Formação acadêmica e complementar OK!
    def process_formacao(self):
        formacao_academica = []
        formacao_posdoc = []
        formacao_complementar = []
        
        # Encontrar todas as seções 'title-wrapper' que contêm os títulos 'Formação acadêmica/titulação' e 'Formação Complementar'
        secoes = self.soup.find_all('div', class_='title-wrapper')
        
        for secao in secoes:
            titulo_h1 = secao.find('h1')
            if titulo_h1:
                titulo = titulo_h1.get_text(strip=True)
                data_cell = secao.find('div', class_='layout-cell layout-cell-12 data-cell')
                if data_cell:
                    # Processar cada item de formação dentro da data_cell
                    # A estrutura é sempre a mesma: ano à direita, descrição à esquerda
                    anos_divs = data_cell.find_all('div', class_='layout-cell layout-cell-3 text-align-right')
                    descricoes_divs = data_cell.find_all('div', class_='layout-cell layout-cell-9')
                    
                    for ano_div, descricao_div in zip(anos_divs, descricoes_divs):
                        ano = ano_div.get_text(strip=True)
                        descricao = descricao_div.get_text(separator=' ', strip=True).replace(' .', '.')
                        formacao = {"Ano": ano, "Descrição": descricao}
                        
                        if 'Formação acadêmica/titulação' in titulo:
                            formacao_academica.append(formacao)
                        elif 'Pós-doutorado' in titulo:
                            formacao_posdoc.append(formacao)
                        elif 'Formação Complementar' in titulo:
                            formacao_complementar.append(formacao)

        # Armazenar ou retornar os dados de formação
        self.estrutura["Formação"] = {
            "Acadêmica": formacao_academica,
            "Pos-Doc": formacao_posdoc,
            "Complementar": formacao_complementar
        }
        
        # Retorna o dicionário de formação se necessário
        return self.estrutura["Formação"]

    ## Linhas de Pesquisa OK!
    def process_linhas_pesquisa(self):
        linhas_pesquisa = []
        # Encontrar a seção específica de linhas de pesquisa
        secoes = self.soup.find_all('div', class_='title-wrapper')
        for secao in secoes:
            titulo_h1 = secao.find('h1')
            if titulo_h1 and 'Linhas de pesquisa' in titulo_h1.get_text(strip=True):
                # Encontrar todos os blocos de dados dentro da seção
                data_cell = secao.find('div', class_='layout-cell-12')
                if data_cell:
                    # Encontrar todos os elementos de título e descrição dentro da data_cell
                    elements = data_cell.find_all(recursive=False)
                    detalhes = ""
                    for i, element in enumerate(elements):
                        if element.name == 'div' and 'layout-cell-9' in element.get('class', []):
                            descricao = element.get_text(strip=True)
                            if 'Objetivo:' in descricao and i+1 < len(elements):
                                descricao = element.find_previous_sibling('div', class_='layout-cell-9').get_text(strip=True)
                                detalhes += element.get_text(separator=' ', strip=True) + ' '
                            elif element.name == 'div' and 'text-align-right' in element.get('class', []):
                                continue

                            linhas_pesquisa.append({
                                "Descrição": descricao,
                                "Detalhes": detalhes.strip()
                            })
        
        self.estrutura["Linhas de Pesquisa"] = linhas_pesquisa
        return linhas_pesquisa

    ## Atuação Profissional OK!
    def process_atuacao_profissional(self):
        atuacoes_profissionais = []
        secoes = self.soup.find_all('div', class_='title-wrapper')
        for secao in secoes:
            titulo_h1 = secao.find('h1')
            if titulo_h1 and 'Atuação Profissional' in titulo_h1.get_text(strip=True):
                data_cell = secao.find('div', class_='layout-cell layout-cell-12 data-cell')
                if data_cell:
                    # Iniciamos a coleta de dados do primeiro bloco após inst_back
                    elements = data_cell.find_all(recursive=False)
                    current_instituicao = None
                    current_block = []

                    # Iteramos sobre os elementos para capturar informações até a próxima inst_back
                    for element in elements:
                        if element.name == 'div' and 'inst_back' in element.get('class', []):
                            if current_instituicao:  # Se já havia uma instituição, processamos o bloco acumulado
                                self.extract_atuacao_from_block(current_block, atuacoes_profissionais, current_instituicao)
                                current_block = []  # Reiniciamos o bloco para a próxima instituição
                            current_instituicao = element.get_text(strip=True)
                        elif current_instituicao:  # Estamos dentro do bloco de uma instituição
                            current_block.append(element)

                    # Não esqueça de processar o último bloco
                    if current_instituicao and current_block:
                        self.extract_atuacao_from_block(current_block, atuacoes_profissionais, current_instituicao)

        self.estrutura["Atuação Profissional"] = atuacoes_profissionais
        return self.estrutura["Atuação Profissional"]

    def extract_atuacao_from_block(self, block, atuacoes_profissionais, instituicao_nome):
        ano_pattern = re.compile(r'(\d{2}/)?\d{4}\s*-\s*(\d{2}/)?(?:\d{4}|Atual)')
        # Removemos os padrões que não serão usados diretamente na identificação de elementos

        ano = None
        descricao = None
        outras_informacoes = []

        for element in block:
            # Captura o ano e descrição
            if element.name == 'div' and 'text-align-right' in element.get('class', []):
                if ano_pattern.search(element.get_text(strip=True)):
                    if ano:  # Se um ano já foi capturado, então terminamos de processar o bloco anterior
                        atuacao = {
                            "Instituição": instituicao_nome,
                            "Ano": ano,
                            "Descrição": descricao,
                            "Outras informações": ' '.join(outras_informacoes)
                        }
                        atuacoes_profissionais.append(atuacao)
                        outras_informacoes = []  # Reiniciamos a lista para o próximo bloco
                    ano = element.get_text(strip=True)
                    descricao = element.find_next('div', class_='layout-cell-9').get_text(separator=' ', strip=True) if descricao else ""
            elif element.name == 'div' and 'layout-cell-9' in element.get('class', []):
                # Acumula todas as informações das divs 'layout-cell-9' dentro do mesmo bloco
                outras_infos = element.get_text(separator=' ', strip=True)
                if outras_infos:  # Verifica se há texto dentro do elemento
                    outras_informacoes.append(outras_infos)

        # Verifica se ainda existe um bloco a ser adicionado após o loop
        if ano:
            atuacao = {
                "Instituição": instituicao_nome,
                "Ano": ano,
                "Descrição": descricao,
                "Outras informações": ' '.join(outras_informacoes)
            }
            atuacoes_profissionais.append(atuacao)

        return atuacoes_profissionais

    def process_producao_bibliografica(self):
        # Inicializa a lista de produções bibliográficas
        self.estrutura["ProducaoBibliografica"] = {
            "Artigos completos publicados em periódicos": [],
            "Livros e capítulos": [],
            "Trabalhos completos publicados em anais de congressos": [],
            # Adicione mais categorias conforme necessário
        }

        # Mapeia os identificadores das seções para as categorias de produção bibliográfica
        secoes = {
            "ArtigosCompletos": "Artigos completos publicados em periódicos",
            "LivrosCapitulos": "Livros e capítulos",
            "TrabalhosPublicadosAnaisCongresso": "Trabalhos completos publicados em anais de congressos",
        }

        # Percorre cada seção de interesse no documento HTML
        for secao_id, categoria in secoes.items():
            secao_inicio = self.soup.find("a", {"name": secao_id})
            if not secao_inicio:
                continue

            # Encontra todos os itens dentro da seção até a próxima seção
            proxima_secao = secao_inicio.find_next_sibling("a", href=True)
            itens_secao = []
            atual = secao_inicio.find_next_sibling("div", class_="layout-cell layout-cell-11")
            while atual and atual != proxima_secao:
                if atual.text.strip():
                    itens_secao.append(atual.text.strip())
                atual = atual.find_next_sibling("div", class_="layout-cell layout-cell-11")

            # Adiciona os itens encontrados à categoria correspondente
            self.estrutura["ProducaoBibliografica"][categoria].extend(itens_secao)

    def process_producao_bibliografica(self):
        producoes = []
        secoes = self.soup.find_all('div', class_='title-wrapper')
        for secao in secoes:
            titulo_h1 = secao.find('h1')
            if titulo_h1 and 'Produções' in titulo_h1.get_text(strip=True):
                data_cell = secao.find('div', class_='layout-cell layout-cell-12 data-cell')
                div_artigos = data_cell.find_all("div", id="artigos-completos")
                for div in div_artigos:
                    if div:
                        # Iniciamos a coleta de dados do primeiro bloco após inst_back
                        articles = div.find_all("div", class_="artigo-completo", recursive=False)
                        # print(f'{len(articles)} divs de artigos')
                        current_instituicao = None
                        current_block = []

                        # Iteramos sobre os elementos para capturar informações até a próxima inst_back
                        for element in articles:
                            if element.name == 'div' and 'inst_back' in element.get('class', []):
                                if current_instituicao:  # Se já havia uma produção, processamos o bloco acumulado
                                    self.extract_producao_from_block(current_block, producoes, current_instituicao)
                                    current_block = []  # Reiniciamos o bloco para a próxima instituição
                                current_instituicao = element.get_text(strip=True)
                            elif current_instituicao:  # Estamos dentro do bloco de uma instituição
                                current_block.append(element)

                        # Não esqueça de processar o último bloco
                        if current_instituicao and current_block:
                            self.extract_producao_from_block(current_block, producoes, current_instituicao)

        self.estrutura["ProducaoBibliografica"] = producoes
        return self.estrutura["ProducaoBibliografica"]

    def extract_producao_from_block(self, block, producoes, instituicao_nome):
        ano_pattern = re.compile(r'(\d{2}/)?\d{4}\s*-\s*(\d{2}/)?(?:\d{4}|Atual)')
        # Removemos os padrões que não serão usados diretamente na identificação de elementos

        ano = None
        descricao = None
        outras_informacoes = []

        for element in block:
            # Captura o ano e descrição
            if element.name == 'div' and 'text-align-right' in element.get('class', []):
                if ano_pattern.search(element.get_text(strip=True)):
                    if ano:  # Se um ano já foi capturado, então terminamos de processar o bloco anterior
                        atuacao = {
                            "Instituição": instituicao_nome,
                            "Ano": ano,
                            "Descrição": descricao,
                            "Outras informações": ' '.join(outras_informacoes)
                        }
                        producoes.append(atuacao)
                        outras_informacoes = []  # Reiniciamos a lista para o próximo bloco
                    ano = element.get_text(strip=True)
                    descricao = element.find_next('div', class_='layout-cell-9').get_text(separator=' ', strip=True) if descricao else ""
            elif element.name == 'div' and 'layout-cell-9' in element.get('class', []):
                # Acumula todas as informações das divs 'layout-cell-9' dentro do mesmo bloco
                outras_infos = element.get_text(separator=' ', strip=True)
                if outras_infos:  # Verifica se há texto dentro do elemento
                    outras_informacoes.append(outras_infos)

        # Verifica se ainda existe um bloco a ser adicionado após o loop
        if ano:
            atuacao = {
                "Instituição": instituicao_nome,
                "Ano": ano,
                "Descrição": descricao,
                "Outras informações": ' '.join(outras_informacoes)
            }
            producoes.append(atuacao)

        return producoes

    def extrair_texto_sup(element):
        # Busca por todos os elementos <sup> dentro do elemento fornecido
        sup_elements = element.find_all('sup')
        # Lista para armazenar os textos extraídos
        textos_extras = []

        for sup in sup_elements:
            # Verifica se o elemento <sup> contém um elemento <img> com a classe 'ajaxJCR'
            if sup.find('img', class_='ajaxJCR'):
                # Extrai o valor do atributo 'original-title', se disponível
                texto = sup.find('img')['original-title'] if sup.find('img').has_attr('original-title') else None
                if texto:
                    textos_extras.append(texto)
        
        return textos_extras

    def extrair_dados_jcr(texto):
        # Regex para capturar o nome do periódico e o fator de impacto
        regex = r"(.+?)\s*\((\d{4}-\d{4})\)<br />\s*Fator de impacto \(JCR (\d{4})\):\s*(\d+\.\d+)"
        match = re.search(regex, texto)

        if match:
            periódico = f"{match.group(1)} ({match.group(2)})"
            fator_de_impacto = f"Fator de impacto (JCR {match.group(3)}): {match.group(4)}"
            return periódico, fator_de_impacto
        else:
            return None, None

    def extrair_dados_jcr(self, html_element):
        sup_tag = html_element.find('sup')
        img_tag = sup_tag.find('img')
        attributes_dict = {}
        # Extraia os atributos básicos
        # not_extract=['class','id','src']
        # attributes_dict = {key: value for key, value in img_tag.attrs.items() if key != 'original-title' and key not in not_extract}
        issn = sup_tag.find('img', class_='data-issn')
        # print(f'ISSN: {issn}')
        attributes_dict['data-issn'] = issn
        # original_title = sup_tag['original-title'].replace('&lt;br /&gt;', '')
        original_title = sup_tag.find('img', class_='original-title')
        # print(f'ISSN: {original_title}')
        parts = original_title.split(': ')
        periodico_info = parts[0].split('(')
        fator_impacto = parts[1]

        # Atualiza o dicionário com as informações processadas
        attributes_dict['periodico'] = f"{periodico_info[0].strip()} ({periodico_info[1].split('<br />')[0].strip(')')})"
        attributes_dict['fator_impacto'] = float(fator_impacto.split(' ')[0])
        attributes_dict['JCR'] = parts[0].split('(')[-1].split(')')[0]
        return attributes_dict
    
    def extract_year(self, soup):
        # Encontre o elemento <span> com a classe 'informacao-artigo' e data-tipo-ordenacao='ano'
        year_span = soup.findChild('span', {'class': 'informacao-artigo', 'data-tipo-ordenacao': 'ano'})
        
        # Recupera o texto do elemento, que deve ser o ano
        year = year_span.text if year_span else 'Ano não encontrado'

        return year

    def extract_first_author(self, soup):
        # Encontre o elemento <span> com a classe 'informacao-artigo' e data-tipo-ordenacao='autor'
        author_span = soup.findChild('span', {'class': 'informacao-artigo', 'data-tipo-ordenacao': 'autor'})
        
        # Recupera o texto do elemento, que deve ser o nome do primeiro autor
        first_author = author_span.text if author_span else 'Ano não encontrado'

        return first_author

    def extract_periodico(self, soup):
        # Encontre o elemento <span> com a classe 'informacao-artigo' e data-tipo-ordenacao='autor'
        img_tag = soup.findChild('sup')
        dados_periodico = img_tag.findChild('img', class_='original-title')
        if dados_periodico:
            parts = dados_periodico.split('(')
            print(parts)
        else:
            print(f"Não foi possível extrair dados do periódico de {soup}")
        # Recupera o texto do elemento, que deve ser o nome do primeiro autor
        periodico = dados_periodico.text if dados_periodico else None

        return periodico
    
    def extract_qualis(self, soup):
        # Extração de informações do Qualis a partir do elemento 'p'
        p_tag = soup.find('p')
        qualis_text = p_tag.get_text(strip=True) if p_tag else ''
        qualis_match = re.search(r'[ABC]\d', qualis_text)
        qualis = qualis_match.group(0) if qualis_match else 'Indisponível'

        # Extração de informações JCR a partir do elemento 'sup'
        sup_tag = soup.find('sup')
        jcr_info = sup_tag.find('img')['original-title'] if sup_tag and sup_tag.find('img') else ''
        jcr_parts = jcr_info.split('<br />') if jcr_info else []
        jcr = jcr_parts[-1].split(': ')[-1].strip() if len(jcr_parts) > 1 else 'Indisponível'

        # Compilando resultados
        results = {
            'Qualis': qualis,
            'JCR': jcr
        }
        return results

    def extract_info(self):
        soup = BeautifulSoup(self.html_element, 'html.parser')
        qualis_info = self.extract_qualis(soup)

        # Extrai os autores
        autores = soup.find_all('span', class_='informacao-artigo', data_tipo_ordenacao='autor')
        primeiro_autor = autores[0].text if autores else None
        # Considera todos os textos após o autor como parte da lista de autores até um elemento estrutural significativo (<a>, <b>, <sup>, etc.)
        autores_texto = self.html_element.split('autor">')[-1].split('</span>')[0] if autores else ''

        ano_tag = soup.find('span', {'class': 'informacao-artigo', 'data-tipo-ordenacao': 'ano'})
        ano = int(ano_tag.text) if ano_tag else 'Ano não disponível'

        # Extrai o título, periódico, e outras informações diretamente do texto
        texto_completo = soup.get_text(separator=' ', strip=True)
        
        # Assume que o título vem após os autores e termina antes de uma indicação de periódico ou volume
        titulo_match = re.search(r'; ([^;]+?)\.', texto_completo)
        titulo = titulo_match.group(1) if titulo_match else None

        # Periódico e detalhes como volume, página, etc., 
        periodico_match = re.search(r'(\. )([^.]+?),( v\. \d+, p\. \d+, \d+)', texto_completo)
        periodico = periodico_match.group(2) if periodico_match else None
        detalhes_periodico = periodico_match.group(3) if periodico_match else None

        # Extrai citações se disponível
        citacoes = soup.find('span', class_='numero-citacao')
        citacoes = int(citacoes.text) if citacoes else 0

        # Extrai ISSN
        issn = soup.find('img', class_='ajaxJCR')
        issn = issn['data-issn'] if issn else None

        # Qualis/CAPES pode ser extraído se existir um padrão identificável
        qualis_capes = "quadriênio 2017-2020"  # Neste exemplo, hardcoded, mas pode ser ajustado

        # Monta o dicionário de resultados
        resultado = {
            "dados_gerais": texto_completo,
            "primeiro_autor": primeiro_autor,
            "ano": ano,
            "autores": autores_texto,
            "titulo": titulo,
            "periodico": f"{periodico}{detalhes_periodico}",
            "data-issn": issn,
            "impacto": qualis_info.get('JCR'),
            "Qualis/CAPES": qualis_capes,
            "qualis": qualis_info.get('Qualis'),
            "citacoes": citacoes,
        }

        return resultado, json.dumps(resultado, ensure_ascii=False)
    
    def process_areas(self):
        self.estrutura["Áreas"]={}
        secoes = self.soup.find_all('div', class_='title-wrapper')
        for secao in secoes:
            titulo_h1 = secao.find('h1')
            if titulo_h1 and 'Áreas de atuação' in titulo_h1.get_text(strip=True):
                data_cell = secao.find('div', class_='layout-cell layout-cell-12 data-cell')
                ## Extrair cada área de atuação
                ocorrencias = {}
                # Encontrar todos os elementos irmãos seguintes de div_cita_artigos
                next_siblings = data_cell.findChildren("div")
                # print(len(next_siblings))
                # Listas para armazenar os divs encontrados
                divs_indices = []
                divs_ocorrencias = []

                # Iterar sobre os elementos irmãos
                for sibling in next_siblings:
                    # Verificar se o irmão tem a classe "cita-artigos"
                    if 'title-wrapper' in sibling.get('class', []):
                        # Encontramos o marcador para parar, sair do loop
                        break
                    # Verificar as outras classes e adicionar aos arrays correspondentes
                    if 'layout-cell layout-cell-3 text-align-right' in " ".join(sibling.get('class', [])):
                        divs_indices.append(sibling)
                    elif 'layout-cell layout-cell-9' in " ".join(sibling.get('class', [])):
                        divs_ocorrencias.append(sibling)
                
                if len(divs_indices) == len(divs_ocorrencias):
                    # Itera sobre o intervalo do comprimento de uma das listas
                    for i in range(len(divs_indices)):
                        # Usa o texto ou outro identificador único dos elementos como chave e valor
                        chave = divs_indices[i].get_text(strip=True)
                        valor = divs_ocorrencias[i].get_text(strip=True)

                        # Adiciona o par chave-valor ao dicionário
                        ocorrencias[chave] = valor

                self.estrutura["Áreas"] = ocorrencias

    def process_projetos_pesquisa(self):
        self.estrutura["ProjetosPesquisa"] = []
        secoes = self.soup.find_all('div', class_='title-wrapper')
        for secao in secoes:
            titulo_h1 = secao.find('h1')
            if titulo_h1 and 'Projetos de pesquisa' in titulo_h1.get_text(strip=True):
                data_cell = secao.find_next('div', class_='layout-cell layout-cell-12 data-cell')
                next_siblings = data_cell.find_all("div", recursive=False)

                chave = None
                titulo_projeto = None
                descricao = None
                estado = 0  # Estado inicial

                for sibling in next_siblings:
                    if 'title-wrapper' in sibling.get('class', []):
                        break  # Encontrou nova seção, finaliza o loop

                    classes = " ".join(sibling.get('class', []))
                    if estado == 0 and 'layout-cell layout-cell-3 text-align-right' in classes:
                        chave = sibling.get_text(strip=True)
                        estado = 1
                    elif estado == 1 and 'layout-cell layout-cell-9' in classes:
                        titulo_projeto = sibling.get_text(strip=True).replace('\t','').replace('\n',' ')
                        estado = 2  # Ignora a próxima "layout-cell-3"
                    elif estado == 2 and 'layout-cell layout-cell-3 text-align-right' in classes:
                        estado = 3  # Próximo estado para buscar a descrição
                    elif estado == 3 and 'layout-cell layout-cell-9' in classes:
                        descricao = sibling.get_text(strip=True).replace('\t','').replace('\n',' ')
                        # Prepara para adicionar ao dicionário e reinicia o ciclo
                        if chave and titulo_projeto:
                            projeto_pesquisa = {
                                "chave": chave,
                                "titulo_projeto": titulo_projeto,
                                "descricao": descricao
                            }
                            self.estrutura["ProjetosPesquisa"].append(projeto_pesquisa)
                            chave = titulo_projeto = descricao = None  # Reinicia para o próximo ciclo
                            estado = 0  # Volta ao estado inicial

    def process_projetos_extensao(self):
        self.estrutura["ProjetosExtensão"] = []
        secoes = self.soup.find_all('div', class_='title-wrapper')
        for secao in secoes:
            titulo_h1 = secao.find('h1')
            if titulo_h1 and 'Projetos de extensão' in titulo_h1.get_text(strip=True):
                data_cell = secao.find_next('div', class_='layout-cell layout-cell-12 data-cell')
                next_siblings = data_cell.find_all("div", recursive=False)

                chave = None
                titulo_projeto = None
                descricao = None
                estado = 0  # Estado inicial

                for sibling in next_siblings:
                    if 'title-wrapper' in sibling.get('class', []):
                        break  # Encontrou nova seção, finaliza o loop

                    classes = " ".join(sibling.get('class', []))
                    if estado == 0 and 'layout-cell layout-cell-3 text-align-right' in classes:
                        chave = sibling.get_text(strip=True)
                        estado = 1
                    elif estado == 1 and 'layout-cell layout-cell-9' in classes:
                        titulo_projeto = sibling.get_text(strip=True).replace('\t','').replace('\n',' ')
                        estado = 2  # Ignora a próxima "layout-cell-3"
                    elif estado == 2 and 'layout-cell layout-cell-3 text-align-right' in classes:
                        estado = 3  # Próximo estado para buscar a descrição
                    elif estado == 3 and 'layout-cell layout-cell-9' in classes:
                        descricao = sibling.get_text(strip=True).replace('\t','').replace('\n',' ')
                        # Prepara para adicionar ao dicionário e reinicia o ciclo
                        if chave and titulo_projeto:
                            projeto_pesquisa = {
                                "chave": chave,
                                "titulo_projeto": titulo_projeto,
                                "descricao": descricao
                            }
                            self.estrutura["ProjetosExtensão"].append(projeto_pesquisa)
                            chave = titulo_projeto = descricao = None  # Reinicia para o próximo ciclo
                            estado = 0  # Volta ao estado inicial

    def process_projetos_desenvolvimento(self):
        self.estrutura["ProjetosDesenvolvimento"] = []
        secoes = self.soup.find_all('div', class_='title-wrapper')
        for secao in secoes:
            titulo_h1 = secao.find('h1')
            if titulo_h1 and 'Projetos de desenvolvimento' in titulo_h1.get_text(strip=True):
                data_cell = secao.find_next('div', class_='layout-cell layout-cell-12 data-cell')
                next_siblings = data_cell.find_all("div", recursive=False)

                chave = None
                titulo_projeto = None
                descricao = None
                estado = 0  # Estado inicial

                for sibling in next_siblings:
                    if 'title-wrapper' in sibling.get('class', []):
                        break  # Encontrou nova seção, finaliza o loop

                    classes = " ".join(sibling.get('class', []))
                    if estado == 0 and 'layout-cell layout-cell-3 text-align-right' in classes:
                        chave = sibling.get_text(strip=True)
                        estado = 1
                    elif estado == 1 and 'layout-cell layout-cell-9' in classes:
                        titulo_projeto = sibling.get_text(strip=True).replace('\t','').replace('\n',' ')
                        estado = 2  # Ignora a próxima "layout-cell-3"
                    elif estado == 2 and 'layout-cell layout-cell-3 text-align-right' in classes:
                        estado = 3  # Próximo estado para buscar a descrição
                    elif estado == 3 and 'layout-cell layout-cell-9' in classes:
                        descricao = sibling.get_text(strip=True).replace('\t','').replace('\n',' ')
                        # Prepara para adicionar ao dicionário e reinicia o ciclo
                        if chave and titulo_projeto:
                            projeto_pesquisa = {
                                "chave": chave,
                                "titulo_projeto": titulo_projeto,
                                "descricao": descricao
                            }
                            self.estrutura["ProjetosDesenvolvimento"].append(projeto_pesquisa)
                            chave = titulo_projeto = descricao = None  # Reinicia para o próximo ciclo
                            estado = 0  # Volta ao estado inicial

    def process_projetos_outros(self):
        self.estrutura["ProjetosOutros"] = []
        secoes = self.soup.find_all('div', class_='title-wrapper')
        for secao in secoes:
            titulo_h1 = secao.find('h1')
            if titulo_h1 and 'Outros Projetos' in titulo_h1.get_text(strip=True):
                data_cell = secao.find_next('div', class_='layout-cell layout-cell-12 data-cell')
                next_siblings = data_cell.find_all("div", recursive=False)

                chave = None
                titulo_projeto = None
                descricao = None
                estado = 0  # Estado inicial

                for sibling in next_siblings:
                    if 'title-wrapper' in sibling.get('class', []):
                        break  # Encontrou nova seção, finaliza o loop

                    classes = " ".join(sibling.get('class', []))
                    if estado == 0 and 'layout-cell layout-cell-3 text-align-right' in classes:
                        chave = sibling.get_text(strip=True)
                        estado = 1
                    elif estado == 1 and 'layout-cell layout-cell-9' in classes:
                        titulo_projeto = sibling.get_text(strip=True).replace('\t','').replace('\n',' ')
                        estado = 2  # Ignora a próxima "layout-cell-3"
                    elif estado == 2 and 'layout-cell layout-cell-3 text-align-right' in classes:
                        estado = 3  # Próximo estado para buscar a descrição
                    elif estado == 3 and 'layout-cell layout-cell-9' in classes:
                        descricao = sibling.get_text(strip=True).replace('\t','').replace('\n',' ')
                        # Prepara para adicionar ao dicionário e reinicia o ciclo
                        if chave and titulo_projeto:
                            projeto_pesquisa = {
                                "chave": chave,
                                "titulo_projeto": titulo_projeto,
                                "descricao": descricao
                            }
                            self.estrutura["ProjetosOutros"].append(projeto_pesquisa)
                            chave = titulo_projeto = descricao = None  # Reinicia para o próximo ciclo
                            estado = 0  # Volta ao estado inicial
                            
                                
    def process_producoes(self):
        self.estrutura["Produções"]={}
        dados_artigos = []
        ano=''
        issn=''
        titulo=''
        autores=''
        primeiro_autor=''
        fator_impacto = ''
        qualis_info = ''
        secoes = self.soup.find_all('div', class_='title-wrapper')
        for secao in secoes:
            titulo_h1 = secao.find('h1')
            if titulo_h1 and 'Produções' in titulo_h1.get_text(strip=True):
                data_cell = secao.find('div', class_='layout-cell layout-cell-12 data-cell')
                ## Extrair dados dos artigos em periódicos
                div_artigos = data_cell.find_all("div", id="artigos-completos", recursive=False)
                for div_artigo in div_artigos:
                    subsecao = div_artigo.find('b')
                    if subsecao:
                        subsec_name = subsecao.get_text(strip=True)
                        # print(subsec_name)
                    artigos_completos = div_artigo.find_all("div", class_="artigo-completo", recursive=False)
                    for artigo_completo in artigos_completos:
                        dados_qualis = artigo_completo.find('p')
                        layout_cell = artigo_completo.find("div", class_="layout-cell layout-cell-11")
                        # print(f'\nlayout_cell: {layout_cell}')

                        # Extrai ano da publicação
                        ano_tag = layout_cell.find('span', {'class': 'informacao-artigo', 'data-tipo-ordenacao': 'ano'})
                        ano = int(ano_tag.text) if ano_tag else 'Ano não disponível'                        
                        ano = self.extract_year(layout_cell)

                        # Extrair estrato qualis e impacto JCR
                        qualis_info = self.extract_qualis(layout_cell)

                        # Extrai o título, periódico, e outras informações diretamente do texto
                        texto_completo = layout_cell.get_text(separator=' ', strip=True)
                        
                        # Assume que título vem após autores e termina antes de indicação de periódico ou volume
                        titulo_match = re.search(r'; ([^;]+?)\.', texto_completo)
                        autores = titulo_match.groups() if titulo_match else None

                        # Expressão regular para capturar as partes especificadas
                        pattern = re.compile(
                            r'(?P<primeiro_autor>.*?) ' # Capta tudo até o primeiro espaço antes do ano, de maneira não gananciosa
                            r'(?P<ano>\d{4}) ' # Captura o ano como uma sequência de 4 dígitos
                            r'(?P<autores>.+?) ' # Capta tudo após o ano até chegar no ponto que indica o fim dos autores, de maneira não gananciosa
                            r'\. ' # Capta o ponto e o espaço que indica o término da seção de autores
                            r'(?P<titulo_revista>.+?) ' # Capta o título de maneira não gananciosa até encontrar "v. "
                            r'v\. ' # Identifica o início dos detalhes da publicação, marcando o fim do título
                        )

                        # Busca na string pelos padrões
                        match = pattern.search(texto_completo)

                        # Verifica se houve correspondência e extrai os grupos
                        if match:
                            primeiro_autor = match.group('primeiro_autor')
                            ano = match.group('ano')
                            autores = match.group('autores')
                            titulo = match.group('titulo_revista').split('. ')[0]
                            revista = match.group('titulo_revista').split('. ')[1].replace(' ,','')
                        else:
                            print("Não foi possível extrair dados de layout_cell.")

                        # span_transform = layout_cell.find("span", class_="transform")
                        # print(f'\nspan_transform: {span_transform}')
                        # spans_infoartigo = layout_cell.find_all("span", class_="informacao-artigo")
                        # print(f'{len(spans_infoartigo)} spans de info_artigo')
                        img_tag = layout_cell.find("img")
                        if img_tag:
                            # print(f'{img_tag} img de info_artigo')
                            # Acessando diretamente os atributos do elemento <img> encontrado
                            original_title = img_tag.get('original-title')
                            # print(f'\noriginal_title: {original_title}')
                            issn = img_tag.get('data-issn')
                            # print(f'data-issn: {issn}')
                        else:
                            print('Não foi possível extrair originall_title')
                        if dados_qualis:
                            dados_qualis_txt = dados_qualis.get_text(strip=True)
                            segmentos = dados_qualis_txt.split(',')
                            if len(segmentos) >= 3:
                                # Ajusta para garantir que nenhum índice será acessado fora dos limites
                                segmentos_dict = {
                                    "ano": ano,
                                    "qualis": segmentos[0].strip(),
                                    "fator_impacto_jcr": qualis_info['JCR'],                                    
                                    "fonte": " ".join(segmentos[2:]).replace("fonteQualis/","").strip(),
                                    "ISSN": segmentos[1].replace("ISSN","").strip() if len(segmentos) > 1 else "",
                                    "data_issn": issn,
                                    "titulo": titulo,
                                    "primeiro_autor": primeiro_autor,
                                    "autores": autores,
                                    "revista": revista,
                                    "dados_completos": texto_completo,
                                }
                                dados_artigos.append(segmentos_dict) 
                
                self.estrutura["Produções"][subsec_name] = dados_artigos

                ## Extrair demais produções
                divs_cita_artigos = data_cell.find_all("div", class_="cita-artigos", recursive=False)
                for div_cita_artigos in divs_cita_artigos:
                    ocorrencias = {}
                    subsecao = div_cita_artigos.find('b')
                    if subsecao:
                        subsec_name = subsecao.get_text(strip=True)
                        # print(subsec_name)
                    # Encontrar todos os elementos irmãos seguintes de div_cita_artigos
                    next_siblings = div_cita_artigos.find_next_siblings("div")

                    # Listas para armazenar os divs encontrados
                    divs_indices = []
                    divs_ocorrencias = []

                    # Iterar sobre os elementos irmãos
                    for sibling in next_siblings:
                        # Verificar se o irmão tem a classe "cita-artigos"
                        if 'cita-artigos' in sibling.get('class', []):
                            # Encontramos o marcador para parar, sair do loop
                            break
                        # Verificar as outras classes e adicionar aos arrays correspondentes
                        if 'layout-cell layout-cell-1 text-align-right' in " ".join(sibling.get('class', [])):
                            divs_indices.append(sibling)
                        elif 'layout-cell layout-cell-11' in " ".join(sibling.get('class', [])):
                            divs_ocorrencias.append(sibling)
                    
                    if len(divs_indices) == len(divs_ocorrencias):
                        # Itera sobre o intervalo do comprimento de uma das listas
                        for i in range(len(divs_indices)):
                            # Usa o texto ou outro identificador único dos elementos como chave e valor
                            chave = divs_indices[i].get_text(strip=True)
                            valor = divs_ocorrencias[i].get_text(strip=True)

                            # Adiciona o par chave-valor ao dicionário
                            ocorrencias[chave] = valor

                    self.estrutura["Produções"][subsec_name] = ocorrencias

        return self.estrutura["Produções"]
    
    def process_bancas(self):
        self.estrutura["Bancas"]={}
        secoes = self.soup.find_all('div', class_='title-wrapper')
        for secao in secoes:
            titulo_h1 = secao.find('h1')
            if titulo_h1 and 'Bancas' in titulo_h1.get_text(strip=True):
                data_cell = secao.find('div', class_='layout-cell layout-cell-12 data-cell')
                subsecoes = data_cell.find_all('div', class_='inst_back')
                geral={}
                for subsecao in subsecoes:
                    if subsecao:
                        subsec_name = subsecao.get_text(strip=True)
                        # print(subsec_name)
                    ## Extrair cada banca
                    divs_cita_artigos = data_cell.find_all("div", class_="cita-artigos", recursive=False)
                    for div_cita_artigos in divs_cita_artigos:
                        ocorrencias = {}
                        subsecao = div_cita_artigos.find('b')
                        if subsecao:
                            subsec_name = subsecao.get_text(strip=True)
                            # print(subsec_name)
                        # Encontrar todos os elementos irmãos seguintes de div_cita_artigos
                        next_siblings = div_cita_artigos.find_next_siblings("div")

                        # Listas para armazenar os divs encontrados
                        divs_indices = []
                        divs_ocorrencias = []

                        # Iterar sobre os elementos irmãos
                        for sibling in next_siblings:
                            # Verificar se o irmão tem a classe "cita-artigos"
                            if 'cita-artigos' in sibling.get('class', []):
                                # Encontramos o marcador para parar, sair do loop
                                break
                            # Verificar as outras classes e adicionar aos arrays correspondentes
                            if 'layout-cell layout-cell-1 text-align-right' in " ".join(sibling.get('class', [])):
                                divs_indices.append(sibling)
                            elif 'layout-cell layout-cell-11' in " ".join(sibling.get('class', [])):
                                divs_ocorrencias.append(sibling)
                        
                        if len(divs_indices) == len(divs_ocorrencias):
                            # Itera sobre o intervalo do comprimento de uma das listas
                            for i in range(len(divs_indices)):
                                # Usa o texto ou outro identificador único dos elementos como chave e valor
                                chave = divs_indices[i].get_text(strip=True).replace('\t','').replace('\n',' ')
                                valor = divs_ocorrencias[i].get_text(strip=True).replace('\t','').replace('\n',' ')

                                # Adiciona o par chave-valor ao dicionário
                                ocorrencias[chave] = valor
                        # geral.append(ocorrencias)
                        self.estrutura["Bancas"][subsec_name] = ocorrencias

    def process_orientacoes(self):
        self.estrutura["Orientações"]={}
        secoes = self.soup.find_all('div', class_='title-wrapper')
        for secao in secoes:
            titulo_h1 = secao.find('h1')
            if titulo_h1 and 'Orientações' in titulo_h1.get_text(strip=True):
                data_cell = secao.find('div', class_='layout-cell layout-cell-12 data-cell')
                subsecoes = data_cell.find_all('div', class_='inst_back')
                for subsecao in subsecoes:
                    ocorrencias = {}
                    if subsecao:
                        subsec_name = subsecao.get_text(strip=True)
                        # print(f'Seção: {subsec_name}')
                        if subsec_name not in self.estrutura["Orientações"]:                       
                            self.estrutura["Orientações"][subsec_name] = []
                    
                    ## Extrair cada tipo de orientação
                    divs_cita_artigos = data_cell.find_all("div", class_="cita-artigos", recursive=False)
                    for div_cita_artigos in divs_cita_artigos:
                        ocorrencias = {}
                        subsubsecao = div_cita_artigos.find('b')
                        if subsubsecao:
                            subsubsecao_name = subsubsecao.get_text(strip=True)
                            # print(f'      Subseção: {subsubsecao_name}')
                        # Encontrar todos os elementos irmãos seguintes de div_cita_artigos
                        next_siblings = div_cita_artigos.find_next_siblings("div")

                        # Listas para armazenar os divs encontrados
                        divs_indices = []
                        divs_ocorrencias = []

                        # Iterar sobre os elementos irmãos
                        for sibling in next_siblings:
                            # Verificar se o irmão tem a classe "cita-artigos" ou "inst_back"
                            if 'cita-artigos' in sibling.get('class', []) or 'inst_back' in sibling.get('class', []):
                                # Encontramos o marcador para parar, sair do loop
                                break
                            # Verificar as outras classes e adicionar aos arrays correspondentes
                            if 'layout-cell layout-cell-1 text-align-right' in " ".join(sibling.get('class', [])):
                                divs_indices.append(sibling)
                            elif 'layout-cell layout-cell-11' in " ".join(sibling.get('class', [])):
                                divs_ocorrencias.append(sibling)
                        
                        if len(divs_indices) == len(divs_ocorrencias):
                            # Itera sobre o intervalo do comprimento de uma das listas
                            for i in range(len(divs_indices)):
                                # Usa o texto ou outro identificador único dos elementos como chave e valor
                                chave = divs_indices[i].get_text(strip=True).replace('\t','').replace('\n',' ')
                                valor = divs_ocorrencias[i].get_text(strip=True).replace('\t','').replace('\n',' ')

                                # Adiciona o par chave-valor ao dicionário
                                ocorrencias[chave] = valor

                        self.estrutura["Orientações"][subsec_name].append({subsubsecao_name: ocorrencias})

    def process_all(self):
        ## IDENTIFICAÇÃO/FORMAÇÃO
        self.process_identification()           # Ok!
        self.process_idiomas()                  # Ok!
        self.process_formacao()                 # Ok!
        ## ATUAÇÃO
        self.process_atuacao_profissional()     # Ok!
        self.process_linhas_pesquisa()          # Ok!
        self.process_areas()                    # Ok!
        ## PROJETOS
        self.process_projetos_pesquisa()        # Ok!
        self.process_projetos_extensao()        # Ok!
        self.process_projetos_desenvolvimento() # Ok!
        self.process_projetos_outros()          # Ok!
        ## PRODUÇÕES
        self.process_producoes()                # Ok!       
        ## EDUCAÇÃO
        self.process_bancas()                   # Ok!
        self.process_orientacoes()              # Ok!

        # TO-DO-LATER em Produções extraindo vazio
        # "Citações": {},                           
        # "Resumos publicados em anais de congressos (artigos)": {} 

    def to_json(self):
        self.process_all()
        return json.dumps(self.estrutura, ensure_ascii=False, indent=4)    