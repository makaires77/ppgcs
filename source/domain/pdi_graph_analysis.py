import json
import networkx as nx
import multiprocessing as mp
from collections import defaultdict
import torch
import torch_geometric
from dgl import DGLGraph

class PDIGraphBuilder:
    def __init__(self, demandas_path, questoes_path, pesquisadores_path, macroprocessos_path):
        self.questoes = self._load_json(questoes_path)
        self.demandas = self._load_json(demandas_path)
        self.pesquisadores = self._load_json(pesquisadores_path)
        self.macroprocessos = self._load_json(macroprocessos_path)
        self.grafo = nx.MultiDiGraph()

    def _load_json(self, path):
        with open(path, 'r') as f:
            return json.load(f)

    def _extract_competencias(self, texto):
        try:
            # Tradução (se necessário)
            if idioma != 'en':
                translator = Translator()
                texto = translator.translate(texto, dest='en').text

            # Normalização e limpeza (em inglês)
            nlp = spacy.load("en_core_web_sm")
            doc = nlp(texto.lower())

            # Normalização do texto
            texto = unidecode(texto).lower()  # Remove acentos e converte para minúsculas

            # Criação de padrões para o Matcher (exemplos)
            matcher = Matcher(nlp.vocab)
            patterns = [
                [{"POS": "NOUN"}],  # Substantivos
                [{"POS": "ADJ"}, {"POS": "NOUN"}],  # Adjetivo + Substantivo
                [{"POS": "VERB"}, {"POS": "NOUN"}]  # Verbo + Substantivo
            ]
            matcher.add("Competencias", patterns)

            # Extração de termos
            matches = matcher(doc)
            termos = [doc[start:end].text for _, start, end in matches]

            # Filtragem de termos
            stopwords = nlp.Defaults.stop_words  # Carrega as stopwords do spaCy
            competencias = [
                termo
                for termo in termos
                if termo not in stopwords and len(termo) > 2  # Remove stopwords e termos curtos
            ]

            return competencias

        except Exception as e:
            print(f"Erro ao extrair competências: {e}")
            return []  # Retorna lista vazia em caso de erro

    def build_graph(self):
        # Adiciona nós de produtos, processos e macroprocessos
        for bloco in self.demandas['blocos']:
            for desafio in bloco['desafios']:
                for produto in desafio['produtos']:
                    self.grafo.add_node(produto['id'], tipo='produto', nome=produto['nome'])

                for plataforma in desafio.get('plataformas', []):
                    self.grafo.add_node(plataforma['id'], tipo='plataforma', nome=plataforma['nome'])
                    self.grafo.add_edge(desafio['id'], plataforma['id'], tipo='possui_plataforma')

        for macroprocesso_id, macroprocesso in self.macroprocessos.items():
            self.grafo.add_node(macroprocesso_id, tipo='macroprocesso', nome=macroprocesso['nome'])
            for processo_id, processo in macroprocesso['processos'].items():
                self.grafo.add_node(processo_id, tipo='processo', nome=processo['nome'])
                self.grafo.add_edge(macroprocesso_id, processo_id, tipo='possui_processo')
                for entidade in processo['entidades']:
                    self.grafo.add_node(entidade, tipo='entidade')
                    self.grafo.add_edge(processo_id, entidade, tipo='possui_entidade')

        # Adiciona nós de pesquisadores e suas competências
        for pesquisador in self.pesquisadores:
            lattes_id = pesquisador['Identificação']['ID Lattes']
            self.grafo.add_node(lattes_id, tipo='pesquisador', nome=pesquisador['Identificação']['Nome'])
            for campo in ['Formação', 'Atuação Profissional', 'Linhas de Pesquisa', 'Áreas', 'Produções']:
                for _, info in pesquisador.get(campo, {}).items():
                    idioma = detect(info)  # Detectar o idioma do texto
                    competencias = self._extract_competencias(info, idioma)  # Extrair competências no idioma correto
                    for competencia in competencias:
                        self.grafo.add_node(competencia, tipo='competencia')
                        self.grafo.add_edge(lattes_id, competencia, tipo='possui_competencia')

    def save_graph(self, path="grafo_multiplex.gml"):
        nx.write_gml(self.grafo, path)

# Exemplo de uso (após implementar _extract_competencias):
builder = PDIGraphBuilder("demandas.json", "pesquisadores.json", "macroprocessos.json")
builder.build_graph()
builder.save_graph()