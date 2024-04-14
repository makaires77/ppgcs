import torch
import hashlib
import logging
from neo4j import GraphDatabase
from tqdm import tqdm, tqdm_notebook
from transformers import BertModel, BertTokenizer

# Configuração do logging
logging.basicConfig(filename='HierarchicalSemanticMatcher.log', level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

# Carregar o modelo e o tokenizer do BERT Multilíngue
model_name = 'bert-base-multilingual-cased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

class HierarchicalSemanticMatcher:
    def __init__(self, neo4j_uri, neo4j_user, neo4j_password):
        self._driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))

    def _execute_query(self, query, parameters=None):
        with self._driver.session() as session:
            return session.run(query, parameters)

    @staticmethod
    def get_embedding(text):
        """
        Obtenha o embedding de um texto usando o modelo BERT Multilíngue.
        Args:
        - text (str): O texto a ser vetorizado.
        Returns:
        - List[float]: O embedding do texto.
        """
        inputs = tokenizer(text, return_tensors="pt", max_length=128, truncation=True, padding="max_length")
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().tolist()

    # Vetorizar títulos de especialidades em dados já no Neo4j com vetores pré-treinados
    def vectorize_specialty_names(self):
        logging.info("Iniciando vetorização dos nomes das especialidades...")
        query = "MATCH (e:Especialidade) RETURN e.name AS name"
        result = self._driver.session().run(query)
        embeddings = []

        # Converter resultados em uma lista para poder usar o tqdm
        result_list = [record['name'] for record in result]
        
        logging.info(f"Total de nomes a serem vetorizados: {len(result_list)}")

        if not result_list:
            logging.warning("Nenhum nome de especialidade foi recuperado da base de dados.")
            return embeddings

        for name in tqdm_notebook(result_list, desc="Vetorizando nomes de especialidades"):
            embedding = self.get_embedding(name) # vetoriza
            embeddings.append({"name": name, "embedding": embedding})

            # translated_name = self._translate_text(name) # traduz
            # embedding = self.get_embedding(translated_name) # vetoriza
            # embeddings.append({"name": translated_name, "embedding": embedding})            

            # Atualizar nome no nó 'Especialidade' com o nome traduzido
            # update_query = """
            # MATCH (e:Especialidade {name: $original_name})
            # SET e.name = $translated_name
            # """
            # self._execute_query(update_query, parameters={"original_name": name, "translated_name": translated_name})
        
        logging.info(f"Vetorização concluída. {len(embeddings)} nomes foram vetorizados.")
        return embeddings

    # Persistir vetores gerados no Neo4j
    def update_specialty_nodes_with_embeddings(self, specialty_embeddings):
        """
        Atualiza os nós 'Especialidade' com os embeddings fornecidos.
        Args:
        - specialty_embeddings (List[Dict]): Lista de dicionários contendo os nomes (traduzidos) e seus embeddings.
        """
        query = """
        UNWIND $specialty_embeddings AS specialty_data
        MATCH (e:Especialidade {name: specialty_data.name})
        SET e.embedding = specialty_data.embedding
        """
        self._execute_query(query, parameters={"specialty_embeddings": specialty_embeddings})

    # Recuperar e Vetorizar títulos de subáreas em dados já no Neo4j usando vetores pré-treinados
    def vectorize_subarea_names(self):
        logging.info("Iniciando vetorização dos nomes das subareas...")
        query = "MATCH (e:Subárea) RETURN e.name AS name"
        result = self._driver.session().run(query)
        embeddings = []

        # Converter resultados em uma lista para poder usar o tqdm
        result_list = [record['name'] for record in result]
        
        logging.info(f"Total de nomes a serem vetorizados: {len(result_list)}")

        if not result_list:
            logging.warning("Nenhum nome de subárea foi recuperado da base de dados.")
            return embeddings

        for name in tqdm(result_list, desc="Vetorizando nomes de subáreas"):
            embedding = self.get_embedding(name) # vetoriza o nome de subárea em português
            embeddings.append({"name": name, "embedding": embedding})
            
            # translated_name = self._translate_text(name) # traduz
            # embedding = self.get_embedding(translated_name) # vetoriza o nome de subárea
            # embeddings.append({"name": translated_name, "embedding": embedding})

            # Atualizar nome no nó 'Subárea' com o nome traduzido
            # update_query = """
            # MATCH (e:Subárea {name: $original_name})
            # SET e.name = $translated_name
            # """
            # self._execute_query(update_query, parameters={"original_name": name, "translated_name": translated_name})
        
        logging.info(f"Vetorização concluída. {len(embeddings)} nomes foram vetorizados.")
        return embeddings

    # Persistir vetores gerados no Neo4j
    def update_subarea_nodes_with_embeddings(self, subarea_embeddings):
        """
        Atualiza os nós 'Subárea' com os embeddings fornecidos.
        Args:
        - subarea_embeddings (List[Dict]): Lista de dicionários contendo os nomes (traduzidos) e seus embeddings.
        """
        query = """
        UNWIND $subarea_embeddings AS subarea_data
        MATCH (s:Subárea {name: subarea_data.name})
        SET s.embedding = subarea_data.embedding
        """
        self._execute_query(query, parameters={"subarea_embeddings": subarea_embeddings})

    def generate_publication_id(self, title, author_index, year):
        """
        Gera um identificador único para uma publicação com base no título, índice do autor e ano.
        Args:
        - title (str): O título da publicação.
        - author_index (str): O índice do autor da publicação.
        - year (str): O ano da publicação.
        Returns:
        - str: O identificador único gerado.
        """
        id_string = f"{title}-{author_index}-{year}"
        # Utiliza SHA-256 para garantir um tamanho fixo e unicidade
        hashed_id = hashlib.sha256(id_string.encode()).hexdigest()
        return hashed_id

    def extract_titles_from_data(self, data_dict):
        """
        Extract titles, author indexes, and years from the given data dictionary structure.
        Args:
        - data_dict (Dict): The dictionary structure containing the data.
        Returns:
        - List[Dict]: A list of dictionaries with titles, author indexes, years, and their embeddings.
        """
        indices = []
        publications = []
        embeddings = []
        titles = []

        for author_index,value in tqdm(enumerate(data_dict), total=len(data_dict)):
            year = value.get('ano')
            title = value.get('titulo')
            idlattes = value.get('id_lattes')
            issn = value.get('issn')
            impact_fator = value.get('impact_factor')
            titles.append(title)
            indices.append(author_index)

            # Obter embedding para o título
            embedding = HierarchicalSemanticMatcher.get_embedding(title)
            embeddings.append(embedding)
            
            # Combinar título, índice do autor, ano e seu embedding em um único dicionário
            dic_embeeding = {"author_index": author_index, "id_lattes": idlattes, "year": year, "title": title, "impact_factor": impact_fator, "issn": issn, "embedding": embedding}
            publications.append(dic_embeeding)

        return publications

    def add_publications_with_embeddings(self, publications_list):
        """
        Adicione nós de 'Publicação' com embeddings de títulos e outras informações ao banco de dados Neo4j.
        Args:
        - publications_list (List[Dict]): Uma lista de dicionários contendo títulos, índices de autor, anos e seus embeddings.
        """
        query = """
        UNWIND $publications_list AS publication_data
        MERGE (p:Publicacao {titulo: publication_data.title, author_index: publication_data.author_index, ano: publication_data.year})
        ON CREATE SET p.titulo = publication_data.title, p.author_index = publication_data.author_index, p.ano = publication_data.year, p.embedding = publication_data.embedding
        """
        self._execute_query(query, parameters={"publications_list": publications_list})

    def add_titles_with_embeddings(self, titles):
        query = """
        UNWIND $titles AS title
        MERGE (t:Title {name: title.title})
        SET t.embedding = title.embedding
        """
        self._execute_query(query, titles=titles)

    def predict_grande_area(self, article_title):
        article_embedding = self.get_embedding(article_title)
        article_embedding_tensor = torch.tensor([article_embedding], dtype=torch.float)
        specialty_embeddings = self.neo4j_service.get_all_embeddings("Especialidade")
        subarea_embeddings = self.neo4j_service.get_all_embeddings("Subárea")
        most_similar_node_id = self.find_most_similar_node(article_embedding_tensor, specialty_embeddings + subarea_embeddings)
        grande_area = self.find_grande_area(most_similar_node_id)
        return grande_area        

    def add_publications_with_embeddings(self, titles_list):
        """
        Adicione nós de 'Publicação' com embeddings de títulos ao banco de dados Neo4j.
        Args:
        - titles_list (List[Dict]): Uma lista de dicionários contendo títulos e seus embeddings.
        """
        query = """
        UNWIND $titles_list AS title_data
        MERGE (p:Publicacao {titulo: title_data.title})
        SET p.ano = title_data.year
        SET p.issn = title_data.issn
        SET p.fator_impacto = title_data.impact_factor
        SET p.embedding = title_data.embedding
        """
        self._execute_query(query, parameters={"titles_list": titles_list})   