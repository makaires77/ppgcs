## Classes:
# GraphBuilder
# GraphEmbeddingLearner
# RecommenderEngine
# ReportGenerator
# PDIRecommendationSystem

from node2vec import Node2Vec
import networkx as nx

class GraphBuilder:
    def __init__(self):
        self.graph = nx.Graph()

    def add_nodes_from_dataframe(self, df, node_type, node_id_col, node_attr_cols=[]):
        """
        Adds nodes to the graph from a DataFrame.

        Args:
            df: The DataFrame containing the node data
            node_type: The type of the nodes (e.g., 'researcher', 'product', 'funding_opportunity')
            node_id_col: The name of the column in the DataFrame containing the unique node IDs
            node_attr_cols: A list of column names to be added as node attributes
        """
        for _, row in df.iterrows():
            node_id = row[node_id_col]
            node_attrs = {col: row[col] for col in node_attr_cols if col in df.columns}
            self.graph.add_node(node_id, type=node_type, **node_attrs)

    def add_edges_from_dataframe(self, df, source_col, target_col, edge_attr_cols=[]):
        """
        Adds edges to the graph from a DataFrame

        Args:
            df: The DataFrame containing the edge data
            source_col: The name of the column in the DataFrame containing the source node IDs
            target_col: The name of the column in the DataFrame containing the target node IDs
            edge_attr_cols: A list of column names to be added as edge attributes
        """
        for _, row in df.iterrows():
            source_id = row[source_col]
            target_id = row[target_col]
            edge_attrs = {col: row[col] for col in edge_attr_cols if col in df.columns}
            self.graph.add_edge(source_id, target_id, **edge_attrs)

    def build_graph(self, curriculos_df, produtos_df, editais_df):
        """
        Builds the knowledge graph by adding nodes and edges from the provided DataFrames

        Args:
            curriculos_df: The DataFrame containing researcher CV data
            produtos_df: The DataFrame containing strategic health products data
            editais_df: The DataFrame containing funding opportunities data
        """

        # Add nodes for researchers, products, and funding opportunities
        self.add_nodes_from_dataframe(curriculos_df, 'researcher', 'id_pesquisador', ['nome_pesquisador', 'area_pesquisa'])  # Substitua os nomes das colunas conforme necessário
        self.add_nodes_from_dataframe(produtos_df, 'product', 'id_produto', ['nome_produto', 'area_terapeutica'])        # Substitua os nomes das colunas conforme necessário
        self.add_nodes_from_dataframe(editais_df, 'funding_opportunity', 'id_edital', ['titulo_edital', 'area_tematica'])  # Substitua os nomes das colunas conforme necessário

        # Add edges based on relationships between entities (you'll need to define these relationships based on your data)
        # Example: Connect researchers to products they work on
        # self.add_edges_from_dataframe(relacionamento_pesquisador_produto_df, 'id_pesquisador', 'id_produto', ['tipo_relacionamento']) 

        # Example: Connect products to relevant funding opportunities
        # self.add_edges_from_dataframe(relacionamento_produto_edital_df, 'id_produto', 'id_edital', ['relevancia']) 

        return self.graph


class GraphEmbeddingLearner:
    def __init__(self, dimensions=128, walk_length=30, num_walks=200, workers=4, p=1, q=1):
        """
        Initializes the GraphEmbeddingLearner with Node2Vec parameters.

        Args:
            dimensions: The dimensionality of the learned node embeddings (default: 128).
            walk_length: The length of random walks to generate (default: 30).
            num_walks: The number of random walks to generate for each node (default: 200).
            workers: The number of worker threads to use for generating random walks (default: 4).
            p: The return parameter for Node2Vec (default: 1).
            q: The in-out parameter for Node2Vec (default: 1).
        """
        self.dimensions = dimensions
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.workers = workers
        self.p = p
        self.q = q

    def learn_embeddings(self, graph):
        """
        Learns node embeddings for the given graph using Node2Vec.

        Args:
            graph: The NetworkX graph to learn embeddings for.

        Returns:
            A dictionary mapping node IDs to their learned embeddings.
        """
        try:
            node2vec = Node2Vec(graph, dimensions=self.dimensions, walk_length=self.walk_length, num_walks=self.num_walks, workers=self.workers, p=self.p, q=self.q)
            model = node2vec.fit(window=10, min_count=1, batch_words=4)  # You can adjust these parameters as needed
            node_embeddings = {node: model.wv[str(node)] for node in graph.nodes()}
            return node_embeddings
        except Exception as e:
            logging.error(f"Erro ao aprender embeddings com Node2Vec: {e}")
            raise

from sklearn.metrics.pairwise import cosine_similarity

class RecommenderEngine:
    def __init__(self, node_embeddings, sentence_embeddings, curriculos_df, produtos_df, editais_df):
        """
        Initializes the RecommenderEngine with node embeddings, sentence embeddings, and DataFrames.

        Args:
            node_embeddings: A dictionary mapping node IDs to their learned embeddings from the graph.
            sentence_embeddings: A dictionary mapping text IDs to their sentence embeddings.
            curriculos_df: The DataFrame containing researcher CV data.
            produtos_df: The DataFrame containing strategic health products data.
            editais_df: The DataFrame containing funding opportunities data.
        """
        self.node_embeddings = node_embeddings
        self.sentence_embeddings = sentence_embeddings
        self.curriculos_df = curriculos_df
        self.produtos_df = produtos_df
        self.editais_df = editais_df

    def get_recommendations(self, entity_id, entity_type, top_n=5):
        """
        Gets recommendations for a given entity (researcher or product) based on semantic similarity.

        Args:
            entity_id: The ID of the entity to get recommendations for.
            entity_type: The type of the entity ('researcher' or 'product').
            top_n: The number of top recommendations to return (default: 5).

        Returns:
            A list of tuples containing the recommended entity IDs and their similarity scores.
        """

        if entity_type not in ['researcher', 'product']:
            raise ValueError("Tipo de entidade inválido. Use 'researcher' ou 'product'.")

        # Get the embedding of the entity
        entity_embedding = self.node_embeddings.get(entity_id)
        if entity_embedding is None:
            logging.warning(f"Embedding não encontrado para a entidade {entity_id} do tipo {entity_type}")
            return []  # Ou retorne uma lista vazia ou um valor padrão, dependendo da sua lógica

        # Get embeddings of all funding opportunities
        funding_opportunity_embeddings = [self.sentence_embeddings.get(edital_id) for edital_id in self.editais_df['id_edital']]

        # Calculate cosine similarity between the entity and all funding opportunities
        similarities = cosine_similarity([entity_embedding], funding_opportunity_embeddings)[0]

        # Get the top_n most similar funding opportunities
        top_indices = similarities.argsort()[-top_n:][::-1]
        top_recommendations = [(self.editais_df.iloc[i]['id_edital'], similarities[i]) for i in top_indices]

        return top_recommendations

    def identify_skill_gaps(self, product_id):
        """
        Identifies skill gaps for a given product by comparing its needs to researcher skills.

        Args:
            product_id: The ID of the product to identify skill gaps for.

        Returns:
            A list of skills that are needed for the product but not possessed by any researcher.
        """

        # Get the skills required for the product (you'll need to define how to extract these from your data)
        required_skills = self.produtos_df[self.produtos_df['id_produto'] == product_id]['habilidades_necessarias'].tolist()  # Substitua 'habilidades_necessarias' pelo nome da coluna relevante
        if not required_skills:
            logging.warning(f"Habilidades necessárias não encontradas para o produto {product_id}")
            return []

        # Get the skills of all researchers (you'll need to define how to extract these from your data)
        researcher_skills = set()
        for _, row in self.curriculos_df.iterrows():
            researcher_skills.update(row['habilidades'].split(', '))  # Substitua 'habilidades' pelo nome da coluna relevante

        # Identify skill gaps
        skill_gaps = [skill for skill in required_skills if skill not in researcher_skills]

        return skill_gaps


import jinja2
import os
import matplotlib.pyplot as plt

class ReportGenerator:
    def __init__(self, template_dir="~/ppgcs/source/template/"):
        """
        Initializes the ReportGenerator with the directory containing Jinja templates.

        Args:
            template_dir: The path to the directory containing Jinja template files (default: "~/ppgcs/source/template/").
        """
        self.template_loader = jinja2.FileSystemLoader(searchpath=os.path.expanduser(template_dir))
        self.template_env = jinja2.Environment(loader=self.template_loader)

    def generate_recommendations_report(self, recommendations, entity_id, entity_type, editais_df):
        """
        Generates an HTML report with recommendations for a given entity.

        Args:
            recommendations: A list of tuples containing recommended entity IDs and their similarity scores.
            entity_id: The ID of the entity the recommendations are for.
            entity_type: The type of the entity ('researcher' or 'product').
            editais_df: The DataFrame containing funding opportunities data.
        """
        template = self.template_env.get_template("recommendations_report.html")  # Assuming you have this template

        # Prepare data for the template
        recommendations_data = []
        for edital_id, similarity in recommendations:
            edital_data = editais_df[editais_df['id_edital'] == edital_id].iloc[0]  # Get edital details
            recommendations_data.append({
                'id_edital': edital_id,
                'titulo_edital': edital_data['titulo_edital'],  # Adjust column names as needed
                'area_tematica': edital_data['area_tematica'],
                'similarity': similarity
            })

        # Render the template
        report_content = template.render(
            entity_id=entity_id,
            entity_type=entity_type,
            recommendations=recommendations_data
        )

        # Save the report
        with open(f"recommendations_report_{entity_id}.html", "w", encoding="utf-8") as f:
            f.write(report_content)

        print(f"Relatório de recomendações gerado com sucesso: recommendations_report_{entity_id}.html")

    def generate_skill_gaps_report(self, skill_gaps, product_id):
        """
        Generates an HTML report with skill gaps for a given product.

        Args:
            skill_gaps: A list of skills that are needed but not possessed by any researcher.
            product_id: The ID of the product the skill gaps are for.
        """
        template = self.template_env.get_template("skill_gaps_report.html")  # Assuming you have this template

        # Render the template
        report_content = template.render(
            product_id=product_id,
            skill_gaps=skill_gaps
        )

        # Save the report
        with open(f"skill_gaps_report_{product_id}.html", "w", encoding="utf-8") as f:
            f.write(report_content)

        print(f"Relatório de lacunas de habilidades gerado com sucesso: skill_gaps_report_{product_id}.html")


class PDIRecommendationSystem:
    def __init__(self, curriculos_file, produtos_file, editais_file, use_gpu=True):
        self.data_preprocessor = DataPreprocessor(curriculos_file, produtos_file, editais_file, use_gpu)
        self.graph_builder = GraphBuilder()
        self.embedding_learner = GraphEmbeddingLearner()
        self.report_generator = ReportGenerator()

    def run(self):
        # Carregar e pré-processar os dados
        preprocessed_curriculos, preprocessed_produtos, preprocessed_editais = self.data_preprocessor.preprocess_data()

        # Construir o grafo de conhecimento
        graph = self.graph_builder.build_graph(curriculos_df, produtos_df, editais_df)  # Certifique-se de ter os DataFrames corretos aqui

        # Aprender embeddings dos nós do grafo
        node_embeddings = self.embedding_learner.learn_embeddings(graph)

        # Gerar embeddings de frases para os editais
        sentence_embeddings = {row['id_edital']: model_st.encode(row['texto_para_embedding']) for _, row in editais_df.iterrows()}

        # Criar o RecommenderEngine
        recommender_engine = RecommenderEngine(node_embeddings, sentence_embeddings, curriculos_df, produtos_df, editais_df)

        # Gerar recomendações para pesquisadores
        for _, row in curriculos_df.iterrows():
            researcher_id = row['id_pesquisador']
            recommendations = recommender_engine.get_recommendations(researcher_id, 'researcher')
            self.report_generator.generate_recommendations_report(recommendations, researcher_id, 'pesquisador', editais_df)

        # Gerar recomendações para produtos e identificar lacunas de habilidades
        for _, row in produtos_df.iterrows():
            product_id = row['id_produto']
            recommendations = recommender_engine.get_recommendations(product_id, 'product')
            self.report_generator.generate_recommendations_report(recommendations, product_id, 'produto', editais_df)

            skill_gaps = recommender_engine.identify_skill_gaps(product_id)
            self.report_generator.generate_skill_gaps_report(skill_gaps, product_id)