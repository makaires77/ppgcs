from transformers import BertModel, BertTokenizer
import numpy as np
from neo4j import GraphDatabase
from gensim import corpora, models
from lda_extractor import LDAExtractor

class ClassificadorArtigosCNPq:
    def __init__(self, lda_extractor: LDAExtractor):
        self.lda_extractor = lda_extractor

    def __init__(self, uri, user, password):
        # Inicializar o cliente do Neo4j
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        # Carregar o modelo BERT e o tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        self.model = BertModel.from_pretrained('bert-base-multilingual-cased')
        # Inicializar o modelo LDA para classificação nas GrandeÁreas
        self.lda_model = None
        self.dictionary = None

    def close(self):
        # Fechar a conexão com o banco de dados
        self.driver.close()

    def obter_classificacoes_pesquisador(self, idLattes):
        # Query Neo4j para obter a classificação do pesquisador usando a propriedade "atuacao"
        query = """
        MATCH (p:Person)-[:MEMBER_OF]-(t:Team)-[:INCLUDES]-(o:Organization)
        WHERE p.idLattes = $id_pesquisador
        RETURN p.atuacao as Atuacoes
        """
        with self.driver.session() as session:
            result = session.run(query, id_pesquisador=idLattes)
            record = result.single()
            atuacoes = record['Atuacoes'] if record else []
            # Processar a string de atuacao para extrair as classificações de Grande Área e Área
            classificacoes = self.processar_atuacao(atuacoes)
            return classificacoes

    def obter_hierarquia_classificacao(self):
        # Query Neo4j para obter a hierarquia de classificação completa
        query = """
        MATCH (ga:GrandeArea)-[:CONTÉM_ÁREA]-(a:Area)-[:CONTÉM_SUBÁREA]-(sa:Subarea)-[:CONTÉM_ESPECIALIDADE]-(esp:Especialidade)
        RETURN ga.name as GrandeArea, a.name as Area, sa.name as Subarea, esp.name as Especialidade
        """

        with self.driver.session() as session:
            result = session.run(query)
            hierarquia = []
            for record in result:
                hierarquia.append({
                'GrandeArea': record['GrandeArea'],
                'Area': record['Area'],
                'Subarea': record['Subarea'],
                'Especialidade': record['Especialidade']
                })
        return hierarquia

    def processar_atuacao(self, atuacoes):
        # Onde 'atuacoes' é lista de strings com as áreas de atuação do pesquisador
        # A função deve processar essa lista e extrair as informações relevantes
        grande_areas = set()
        areas = set()

        for atuacao in atuacoes:
            # Processar cada string de atuação e extrair a Grande Área e Área correspondente, dependendo do formato específico das strings de atuação.
            # Por exemplo:
            # Se a string de atuação é "1. Grande área: Ciências Biológicas / Área: Bioquímica / ..."
            # Vai extrair "Ciências Biológicas" como a Grande Área e "Bioquímica" como a Área.
            parts = atuacao.split(' / ')
            if parts:
                grande_area = parts[0].split(': ')[-1].strip()
                area = parts[1].split(': ')[-1].strip() if len(parts) > 1 else None
                grande_areas.add(grande_area)
                if area:
                    areas.add(area)

        return {'GrandeAreas': list(grande_areas), 'Areas': list(areas)}

    def preprocessar_texto(self, texto):
        # Preprocessar texto (tokenização, remoção de stopwords, etc.)
        texto_preprocessado = ''
        return texto_preprocessado

    def vetorizar_texto(self, texto):
        # Vetorizar um texto usando BERT
        inputs = self.tokenizer(texto, return_tensors='pt', max_length=512, truncation=True)
        outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).detach().numpy()

    def treinar_lda_para_grandeareas(self, artigos):
        # Treinar um modelo LDA com os títulos dos artigos para classificação nas GrandeÁreas
        textos = [self.preprocessar_texto(artigo['titulo']) for artigo in artigos]
        self.dictionary = corpora.Dictionary(textos)
        corpus = [self.dictionary.doc2bow(texto) for texto in textos]
        self.lda_model = models.LdaModel(corpus, num_topics=8, id2word=self.dictionary, passes=15)

    def classificar_grandearea_lda(self, titulo):
        # Classificar a GrandeÁrea de um artigo usando o modelo LDA
        texto_preprocessado = self.preprocessar_texto(titulo)
        bow = self.dictionary.doc2bow(texto_preprocessado)
        distribuicao_topicos = self.lda_model.get_document_topics(bow)
        
        # Retornar a GrandeÁrea com maior probabilidade
        grande_area_mais_provavel = max(distribuicao_topicos, key=lambda item: item[1])[0]
        return grande_area_mais_provavel

    def classificar_artigos_similaridade(self, artigos, classificacoes):
        resultados = {}
        for artigo in artigos:
            # Vetorizar o título do artigo usando BERT
            vetor_artigo = self.vetorizar_texto(artigo['titulo'])
            probabilidades = {}

            # Obter as classificações de Grande Área e Área do pesquisador
            id_pesquisador = artigo['autor_id']  # Supondo que cada artigo tenha o ID do autor
            classificacoes_pesquisador = self.obter_classificacoes_pesquisador(id_pesquisador)

            for classificacao in classificacoes:
                # Verificar se a classificação está dentro das áreas de atuação do pesquisador
                if classificacao['GrandeÁrea'] in classificacoes_pesquisador['GrandeAreas'] and \
                classificacao['Área'] in classificacoes_pesquisador['Areas']:
                    vetor_classificacao = self.vetorizar_texto(classificacao['rótulo'])
                    similaridade = self.calcular_similaridade_no_neo4j(vetor_artigo, vetor_classificacao)

                    # Ponderar a similaridade com base na classificação LDA e do pesquisador
                    similaridade_ponderada = self.ponderar_similaridade(similaridade, classificacao, classificacoes_pesquisador)
                    probabilidades[classificacao['rótulo']] = similaridade_ponderada

            resultados[artigo['titulo']] = probabilidades

        return resultados


    def classificar_artigos_hierarquia(self, artigos, hierarquia_classificacao):
        """
        No método `classificar_artigos`, a similaridade ponderada (`similaridade_ponderada`) deve ser calculada levando em conta a proximidade entre o tópico do artigo, determinado pelo LDA, e a hierarquia de classificações do CNPq. Isso pode ser feito com base nos pesos definidos para as 'GrandeÁreas', 'Áreas', 'Subáreas' e 'Especialidades' que correspondem à atuação do pesquisador, que deve ser extraída da base de dados do Neo4j. 

        O método `ponderar_similaridade` precisa ser implementado de maneira a combinar os resultados do LDA e da similaridade textual com a estrutura da árvore de classificações. Esta implementação específica dependerá da lógica de negócio e dos critérios de ponderação que se deseja aplicar.

        O treinamento do modelo LDA e a vetorização dos textos utilizando BERT devem ser cuidadosamente projetados para garantir que os tópicos e os vetores sejam representativos e úteis para a tarefa de classificação.
        """
        resultados = {}
        for artigo in artigos:
            vetor_artigo = self.vetorizar_texto(artigo['titulo'])
            probabilidades = {}
            for classificacao in hierarquia_classificacao:
                vetor_classificacao = self.vetorizar_texto(f"{classificacao['GrandeArea']} {classificacao['Area']} {classificacao['Subarea']} {classificacao['Especialidade']}")
                similaridade = self.calcular_similaridade_no_neo4j(vetor_artigo, vetor_classificacao)

                # Ajustar a similaridade com base na classificação LDA e do pesquisador
                similaridade_ponderada = self.ponderar_similaridade(similaridade, classificacao, artigo['autor_id'])
                probabilidades[classificacao['Especialidade']] = similaridade_ponderada

            resultados[artigo['titulo']] = probabilidades

        return resultados

    def calcular_similaridade_no_neo4j(self, vetor_artigo, vetor_classificacao):
        # Utilizar uma função de similaridade, como a similaridade de cosseno, para comparar os vetores
        return np.dot(vetor_artigo, vetor_classificacao) / (np.linalg.norm(vetor_artigo) * np.linalg.norm(vetor_classificacao))

    def ponderar_similaridade(self, similaridade, classificacao, classificacoes_pesquisador):
        # Ajustar a similaridade baseando-se em pesos específicos ou lógica de negócios
        # Aqui você pode implementar uma lógica para ajustar a similaridade com base na
        # importância relativa das classificações LDA e das informações do pesquisador
        similaridade_ponderada = 0
        return similaridade_ponderada


class TrainingPipeline:
    """
    Essa classe unifica o processo de treinamento em uma sequência lógica, facilitando a execução, manutenção e alterações futuras no pipeline. Dependendo do tamanho e da complexidade dos dados, você pode considerar a implementação de funcionalidades de processamento distribuído e paralelo. 

    Lembre-se de que a eficácia dessas sugestões pode variar dependendo dos requisitos específicos do seu projeto e do ambiente em que ele está sendo desenvolvido.
    """
    def __init__(self, neo4j_connector, lda_extractor, bert_embedder):
        self.neo4j_connector = neo4j_connector
        self.lda_extractor = lda_extractor
        self.bert_embedder = bert_embedder

    def run(self):
        # 1. Extrair e processar dados do Neo4j
        researchers_data = self.neo4j_connector.extract_researchers_data()
        
        # 2. Processar embeddings com BERT
        embeddings = self.bert_embedder.process_embeddings(researchers_data)

        # 3. Executar LDA para tópico modelagem
        lda_model, corpus = self.lda_extractor.fit_transform(embeddings)

        # 4. (Opcional) Outros passos de processamento e análise
        # ...

        # 5. Treinar o modelo de Graph Machine Learning (GML)
        gml_model = self.train_gml_model(lda_model, corpus, embeddings)

        # 6. Salvar o modelo treinado e/ou resultados intermediários
        self.save_model(gml_model)

    def train_gml_model(self, lda_model, corpus, embeddings):
        # Implementação de treinamento do modelo GML
        gml_model = {}
        return gml_model

    def save_model(self, model):
        # Implementação para salvar o modelo treinado
        return
    
# Outros métodos e funções necessárias

# Para esta implementação, seria necessário conhecer o formato exato das strings de atuação para processá-las corretamente. Além disso, as funções `calcular_similaridade_no_neo4j` e `ponderar_similaridade` são esboços genéricos que precisariam ser preenchidos com a lógica específica de similaridade e ponderação desejada.

# Esta implementação assume que você pode acessar as propriedades do nó `Person` diretamente pelo ID, e que a estrutura do dado `atuacao` é uma lista de strings com informações hierárquicas de classificação. As funções de similaridade e ponderação podem ser expandidas com base nos requisitos de negócios e validações de especialistas do domínio.

#Pontos-Chave e Considerações:

# - **Treinamento do LDA**: O método `treinar_lda_para_grandeareas` é responsável por treinar um modelo LDA usando os títulos dos artigos. Isso permitirá classificar os artigos nas 'GrandeÁreas' com base no conteúdo dos títulos.
  
# - **Preprocessamento de Texto**: O método `preprocessar_texto` será usado para preparar os títulos dos artigos antes de alimentá-los ao modelo LDA. Este método deve incluir etapas típicas de PLN, como tokenização, remoção de stopwords, lematização, etc.

# - **Classificação com LDA e Ponderação**: A função `classificar_grandearea_lda` utiliza o modelo LDA para determinar a 'GrandeÁrea' mais provável para um dado título de artigo. O resultado desta classificação é então usado para ponderar as probabilidades na função `ponderar_similaridade`, junto com as informações das 'GrandeÁreas' e 'Áreas' do pesquisador.

# - **Segurança e Gerenciamento de Recursos**: Como a classe interage com um banco de dados Neo4j e processa grandes quantidades de texto, é essencial garantir a segurança das credenciais e gerenciar eficientemente os recursos computacionais.

# Esta abordagem integrada com análise LDA e processamento de texto BERT, juntamente com as informações de classificação do pesquisador e a interação com o Neo4j, oferece um sistema robusto para a classificação de artigos científicos. A combinação dessas técnicas permite uma classificação mais precisa e contextualizada, levando em conta não apenas o conteúdo dos artigos, mas também as redes de pesquisa e as áreas de especialização dos pesquisadores.

# Considerações Adicionais:

# Propagação de Rótulos: 
# O método obter_classificacoes_pesquisador pode usar a propagação de rótulos para determinar as classificações de GrandeÁrea e Área de um pesquisador, principalmente quando essas informações não estão explicitamente disponíveis. Este processo pode ser implementado usando algoritmos de propagação de rótulos disponíveis no Neo4j GDS.

# Filtragem por Classificação de Grande Área e Área: 
# Ao classificar os artigos, a classe agora filtra as classificações de Subárea e Especialidade com base nas GrandeÁreas e Áreas às quais o pesquisador está associado. Isso garante que os artigos sejam classificados apenas dentro das áreas relevantes ao pesquisador.

# Cálculo de Similaridade: 
# A similaridade entre os embeddings dos títulos dos artigos e os rótulos de classificação é calculada para as classificações de Subárea e Especialidade. Esse cálculo pode ser otimizado utilizando o GDS do Neo4j, que oferece algoritmos eficientes para cálculo de similaridade em grandes conjuntos de dados.

# Integração com Neo4j: 
# A classe interage com o Neo4j para realizar consultas e operações relacionadas ao grafo. Isso inclui tanto a obtenção de informações de classificação quanto o cálculo de similaridade.
# Ajustes e Otimização: Dependendo dos resultados iniciais, pode ser necessário ajustar os parâmetros do modelo LDA (como o número de tópicos) ou a lógica de ponderação da similaridade para melhorar a precisão da classificação.
# Integração e Escalabilidade: Ao implementar esta classe em um sistema maior, deve-se considerar a integração com outros componentes do sistema e garantir que a solução seja escalável para lidar com grandes conjuntos de dados.
# Atualização e Manutenção do Modelo: Com o tempo, novos artigos serão inseridos e a classificação nas áreas de pesquisa podem evoluir. Portanto, é importante manter o modelo atualizado e reavaliá-lo periodicamente.
# Usamos técnicas avançadas de PLN, análise de grafos e aprendizado de máquina para criar um sistema de classificação de artigos científicos altamente sofisticado e adaptado às necessidades específicas da classificação multinível do CNPq