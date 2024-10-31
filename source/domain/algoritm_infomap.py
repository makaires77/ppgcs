from py2neo import Graph
import torch

class Neo4jInfomap:
    """
    Classe para aplicar o algoritmo Infomap em um grafo persistido no Neo4j.
    """

    def __init__(self, uri, auth, max_iter=100, tol=1e-6):
        """
        Inicializa a classe com os parâmetros de conexão do Neo4j.

        Args:
          uri: URI de conexão do Neo4j (ex: "bolt://localhost:7687").
          auth: Tupla com as credenciais de autenticação (ex: ("neo4j", "password")).
          max_iter: Número máximo de iterações do Infomap.
          tol: Tolerância para convergência do Infomap.
        """
        self.graph = Graph(uri, auth=auth)
        self.max_iter = max_iter
        self.tol = tol

    def _extract_graph_data(self, label=None, relationship_type=None):
        """
        Extrai os dados do grafo do Neo4j.

        Args:
          label: Rótulo do nó (opcional).
          relationship_type: Tipo de relacionamento (opcional).

        Returns:
          Tupla com a matriz de adjacência (tensor PyTorch) e um dicionário 
          mapeando IDs de nós no Neo4j para índices na matriz.
        """
        # Construir a consulta Cypher com base nos parâmetros
        query = "MATCH (n"
        if label:
            query += f":{label}"
        query += ")-[r"
        if relationship_type:
            query += f":{relationship_type}"
        query += "]->(m) RETURN id(n) AS source, id(m) AS target"

        data = self.graph.run(query).data()

        # Criar mapeamento de IDs para índices
        node_ids = set()
        for row in data:
            node_ids.add(row['source'])
            node_ids.add(row['target'])
        node_id_to_index = {node_id: i for i, node_id in enumerate(node_ids)}

        # Criar matriz de adjacência
        num_nodes = len(node_ids)
        adj_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
        for row in data:
            source_index = node_id_to_index[row['source']]
            target_index = node_id_to_index[row['target']]
            adj_matrix[source_index, target_index] = 1

        return adj_matrix, node_id_to_index

    def _calculate_map_equation(self, adj_matrix, map):
        """
        Calcula a equação do mapa para a estrutura de comunidade dada.

        Args:
          adj_matrix: Matriz de adjacência do grafo (tensor PyTorch).
          map: Estrutura da comunidade (lista de listas).

        Returns:
          Valor da equação do mapa.
        """
        adj_matrix = adj_matrix.to_dense()
        map_equation_value = 0.0
        for community in map:
            community_visit_prob = sum(adj_matrix[node].sum() for node in community) / adj_matrix.sum()
            community_entropy = -sum(
                (adj_matrix[node][:, community].sum() / adj_matrix[node].sum())
                * torch.log2(adj_matrix[node][:, community].sum() / adj_matrix[node].sum())
                for node in community
                if adj_matrix[node].sum() > 0
            )
            map_equation_value += community_visit_prob * community_entropy
        return map_equation_value

    def _move_nodes(self, adj_matrix, map):
        """
        Move iterativamente os nós entre as comunidades para melhorar a equação do mapa.

        Args:
          adj_matrix: Matriz de adjacência do grafo (tensor PyTorch).
          map: Estrutura da comunidade (lista de listas).

        Returns:
          Melhor estrutura de comunidade e seu valor da equação do mapa.
        """
        best_map = [m.copy() for m in map]
        best_map_equation = self._calculate_map_equation(adj_matrix, map)
        for node in range(adj_matrix.shape[0]):
            for community_idx, community in enumerate(map):
                new_map = [m.copy() for m in map]
                new_map[community_idx].append(node)
                original_community_idx = None
                for c_idx, c in enumerate(map):
                    if node in c:
                        original_community_idx = c_idx
                        new_map[c_idx].remove(node)
                        break
                if original_community_idx == community_idx:
                    continue
                new_map_equation = self._calculate_map_equation(adj_matrix, new_map)
                if new_map_equation > best_map_equation:
                    best_map = [m.copy() for m in new_map]
                    best_map_equation = new_map_equation
        return best_map, best_map_equation

    def _infomap(self, adj_matrix):
        """
        Executa o algoritmo Infomap para encontrar a estrutura da comunidade.

        Args:
          adj_matrix: Matriz de adjacência do grafo (tensor PyTorch).

        Returns:
          Estrutura da comunidade (lista de listas).
        """
        num_nodes = adj_matrix.shape[0]
        map = [[i] for i in range(num_nodes)]
        best_map_equation = self._calculate_map_equation(adj_matrix, map)
        for _ in range(self.max_iter):
            map, map_equation = self._move_nodes(adj_matrix, map)
            if abs(map_equation - best_map_equation) < self.tol:
                break
            best_map_equation = map_equation
        return map

    def run(self, label=None, relationship_type=None, projection_name="InfomapProjection"):
        """
        Executa o Infomap no grafo do Neo4j e cria uma projeção.

        Args:
          label: Rótulo do nó (opcional).
          relationship_type: Tipo de relacionamento (opcional).
          projection_name: Nome da projeção a ser criada.
        """
        adj_matrix, node_id_to_index = self._extract_graph_data(label, relationship_type)

        communities = self._infomap(adj_matrix)

        self.graph.run(f"CALL gds.graph.create('{projection_name}', '*', '*')")

        for node_id in node_id_to_index:
            self.graph.run(f"MATCH (n) WHERE id(n) = {node_id} "
                           f"CALL gds.graph.addNode('{projection_name}', id(n), {{}}) "
                           f"YIELD nodeId RETURN nodeId")

        for community_id, community in enumerate(communities):
            for node_id in community:
                for other_node_id in community:
                    if node_id != other_node_id:
                        self.graph.run(f"MATCH (n) WHERE id(n) = {node_id} "
                                       f"MATCH (m) WHERE id(m) = {other_node_id} "
                                       f"CALL gds.graph.addEdge('{projection_name}', id(n), id(m), {{}}) "
                                       f"YIELD relationshipId RETURN relationshipId")

# # Exemplo de uso
# neo4j_infomap = Neo4jInfomap(uri="bolt://localhost:7687", auth=("neo4j", "password"))
# neo4j_infomap.run(label="Pessoa", relationship_type="CONHECE")