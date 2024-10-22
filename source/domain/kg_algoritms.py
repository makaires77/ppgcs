from neo4j import GraphDatabase

class KnowledgeGraph:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
    
    def close(self):
        self.driver.close()

    def create_node(self, tx, node_type, properties):
        query = f"CREATE (n:{node_type} {{{', '.join(f'{k}: ${k}' for k in properties.keys())}}})"
        tx.run(query, **properties)

    def add_data(self, data):
        with self.driver.session() as session:
            for item in data:
                # Processa os casos de uso, vantagens e desvantagens como listas
                item["CASOS DE USO"] = ', '.join(item["CASOS DE USO"])
                item["VANTAGENS"] = ', '.join(item["VANTAGENS"])
                item["DESVANTAGENS"] = ', '.join(item["DESVANTAGENS"])
                session.write_transaction(self.create_node, "Algorithm", item)

if __name__ == "__main__":
    # JSON data a ser persistido
    data = [
        {
            "TIPOS DE APRENDIZADO": "Aprendizado Supervisionado Clássico",
            "ALGORITMO": "Logistic Regression",
            "DESCRIÇÃO": "Algoritmo simples que modela uma relação linear entre as entradas e uma variável de saída categórica (0 ou 1).",
            "CASOS DE USO": ["Predição de risco ao crédito", "Predição de rotatividade de consumidores"],
            "VANTAGENS": ["Facilmente interpretável e explicável", "Menos sujeito ao sobreajuste quando usada a regularização", "Aplicável para predições multi-classe"],
            "DESVANTAGENS": ["Assume a linearidade entre entradas e saídas", "Pode apresentar sobreajuste com poucos dados de alta dimensionalidade"]
        },
        # Adicione mais itens aqui conforme necessário
    ]

    kg = KnowledgeGraph("bolt://localhost:7687", "neo4j", "password")
    kg.add_data(data)
    kg.close()