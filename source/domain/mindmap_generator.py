import json
import pydot
import re
from collections import Counter

class MindmapGenerator:
    def __init__(self, json_file, bib_file=None):
        self.data = self.load_json(json_file)
        self.graph = pydot.Dot(graph_type="graph", rankdir="LR")
        self.bib_file = bib_file

    def load_json(self, json_file):
        with open(json_file, 'r', encoding='utf-8') as file:
            return json.load(file)

    def add_nodes_and_edges(self, data, parent_node=None):
        for key, value in data.items():
            node = pydot.Node(key)
            self.graph.add_node(node)
            if parent_node:
                self.graph.add_edge(pydot.Edge(parent_node, node))

            if isinstance(value, dict):
                self.add_nodes_and_edges(value, node)
            elif isinstance(value, list):
                for item in value:
                    item_node = pydot.Node(item)
                    self.graph.add_node(item_node)
                    self.graph.add_edge(pydot.Edge(node, item_node))

    def generate_mindmap(self, output_file):
        self.add_nodes_and_edges(self.data)
        self.graph.write_png(output_file)

    def read_bib_file(self):
        """Reads a BibTeX file and extracts the type of each entry.

        Returns:
        list: A list of entry types.
        """
        if not self.bib_file:
            raise ValueError("BibTeX file not provided.")

        with open(self.bib_file, 'r') as f:
            bibtex = f.read()

        entry_types = []
        for entry in re.findall(r'@[a-zA-Z]+\{.*?\}', bibtex, re.DOTALL):
            entry_type = re.search(r'@([a-zA-Z]+)\{', entry).group(1)
            entry_types.append(entry_type)

        return entry_types

    def count_entry_types(self):
        """Counts the number of each entry type and returns a pandas DataFrame.

        Returns:
        pandas.DataFrame: A DataFrame with the entry types and their counts.
        """
        entry_types = self.read_bib_file()
        type_counts = Counter(entry_types)
        df = pd.DataFrame(type_counts.items(), columns=['Tipo de Entrada', 'Contagem'])
        return df

    def generate_latex_table(self):
        """Displays the DataFrame in LaTeX format."""
        if not self.bib_file:
            raise ValueError("BibTeX file not provided.")
        df_counts = self.count_entry_types()
        latex_table = df_counts.to_latex(index=False, escape=False)
        print(latex_table)

# Exemplo de uso
# if __name__ == "__main__":
#     json_file = "algorithms.json"  # Nome do arquivo JSON
#     output_file = "mindmap.png"  # Nome do arquivo de saída

#     mindmap_generator = MindmapGenerator(json_file)
#     mindmap_generator.generate_mindmap(output_file)