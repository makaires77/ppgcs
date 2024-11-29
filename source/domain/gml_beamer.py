from pathlib import Path
import json
import pandas as pd

class AblationLatexTemplate:
    def __init__(self):
        self.preamble = """\\documentclass{beamer}
\\usepackage{tikz}
\\usepackage{amsmath}
\\usepackage{graphicx}
\\usepackage{xcolor}

\\definecolor{class0}{RGB}{65,105,225}
\\definecolor{class1}{RGB}{220,20,60}
"""
    
    def generate_node_classification_slide(self, ablation_results):
        return """\\begin{frame}{Node Classification}
\\begin{columns}
\\column{0.5\\textwidth}
    % Visualização do grafo original
\\column{0.5\\textwidth}
    % Resultados da ablação
    \\begin{itemize}
        \\item Acurácia base: $\\gamma_{ij}$
        \\item Sem atenção: $\\gamma_{ij}$ removido
        \\item Sem features: apenas estrutura do grafo
    \\end{itemize}
\\end{columns}
\\end{frame}"""

    def generate_ablation_results(self, results_dict):
        return """\\begin{frame}{Ablation Study Results}
\\begin{table}
\\begin{tabular}{lcc}
\\toprule
Componente & Acurácia & Diferença \\\\
\\midrule
Modelo Completo & %.3f & - \\\\
Sem Atenção & %.3f & %.3f \\\\
Sem Features & %.3f & %.3f \\\\
\\bottomrule
\\end{tabular}
\\end{table}
\\end{frame}""" % tuple(results_dict.values())

    def generate_comparison_plots(self):
        return """\\begin{frame}{Performance Comparison}
\\begin{figure}
\\includegraphics[width=0.8\\textwidth]{comparison_plot}
\\caption{Comparação entre diferentes configurações do modelo}
\\end{figure}
\\end{frame}"""

    def generate_document(self, ablation_results, comparison_data):
        document = [
            self.preamble,
            "\\begin{document}",
            self.generate_node_classification_slide(ablation_results),
            self.generate_ablation_results(ablation_results),
            self.generate_comparison_plots(),
            "\\end{document}"
        ]
        return "\n".join(document)
    

class AblationLatexGenerator:
    def __init__(self, ablation_dir: str):
        self.ablation_dir = Path(ablation_dir)
        self.study_config = self._load_study_config()
        self.trials_data = self._load_trials_data()
        
    def _load_study_config(self):
        """Carrega configuração do estudo de ablação"""
        config_path = self.ablation_dir / "study.json"
        with open(config_path) as f:
            return json.load(f)
            
    def _load_trials_data(self):
        """Carrega dados dos experimentos"""
        trials_path = self.ablation_dir / "trials.tsv"
        return pd.read_csv(trials_path, sep="\t")
    
    def generate_latex(self):
        """Gera código LaTeX do estudo de ablação"""
        return f"""
\\documentclass{{beamer}}
\\usepackage{{booktabs}}
\\usepackage{{graphicx}}
\\usepackage{{amsmath}}

\\title{{Estudo de Ablação em Graph Machine Learning}}
\\subtitle{{{self.study_config['title']}}}
\\author{{{self.study_config['authors']}}}
\\institute{{Análise de Resultados}}

\\begin{{document}}

{self._generate_intro_slide()}
{self._generate_results_slide()}
{self._generate_comparison_slide()}

\\end{{document}}
"""

    def _generate_intro_slide(self):
        """Gera slide de introdução"""
        return """
\\begin{frame}{Configuração do Estudo}
\\begin{itemize}
\\item Modelo: ComplEx
\\item Dataset: Nations
\\item Métricas: Hits@10
\\item Otimizador: Adam
\\end{itemize}
\\end{frame}
"""

    def _generate_results_slide(self):
        """Gera slide com tabela de resultados"""
        results_table = self._format_results_table()
        return f"""
\\begin{{frame}}{{Resultados dos Experimentos}}
\\begin{{table}}
\\centering
\\begin{{tabular}}{{lcc}}
\\toprule
Configuração & Hits@10 & Desvio Padrão \\\\
\\midrule
{results_table}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
\\end{{frame}}
"""

    def _format_results_table(self):
        """Formata resultados para tabela LaTeX"""
        rows = []
        grouped = self.trials_data.groupby('model')
        for model, data in grouped:
            mean = data['hits@10'].mean()
            std = data['hits@10'].std()
            rows.append(f"{model} & {mean:.3f} & {std:.3f} \\\\")
        return "\n".join(rows)

    def _generate_comparison_slide(self):
        """Gera slide com gráfico comparativo"""
        return """
\\begin{frame}{Comparação de Modelos}
\\begin{figure}
\\includegraphics[width=0.8\\textwidth]{comparison_plot.pdf}
\\caption{Comparação de desempenho entre configurações}
\\end{figure}
\\end{frame}
"""

    def save(self, output_path: str):
        """Salva arquivo LaTeX"""
        with open(output_path, 'w') as f:
            f.write(self.generate_latex())

## Template para demonstrar mecanismo de Atenção
class GraphAttentionSlide:
    def __init__(self):
        self.preamble = """\\documentclass{beamer}
\\usepackage{graphicx}
\\usepackage{amsmath}
\\usepackage{tikz}
\\usepackage{xcolor}

\\definecolor{class0}{RGB}{65,105,225}
\\definecolor{class1}{RGB}{220,20,60}

\\title{Graph Attention Retrospective}
\\author{Kimon Fountoulakis, Amit Levi, Shenghao Yang, \\\\
        Aseem Baranwal, Aukosh Jagannath}
\\institute{University of Waterloo}
"""

    def generate_node_classification(self):
        return """\\begin{frame}{Node Classification}
\\begin{columns}
\\column{0.5\\textwidth}
    \\begin{tikzpicture}[node distance=1cm]
        \\node[draw, circle, fill=class0] (1) at (0,0) {1};
        \\node[draw, circle, fill=class0] (2) at (-1,-1) {2};
        \\node[draw, circle, fill=class0] (3) at (1,-1) {3};
        \\node[draw, circle, fill=class0] (4) at (-1,-2) {4};
        \\node[draw, circle, fill=class0] (5) at (1,-2) {5};
        \\draw (2) -- (3);
        \\draw (4) -- (5);
    \\end{tikzpicture}
\\column{0.5\\textwidth}
    \\begin{tikzpicture}[node distance=1cm]
        \\node[draw, circle, fill=class1] (6) at (0,0) {6};
        \\node[draw, circle, fill=class1] (7) at (-1,-1) {7};
        \\node[draw, circle, fill=class1] (8) at (1,-1) {8};
        \\node[draw, circle, fill=class1] (9) at (-1,-2) {9};
        \\node[draw, circle, fill=class1] (10) at (1,-2) {10};
        \\draw (7) -- (8);
        \\draw (9) -- (10);
    \\end{tikzpicture}
\\end{columns}
\\end{frame}"""

    def generate_data_model(self):
        return """\\begin{frame}{Data Model}
\\begin{itemize}
\\item Draw an edge in class 0 with probability $p$
\\item Draw an edge in class 1 with probability $p$
\\item Draw an edge between classes with probability $q$
\\end{itemize}

Draw features for each node using the Gaussian Mixture Model:
\\[ X_i \\sim \\mathcal{N}(\\mu, \\sigma^2I) \\text{ if node } i \\in \\text{Class}_0 \\]
\\[ X_i \\sim \\mathcal{N}(\\nu, \\sigma^2I) \\text{ if node } i \\in \\text{Class}_1 \\]
\\end{frame}"""

    def generate_graph_attention(self):
        return """\\begin{frame}{Graph Attention}
\\[ X'_3 = (\\gamma_{3,1} \\cdot X_1 + \\gamma_{3,3} \\cdot X_3 + \\gamma_{3,5} \\cdot X_5 + \\gamma_{3,7} \\cdot X_7) \\]
\\[ \\gamma_{ij} = \\frac{\\exp(MLP(X_i, X_j))}{\\sum_{k\\in N_i} \\exp(MLP(X_i, X_k))} \\]
\\end{frame}"""

    def generate_latex(self):
        document = [
            self.preamble,
            "\\begin{document}",
            "\\frame{\\titlepage}",
            self.generate_node_classification(),
            self.generate_data_model(),
            self.generate_graph_attention(),
            "\\end{document}"
        ]
        return "\n".join(document)