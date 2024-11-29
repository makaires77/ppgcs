from pathlib import Path
import json
import pandas as pd

class AblationLatexTemplate:
    def __init__(self, ablation_results_dir: str = ''):
        self.preamble = """\\documentclass{beamer}
\\usetheme{Madrid}
\\usepackage{tikz}
\\usepackage{amsmath}
\\usepackage{booktabs}
\\usepackage{graphicx}
\\usepackage{xcolor}

\\definecolor{class0}{RGB}{65,105,225}
\\definecolor{class1}{RGB}{220,20,60}
"""
        self.ablation_dir = Path(ablation_results_dir)
        self.study_config = self._load_study_config()
        self.trials_data = self._load_trials_data()

    def _load_study_config(self):
        """Carrega configuração do estudo de ablação do PyKEEN"""
        config_path = self.ablation_dir / "study_base.json"
        with open(config_path) as f:
            return json.load(f)

    def _load_trials_data(self):
        """Carrega resultados dos experimentos do PyKEEN"""
        trials_path = self.ablation_dir / "trials_base.tsv"
        return pd.read_csv(trials_path, sep="\t")

    def generate_title_slide(self):
        """Gera slide de título"""
        return f"""\\begin{{frame}}
\\titlepage
\\end{{frame}}"""

    def generate_model_slide(self):
        """Gera slide de classificação do modelo"""
        return """\\begin{frame}{Node Classification}
\\begin{columns}
\\column{0.5\\textwidth}
    \\begin{tikzpicture}[node distance=1cm]
        % Visualização do modelo base
        \\node[draw, circle, fill=class0] (1) at (0,0) {1};
        \\node[draw, circle, fill=class0] (2) at (1,0) {2};
        \\draw (1) -- (2);
    \\end{tikzpicture}
\\column{0.5\\textwidth}
    \\begin{itemize}
        \\item Modelo: {self.study_config['models'][0]}
        \\item Dataset: {self.study_config['datasets'][0]}
        \\item Loss: {self.study_config['losses'][0]}
    \\end{itemize}
\\end{columns}
\\end{frame}"""

    def generate_data_model_slide(self):
        """Gera slide do modelo de dados"""
        return """\\begin{frame}{Data Model}
\\begin{itemize}
\\item Dataset: {self.study_config['datasets'][0]}
\\item Features: Gaussian Mixture Model
\\item Training Samples: {len(self.trials_data)}
\\end{itemize}

\\[ X_i \\sim \\mathcal{N}(\\mu, \\sigma^2I) \\text{ if node } i \\in \\text{Class}_0 \\]
\\[ X_i \\sim \\mathcal{N}(\\nu, \\sigma^2I) \\text{ if node } i \\in \\text{Class}_1 \\]
\\end{frame}"""

    def generate_results_slide(self):
        """Gera slide de resultados da ablação"""
        results_table = self._format_results_table()
        return f"""\\begin{{frame}}{{Ablation Results}}
\\begin{{table}}
\\centering
\\begin{{tabular}}{{lcc}}
\\toprule
Configuration & Hits@10 & Loss \\\\
\\midrule
{results_table}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
\\end{{frame}}"""

    def _format_results_table(self):
        """Formata resultados para tabela LaTeX"""
        rows = []
        grouped = self.trials_data.groupby('model')
        for model, data in grouped:
            hits = data['hits@10'].mean()
            loss = data['loss'].mean()
            rows.append(f"{model} & {hits:.3f} & {loss:.3f} \\\\")
        return "\n".join(rows)

    def generate_latex(self):
        """Gera documento LaTeX completo"""
        document = [
            self.preamble,
            "\\begin{document}",
            self.generate_title_slide(),
            self.generate_model_slide(),
            self.generate_data_model_slide(),
            self.generate_results_slide(),
            "\\end{document}"
        ]
        return "\n".join(document)

    def save(self, output_path: str):
        """Salva arquivo LaTeX"""
        with open(output_path, 'w') as f:
            f.write(self.generate_latex())


from IPython.display import display, Latex, HTML, FileLink
import subprocess

class AblationLatexPresenter:
    def __init__(self, ablation_results_dir: str):
        self.ablation_dir = Path(ablation_results_dir)
        self.output_file = None
        
    def compile_latex(self, tex_file: str):
        """Compila o arquivo LaTeX para PDF"""
        try:
            subprocess.run(['pdflatex', tex_file], check=True)
            self.output_file = tex_file.replace('.tex', '.pdf')
            return True
        except subprocess.CalledProcessError:
            print("Erro na compilação do LaTeX")
            return False
            
    def render_preview(self, tex_file: str):
        """Mostra preview do código LaTeX no notebook"""
        with open(tex_file, 'r') as file:
            content = file.read()
            display(Latex(content))
            
    def render_pdf(self):
        """Exibe o PDF no notebook"""
        if self.output_file and Path(self.output_file).exists():
            display(HTML(f'<iframe src="{self.output_file}" width="100%" height="600"></iframe>'))
            
    def download_link(self):
        """Cria link para download do PDF"""
        if self.output_file:
            return FileLink(self.output_file)
            
    def render_presentation(self, tex_file: str):
        """Pipeline completo de renderização"""
        # Preview do LaTeX
        self.render_preview(tex_file)
        
        # Compilação
        if self.compile_latex(tex_file):
            # Exibição do PDF
            self.render_pdf()
            # Link para download
            return self.download_link()


## Versão mais simples
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