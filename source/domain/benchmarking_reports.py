import jinja2
import os

class BenchmarkReportGenerator:
    def __init__(self, template_file):
        """
        Initializes the BenchmarkReportGenerator with a Jinja template file.

        Args:
            template_file: The path to the Jinja template file.
        """
        # Expandir o til para o caminho completo do diretório home
        template_dir = os.path.expanduser("~/ppgcs/source/template/")
        self.template_loader = jinja2.FileSystemLoader(searchpath=template_dir)
        self.template_env = jinja2.Environment(loader=self.template_loader)
        self.template = self.template_env.get_template(template_file)

    def generate_beckmarking_clustering_report(self, benchmark_results, best_model):
        """
        Generates an HTML report using the provided benchmark results and the best model.

        Args:
            benchmark_results: A dictionary containing the benchmarking results for each model.
            best_model: The name of the best model selected based on the benchmarking.
        """

        try:
            # Converter valores para float explicitamente (se necessário)
            for model_name, metrics in benchmark_results.items():
                # Converter 'Tempo de execução' para float, substituindo vírgula por ponto, se necessário
                if isinstance(metrics['Tempo de execução'], str):
                    metrics['Tempo de execução'] = float(metrics['Tempo de execução'].replace(',', '.'))

                # Converter valores em 'Resultados de clustering' para float, se possível
                for algorithm, score in metrics['Resultados de clustering'].items():
                    if isinstance(score, str):
                        try:
                            metrics['Resultados de clustering'][algorithm] = float(score.replace(',', '.'))
                        except ValueError:
                            logging.warning(f"Não foi possível converter o valor '{score}' para float no algoritmo {algorithm} do modelo {model_name}.")
                            metrics['Resultados de clustering'][algorithm] = 'N/A' 

            # Render the template with the provided data
            report_content = self.template.render(
                benchmark_results=benchmark_results,
                best_model=best_model
            )

            # Save the report to an HTML file
            with open("benchmark_report.html", "w", encoding="utf-8") as f:
                f.write(report_content)

            print("Relatório de benchmarking gerado com sucesso: benchmark_report.html")

        except jinja2.TemplateError as e:
            logging.error(f"Erro ao renderizar o template Jinja: {e}")
        except Exception as e:
            logging.error(f"Erro inesperado ao gerar o relatório: {e}")
