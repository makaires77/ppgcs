import os
import bibtexparser
from bibtexparser.bwriter import BibTexWriter

class ConversorBib:
    """
    Classe para converter arquivos RIS e NBIB para BibTeX e consolidar arquivos BibTeX existentes.
    """

    def __init__(self, diretorio_entrada, diretorio_saida):
        """
        Inicializa o ConversorBib com os diretórios de entrada e saída.

        Args:
          diretorio_entrada: O caminho para o diretório contendo os arquivos RIS, NBIB e BibTeX.
          diretorio_saida: O caminho para o diretório onde o arquivo BibTeX consolidado será salvo.
        """
        self.diretorio_entrada = diretorio_entrada
        self.diretorio_saida = diretorio_saida
        self.nome_arquivo_saida = 'referencias_consolidadas.bib'  # Nome do arquivo de saída

    def consolidar_arquivos(self):
        """
        Converte todos os arquivos RIS e NBIB no diretório de entrada para BibTeX 
        e consolida todos os arquivos BibTeX em um único arquivo de saída.
        """
        db = bibtexparser.bibdatabase.BibDatabase()  # Cria um objeto BibDatabase para armazenar todas as entradas

        for nome_arquivo in os.listdir(self.diretorio_entrada):
            caminho_entrada = os.path.join(self.diretorio_entrada, nome_arquivo)
            if os.path.isfile(caminho_entrada) and nome_arquivo.endswith(('.ris', '.nbib', '.bib')):
                try:
                    if nome_arquivo.endswith('.bib'):
                        with open(caminho_entrada, 'r') as f:
                            db_temp = bibtexparser.load(f)  # Carrega o arquivo BibTeX existente
                        db.entries.extend(db_temp.entries)  # Adiciona as entradas ao banco de dados principal
                        print(f"Entradas do arquivo '{nome_arquivo}' adicionadas ao arquivo de saída.")
                    else:
                        bibtex = self._converter_arquivo(caminho_entrada)
                        # Converte a string BibTeX em um objeto BibDatabase temporário
                        db_temp = bibtexparser.loads(bibtex)  
                        db.entries.extend(db_temp.entries)  # Adiciona as entradas ao banco de dados principal
                        print(f"Arquivo '{nome_arquivo}' convertido e adicionado ao arquivo de saída.")
                except ValueError as e:
                    print(f"Erro ao processar arquivo '{nome_arquivo}': {e}")
                except Exception as e:
                    print(f"Erro inesperado ao processar arquivo '{nome_arquivo}': {type(e).__name__} - {e}")

        # Grava o arquivo BibTeX consolidado
        caminho_saida = os.path.join(self.diretorio_saida, self.nome_arquivo_saida)
        writer = BibTexWriter()
        with open(caminho_saida, 'w') as f:
            f.write(writer.write(db))
        print(f"Arquivo BibTeX consolidado salvo em '{caminho_saida}'")

    def _converter_arquivo(self, caminho_arquivo):
        """
        Converte um único arquivo RIS ou NBIB para BibTeX.

        Args:
          caminho_arquivo: O caminho para o arquivo RIS ou NBIB.

        Returns:
          Uma string BibTeX.

        Raises:
          ValueError: Se o tipo de arquivo não for suportado ou houver erro na conversão.
        """
        try:
            with open(caminho_arquivo, 'r') as f:
                conteudo = f.read()
            if caminho_arquivo.endswith('.ris'):
                return self._converter_ris(conteudo)
            elif caminho_arquivo.endswith('.nbib'):
                return self._converter_nbib(conteudo)
            else:
                raise ValueError("Tipo de arquivo não suportado.")
        except Exception as e:
            raise ValueError(f"Erro ao converter arquivo '{caminho_arquivo}': {type(e).__name__} - {e}") from e

    def _converter_ris(self, conteudo):
        """
        Converte o conteúdo de um arquivo RIS para BibTeX.
        """
        try:
            # Divide o arquivo RIS em entradas individuais
            entradas_ris = conteudo.split('ER  -')

            # Inicializa uma lista para armazenar as entradas BibTeX
            entradas_bib = []

            # Itera sobre as entradas RIS
            for entrada_ris in entradas_ris:
                if not entrada_ris.strip():
                    continue

                # Cria um dicionário para armazenar os campos BibTeX
                entrada_bib = {}

                # Extrai os campos RIS e mapeia para os campos BibTeX
                for linha in entrada_ris.splitlines():
                    if linha.startswith('TY  -'):
                        tipo_entrada = linha[6:].strip()
                        if tipo_entrada == 'JOUR':
                            entrada_bib['ENTRYTYPE'] = 'article'
                        # Adicione outros tipos de entrada conforme necessário
                    elif linha.startswith('AU  -'):
                        if 'author' not in entrada_bib:
                            entrada_bib['author'] = linha[6:].strip()
                        else:
                            entrada_bib['author'] += ' and ' + linha[6:].strip()
                    elif linha.startswith('PY  -'):
                        entrada_bib['year'] = linha[6:].strip()
                    elif linha.startswith('TI  -'):
                        entrada_bib['title'] = linha[6:].strip()
                    elif linha.startswith('JO  -'):
                        entrada_bib['journal'] = linha[6:].strip()
                    elif linha.startswith('SP  -'):
                        entrada_bib['pages'] = linha[6:].strip().replace('--', '-') # Correção para páginas com "--"
                    elif linha.startswith('VL  -'):
                        entrada_bib['volume'] = linha[6:].strip()
                    elif linha.startswith('IS  -'):
                        entrada_bib['number'] = linha[6:].strip()
                    elif linha.startswith('SN  -'):
                        entrada_bib['issn'] = linha[6:].strip()
                    elif linha.startswith('UR  -'):
                        entrada_bib['url'] = linha[6:].strip().replace(' ', '')
                    elif linha.startswith('DO  -'):
                        entrada_bib['doi'] = linha[6:].strip()
                    # Adicione outros campos RIS conforme necessário

                # Gera uma chave única para a entrada BibTeX (pode ser personalizada)
                chave_bib = entrada_bib.get('author', '').split(',')[0].strip().replace(" ", "_") + entrada_bib.get('year', '')
                entrada_bib['ID'] = chave_bib

                # Adiciona a entrada BibTeX à lista
                entradas_bib.append(entrada_bib)

            # Cria um objeto BibDatabase
            db = bibtexparser.bibdatabase.BibDatabase()
            db.entries = entradas_bib

            # Grava a saída BibTeX
            writer = BibTexWriter()
            return writer.write(db)
        except Exception as e:
            raise ValueError(f"Erro ao converter arquivo RIS: {type(e).__name__} - {e}") from e


    def _converter_nbib(self, conteudo):
        """
        Converte o conteúdo de um arquivo NBIB para BibTeX.
        """
        try:
            # Cria um dicionário para armazenar os campos BibTeX
            entrada_bib = {}

            # Extrai os campos NBIB e mapeia para os campos BibTeX
            for linha in conteudo.splitlines():
                if linha.startswith('PMID-'):
                    entrada_bib['pmid'] = linha[6:].strip()
                elif linha.startswith('TI  -'):
                    entrada_bib['title'] = linha[6:].strip()
                elif linha.startswith('JT  -'):
                    entrada_bib['journal'] = linha[6:].strip()
                elif linha.startswith('DP  -'):
                    # Extrai o ano da data de publicação
                    entrada_bib['year'] = linha[6:].strip().split()[0]
                elif linha.startswith('FAU -'):
                    if 'author' not in entrada_bib:
                        entrada_bib['author'] = linha[6:].strip()
                    else:
                        entrada_bib['author'] += ' and ' + linha[6:].strip()
                elif linha.startswith('VI  -'):
                    entrada_bib['volume'] = linha[6:].strip()
                elif linha.startswith('IP  -'):
                    entrada_bib['number'] = linha[6:].strip()
                elif linha.startswith('PG  -'):
                    entrada_bib['pages'] = linha[6:].strip().replace('-', '--')  # Correção: utiliza '--' como separador de páginas
                elif linha.startswith('LID -'):
                    # Extrai o DOI do campo LID
                    entrada_bib['doi'] = linha[6:].strip().split('[doi]')[0].strip()
                # Adicione outros campos NBIB conforme necessário

            # Define o tipo de entrada como article
            entrada_bib['ENTRYTYPE'] = 'article'

            # Gera uma chave única para a entrada BibTeX (pode ser personalizada)
            chave_bib = entrada_bib.get('author', '').split(',')[0].strip().replace(" ", "_") + entrada_bib.get('year', '')
            entrada_bib['ID'] = chave_bib

            # Cria um objeto BibDatabase
            db = bibtexparser.bibdatabase.BibDatabase()
            db.entries = [entrada_bib]

            # Grava a saída BibTeX
            writer = BibTexWriter()
            return writer.write(db)
        except Exception as e:
            raise ValueError(f"Erro ao converter arquivo NBIB: {type(e).__name__} - {e}") from e