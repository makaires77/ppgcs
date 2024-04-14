# !pip install PyPDF2
import os, logging
import re
import pandas as pd
from PyPDF2 import PdfReader
from tqdm.notebook import trange, tqdm
from json_fle_manager import JSONFileManager as jfm

class CNPQtree:
    def __init__(self, base_repo_dir):
        self.base_repo_dir = base_repo_dir
        self.folder_utils = os.path.join(base_repo_dir, 'utils')
        self.folder_domain = os.path.join(base_repo_dir, 'source', 'domain')
        self.folder_data_input = os.path.join(base_repo_dir, 'data', 'input')
        self.folder_data_output = os.path.join(base_repo_dir, 'data', 'output')
        self.caminho = os.path.join(self.folder_data_input,'cnpq_tabela-areas-conhecimento.pdf')

    def verifica_ponto_virgula(self, df):
        return df[df['Descricao'].str.contains(';', regex=False)]

    def verifica_virgula(self, df):
        return df[df['Descricao'].str.contains(',', regex=False)]

    def verifica_formato_descricao(self, descricao):
        excecoes = ["de", "do", "da", "dos", "das", "a", "o", "e", "em", "com", "para", "por", "sem"]
        palavras = descricao.split()
        
        for i, palavra in enumerate(palavras):
            if palavra.lower() in excecoes or palavra[0]=="(":
                continue
            if not palavra[0].isupper() or (palavra==palavras[-1] and palavra in excecoes):
                return False, i  # Retornar False e o índice da palavra problemática
        return True, None

        # for idx, word in enumerate(palavras):
        #     # Se a palavra inicia com letra minúscula e não é uma preposição ou artigo
        #     if word[0].islower() and word not in excecoes:
        #         # Aqui verificamos se a palavra anterior termina com uma letra e a palavra atual é uma preposição ou artigo
        #         if idx > 0 and palavras[idx - 1][-1].isalpha() and word in excecoes:
        #             return (False, idx)
        #         # Ou apenas a condição de começar com minúscula e não ser preposição ou artigo
        #         elif idx == 0 or (idx > 0 and not palavras[idx - 1][-1].isalpha()):
        #             return (False, idx)    

    def corrigir_descricao(self, descricao, word_index):
        excecoes = ["de", "do", "da", "dos", "das", "a", "o", "e", "em", "com", "para", "por", "sem"]
        palavras = descricao.split()

        # Se o índice anterior existir e a palavra atual começa com minúscula
        if word_index > 0 and palavras[word_index][0].islower():
            # Checar se a palavra é uma preposição ou artigo e se a anterior termina com uma letra
            if palavras[word_index] in excecoes:
                palavras[word_index - 1] += palavras[word_index]
                del palavras[word_index]
            else:
                # Juntar palavra atual com a palavra anterior
                palavras[word_index - 1] += palavras[word_index]
                del palavras[word_index]

        # Após as correções, juntamos as palavras de volta em uma única string
        nova_descricao = ' '.join(palavras)

        # Imprimindo para debug
        # print(f"Descrição ruim: {descricao}")
        # print(f"Correção feita: {palavra_anterior} + {palavra_incorreta} = {correcao}")
        # print(f"Nova descrição: {nova_descricao}\n")
        
        return nova_descricao


    def extrair_areas(self, caminho):
        texto_completo = ""

        reader = PdfReader(caminho)
        
        for npag, p in tqdm(enumerate(reader.pages), total=len(reader.pages), desc="Processando páginas do PDF das Áreas de pesquisa do CNPq.."):
            texto_completo += p.extract_text()

        texto_completo = texto_completo.replace('\n', ' ').replace(" -","-").replace(" ,",",").strip().replace("ã o","ão")
        texto_completo = re.sub(r'\s?(\d)\s?(\.)\s?(\d{2})\s?(\.)\s?(\d{2})\s?(\.)\s?(\d{2})\s?(-)\s?(\d)\s?', r'\1\2\3\4\5\6\7\8\9', texto_completo)

        pattern = r'(\d\.\d{2}\.\d{2}\.\d{2}-\d)([^0-9]+)'
        matches = re.findall(pattern, texto_completo)

        codigos = [match[0] for match in matches]
        descricoes = [match[1].strip() for match in matches]

        print(f'Total dos códigos   identificados: {len(codigos)}')
        print(f'Total de descrições identificadas: {len(descricoes)}')

        df_linhas = pd.DataFrame({'Codigo': codigos, 'Descricao': descricoes})

        # Verificação da divisão correta dos códigos/descrições
        descricoes_com_numeros = df_linhas[df_linhas['Descricao'].str.contains(r'\d')]
        if not descricoes_com_numeros.empty:
            print(f"Conferência: {len(descricoes_com_numeros)} descrições contêm números!")
        else:
            print(f"Nenhum erro de códigos em descrições!")

        # Identificando e printando a quantidade de possíveis erros
        erros = sum(1 for descricao in descricoes if not self.verifica_formato_descricao(descricao)[0])
        print(f"{erros} possíveis erros de descrição detectados.")

        # Barra de progresso para correção das descrições
        with tqdm(total=df_linhas.shape[0], desc="Corrigindo descrições...") as pbar:
            for index, row in df_linhas.iterrows():
                is_valid, word_index = self.verifica_formato_descricao(row['Descricao'])
                loop_count = 0
                while not is_valid and loop_count < 10:
                    row['Descricao'] = self.corrigir_descricao(row['Descricao'], word_index)
                    is_valid, word_index = self.verifica_formato_descricao(row['Descricao'])
                    loop_count += 1
                if loop_count == 10:
                    print(f"Problema corrigindo descrição: {row['Descricao']}")
                pbar.update(1)

        return df_linhas
    
    def count_unique_for_level(self, level: int):
        df_areas = self.extrair_areas(self.caminho)
        return df_areas['Codigo'].str.split('.', expand=True)[level].nunique()

    # Remover o sufixo após o hífen
    def count_unique_for_last_level(self):
        df_areas = self.extrair_areas(self.caminho)
        return df_areas['Codigo'].str.split('.', expand=True).iloc[:, -1].str.split('-').str[0].nunique()

    # df_areas = self.extrair_areas(self.caminho)
    # levels = df_areas['Codigo'].str.count("\.").iloc[0]  # conta a quantidade de pontos, para determinar o número de níveis

    # unique_counts = [self.count_unique_for_level(i) for i in range(levels)]
    # unique_counts.append(count_unique_for_last_level())

    # qte_grandeareas, qte_areas, qte_subareas, qte_especialidades = unique_counts
    # print(f'Quantidades de codigos:')
    # print(f'  Grande_Área: {qte_grandeareas:2}')
    # print(f'         Área: {qte_areas:2}')
    # print(f'      Subárea: {qte_subareas:2}')
    # print(f'Especialidade: {qte_especialidades:2}')

    # Dividir a coluna 'Codigo' em várias colunas
    # df_split = df_areas['Codigo'].str.split('.', expand=True)

    # Remover o último hífen e dígito das colunas 
    # df_split.iloc[:, -1] = df_split.iloc[:, -1].str.split('-').str[0]

    # def count_sublevels(df, level):
    #     if level == 0:
    #         return df[0].nunique()
    #     else:
    #         return df.groupby(list(range(level)))[level].nunique().reset_index(name="count")["count"].to_list()

    # sublevels_counts = [count_sublevels(df_split, i) for i in range(df_split.shape[1])]

    # Criar uma coluna para armazenar a contagem de subníveis
    # df_areas['SublevelCount'] = df_split.apply(lambda row: [sublevels_counts[col][row[:col].astype(str).tolist().index(row[col-1]) if row[col-1] in row[:col].astype(str).tolist() else -1] if col > 0 else sublevels_counts[col] for col in df_split.columns], axis=1)