import json
import os
from collections import Counter
from itertools import combinations

class JSONFileManager:
    data_folder = '/home/mak/gml_classifier-1/data/output/'

    @staticmethod
    def load_json(path_folder, file):
        if path_folder is None:
            path_folder = '/home/mak/gml_classifier-1/data/output/'
        filepath = os.path.join(path_folder,file)
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)

    @staticmethod
    def check_essential_data(file, essential_keys):
        """
        Verifica se os dados essenciais estão presentes no arquivo JSON.
        
        Parâmetros:
            file (str): Caminho do arquivo JSON a ser verificado.
            essential_keys (list): Lista de chaves essenciais que devem estar presentes nos dicionários do arquivo JSON.
        
        Retorna:
            bool: Retorna True se todos os dados essenciais estiverem presentes, False caso contrário.
        """
        path_folder = '/home/mak/gml_classifier-1/data/output/'
        data = JSONFileManager.load_json(path_folder, file)
        if isinstance(data, list):  # Considera que o arquivo JSON contém uma lista de dicionários
            print(f"Checando: {file}")
            for i in data:
                for item in i.get('processed_data'):
                    # if not all(key in item for key in essential_keys):
                    #     return False
                    # else:
                    print(['ok' if item in essential_keys else 'Elemento não essencial'],item)
                return True
        elif isinstance(data, dict):  # Considera um único dicionário no arquivo JSON
            return all(key in data for key in essential_keys)
        else:
            print(type(data), len(data))
            print(type(i), len(i))
        return False  # Se os dados não são nem lista nem dicionário

    @staticmethod
    def compare_dictionaries(dict1, dict2, path=data_folder):
        differences = []
        for key in set(dict1.keys()).union(dict2.keys()):
            if key in dict1 and key not in dict2:
                differences.append(f"    Divergência em {path}: Chave '{key}' presente no primeiro dicionário, mas ausente no segundo")
            elif key not in dict1 and key in dict2:
                differences.append(f"    Divergência em {path}: Chave '{key}' presente no segundo dicionário, mas ausente no primeiro")
            elif dict1.get(key) != dict2.get(key):
                differences.append(f"    Divergência em {path}: Diferença no valor da chave '{key}', {dict1.get(key)} != {dict2.get(key)}")
        return differences

    @staticmethod
    def compare_lists_of_dictionaries(list1, list2, path=data_folder, show_details=False):
        differences = []
        # Transforma as listas em conjuntos de elementos únicos para comparação
        set1 = {json.dumps(d, sort_keys=True) for d in list1}
        set2 = {json.dumps(d, sort_keys=True) for d in list2}

        unique_in_list1 = set1 - set2
        unique_in_list2 = set2 - set1

        if show_details:
            if len(unique_in_list1) == len(unique_in_list2):
                print(f"    Quantidade de dicionários idênticas nos dois JSON: {len(unique_in_list1)}")
            else:
                print(f"    Quantidade de dicionários diferentes nos dois JSON:")
                print(f"    {len(unique_in_list1):>2} dicionário(s) únicos na primeira lista")
                print(f"    {len(unique_in_list2):>2} dicionário(s) únicos na segunda lista")

            if unique_in_list1 or unique_in_list2:
                differences.append(unique_in_list1)
                differences.append(unique_in_list2)

                print(f"    Divergência no conteúdo das listas em {path}:")
                for i in differences:
                    print(i)

        return differences

    @staticmethod
    def list_json_files(folder):
        json_files = [f for f in os.listdir(folder) if f.endswith('.json')]
        return json_files

    @staticmethod
    def count_items_in_json(data, path=''):
        """
        Conta recursivamente o número de itens em cada chave de um dicionário ou lista em um arquivo JSON,
        agregando a contagem de elementos atômicos quando todos são do mesmo tipo.
        Parâmetros:
            data (dict, list, ou outro tipo): Dados extraídos de um arquivo JSON.
            path (str): Caminho utilizado para rastrear a posição no JSON durante a recursão.
        Retorna:
            dict: Dicionário com o caminho de cada chave, o número de itens nela, e o tipo de dado.
        """
        def count_list_items(lst, current_path):
            """ Conta os elementos de uma lista, incluindo listas e dicionários aninhados. """
            if not lst:
                return {current_path: "0 listas vazias"}
            elif all(isinstance(item, list) for item in lst):
                # Conta os itens de listas aninhadas
                # print("Lista de listas encontrada...")
                count_dict = {}
                for index, sublist in enumerate(lst):
                    new_path = f"{current_path}[{index}]"
                    count_dict.update(count_list_items(sublist, new_path))
                return count_dict
            elif all(isinstance(item, dict) for item in lst):
                # Continua a recursão para listas de dicionários
                # print("Dicionário de dicionários encontrado...")
                count_dict = {}
                for index, item in enumerate(lst):
                    new_path = f"{current_path}[{index}]"
                    count_dict.update(JSONFileManager.count_items_in_json(item, new_path))
                return count_dict
            else:
                # Conta itens atômicos e continua a recursão para listas e dicionários misturados
                # print("Nível de itens atômicos encontrado")
                if all(type(item) is type(lst[0]) for item in lst):  # Todos os itens são do mesmo tipo
                    item_type = type(lst[0]).__name__
                    return {current_path: f"{len(lst)} elementos {item_type}"}
                else:
                    count_dict = {}
                    for item in lst:
                        item_type = type(item).__name__
                        count_dict[item_type] = count_dict.get(item_type, 0) + 1
                    return {current_path: ', '.join(f"{count} elementos {item_type}" for item_type, count in count_dict.items())}

        item_type = type(data).__name__

        if isinstance(data, dict):
            count_dict = {path: f"{len(data)} dicionários"} if path else {}
            for key, value in data.items():
                new_path = f"{path}/{key}" if path else key
                count_dict.update(JSONFileManager.count_items_in_json(value, new_path))
            return count_dict
        elif isinstance(data, list):
            return count_list_items(data, path)
        else:
            return {path: f"1 {item_type}"}

    @classmethod
    def list_relevant_json_files(cls):
        relevant_files = [f for f in os.listdir(cls.data_folder) if f.startswith('output_') and f.endswith('.json')]
        ordenacoes = [True, False, True, False]
        relevant_files = cls.complex_sort(cls,
                                          relevant_files, 
                                           "_", 
                                           ordenacoes)
        return relevant_files

    @classmethod
    def compare_all_json_pairs(cls, show_details=False):
        count = 0
        files = cls.list_relevant_json_files()
        print(f"Analisando arquivos na pasta: {cls.data_folder}")
        print(f"Arquivos analisados:")
        for i in files:
            print(f"  {i}")
        print()
        total_comparisons = 0
        identical_files = []
        divergent_files = []
        try:
            for file1, file2 in combinations(files, 2):
                total_comparisons += 1
                dataset1 = cls.load_json(cls.data_folder, file1)
                dataset2 = cls.load_json(cls.data_folder, file2)
                differences = JSONFileManager.compare_lists_of_dictionaries(dataset1, dataset2, show_details)

                if len(differences) == 0:
                    print(f"Par {total_comparisons:>2}: {len(dataset1)} dicionários nos dois JSONs comparados")
                    identical_files.append((file1, file2))
                else:
                    divergent_files.append((file1, file2))
                    if show_details:
                        print(f"Comparando: {file1} e {file2}")
                        print(f"Encontradas {len(differences)} diferenças entre {file1} e {file2}.")
                        for diff in differences:
                            print(diff)

            # Exibir resumo ao final
            print("\nTotal de arquivos de datasets:", len(files))
            print("\nResumo das Comparações:")
            print(f"Total de comparações por pares realizadas: {total_comparisons}")
            print(f"Quantidade de pares de arquivos idênticos: {len(identical_files)}")
            if identical_files:
                print("Pares idênticos:")
                for file_pair in identical_files:
                    print("   '", file_pair[0], "'")
                    print("   '", file_pair[1], "'")
                    print()

            print(f"Quantidade de pares de arquivos divergentes: {len(divergent_files)}")
            if divergent_files:
                for file_pair in divergent_files:
                    print("   '", file_pair[0], "'")
                    print("   '", file_pair[1], "'")
                    print()

            cls.infer_problematic_file(divergent_files)
        except Exception as e:
            print("Erro ao comparar arquivos JSON")
            print(e)
    
    @classmethod
    def infer_problematic_file(cls, divergent_files):
        file_counts = Counter(file for pair in divergent_files for file in pair)
        most_common = file_counts.most_common(1)

        if most_common:
            problematic_file, count = most_common[0]
            print(f"\nVerifique o arquivo abaixo, esteve presente em {count} pares divergentes:")
            print(f"    '{problematic_file}'")
        else:
            print("\nNão foi possível inferir um arquivo problemático.")

    @classmethod
    def compare_specific_files(cls, file1, file2, show_details=True):
        print(f"Comparando: {file1} e {file2}")
        dataset1 = JSONFileManager.load_json(cls.data_folder, file1)
        dataset2 = JSONFileManager.load_json(cls.data_folder, file2)
        differences = JSONFileManager.compare_lists_of_dictionaries(dataset1, dataset2)
        num_differences = len(differences)

        if num_differences == 0:
            print(f"Os arquivos {file1} e {file2} são idênticos.")
        else:
            print(f"Encontradas {num_differences} diferenças entre {file1} e {file2}.")

            if show_details:
                print("    Detalhes das diferenças:")
                for diff in differences:
                    print(f"    {diff}")

    def complex_sort(cls, lst, separator, order_list):
        """
        Ordena uma lista com base em substrings identificados por um separador, seguindo uma ordem específica.
        Adiciona validação para lidar com listas que têm mais separadores do que elementos em order_list.
        
        :param lst: Lista de strings a ser ordenada.
        :param separator: Caractere usado como separador para identificar substrings.
        :param order_list: Lista de booleanos, onde True indica ordem crescente e False decrescente para cada substring.
        :return: Lista ordenada conforme as regras especificadas.
        """

        # Função auxiliar para comparar elementos com base nas regras de ordenação
        def compare_items(item):
            # Divide o item nas substrings
            parts = item.split(separator)

            # Prepara as substrings para comparação
            compared_parts = []
            for i, part in enumerate(parts):
                # Usa a ordem especificada se disponível, caso contrário, usa ordem crescente
                order = order_list[i] if i < len(order_list) else True

                # Inverte a ordem se necessário
                compared_parts.append(part if order else -ord(part[0]) if part else 0)

            return compared_parts

        # Ordena a lista usando a função de comparação
        return sorted(lst, key=compare_items)
