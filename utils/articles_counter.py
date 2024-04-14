import re
import logging
import pandas as pd
from datetime import datetime, timedelta

class ArticlesCounter:
    def __init__(self, dict_list):
        self.data_list = dict_list

    def dias_desde_atualizacao(self, data_atualizacao_str):
        # Converte a data de atualização em um objeto datetime
        data_atualizacao = datetime.strptime(data_atualizacao_str, '%d/%m/%Y')
        
        # Obtém a data atual
        data_atual = datetime.now()
        
        # Calcula a diferença em dias
        diferenca_dias = (data_atual - data_atualizacao).days if data_atualizacao else None
        return diferenca_dias

    def extrair_data_atualizacao(self, dict_list):
        ids_lattes_grupo=[]
        nomes_curriculos=[]
        dts_atualizacoes=[]
        tempos_defasagem=[]
        qtes_artcomplper=[]

        for index, dic in enumerate(dict_list):
            try:
                info_nam = dic.get('name',{})
                nomes_curriculos.append(info_nam)
                info_pes = dic.get('InfPes', {})
                if type(info_pes) == dict:
                    processar = info_pes.values()
                elif type(info_pes) == list:
                    processar = info_pes
                for line in processar:
                    try:
                        id_pattern = re.search(r'ID Lattes: (\d+)', line)
                        dt_pattern = re.search(r'\d{2}/\d{2}/\d{4}', line)
                        id_lattes =  id_pattern.group(1) if id_pattern else None
                        if id_lattes:
                            ids_lattes_grupo.append(id_lattes)
                        data_atualizacao = dt_pattern.group() if dt_pattern else None
                        if data_atualizacao:
                            dts_atualizacoes.append(data_atualizacao)
                            tempo_atualizado = self.dias_desde_atualizacao(data_atualizacao)
                            tempos_defasagem.append(tempo_atualizado)                    
                    except Exception as e:
                        pass
                        # print(e)

                info_art = dic.get('Produções', {}).get('Produção bibliográfica', {}).get('Artigos completos publicados em periódicos', {})
                qtes_artcomplper.append(len(info_art.values()))
            except Exception as e:
                logging.error(f"Erro no dicionário {index}: {e}")
                logging.error(f"Dicionário com problema: {dic}")
                continue

        dtf_atualizado = pd.DataFrame({"id_lattes": ids_lattes_grupo,
                                       "curriculos": nomes_curriculos, 
                                       "ultima_atualizacao": dts_atualizacoes,
                                       "dias_defasagem": tempos_defasagem,
                                       "qte_artigos_periodicos": qtes_artcomplper,
                                       })
        return dtf_atualizado