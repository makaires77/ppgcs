import re
import torch
import torch.nn as nn
from py2neo import Graph
from jsonschema import validate
from sentence_transformers import SentenceTransformer


class KGNN(torch.nn.Module):

    def __init__(self, embedding_model_name, neo4j_uri, neo4j_user, neo4j_password):
        super().__init__()

        # Inicializa o modelo de embedding
        self.embedding_model = SentenceTransformer(embedding_model_name)

        # Inicializa a conexão com o Neo4j
        self.graph = Graph(neo4j_uri, auth=(neo4j_user, neo4j_password))

    def _formatar_propriedades(self, propriedades):
        """
        Formata as propriedades para a query Cypher, removendo espaços
        e caracteres inválidos.
        """
        propriedades_formatadas = []
        for chave, valor in propriedades.items():
            # Remove espaços e caracteres especiais da chave
            chave = re.sub(r"[^a-zA-Z0-9_]", "", chave)

            # Escapa aspas duplas em strings
            if isinstance(valor, str):
                valor = valor.replace('"', '\\"')
                valor = f'"{valor}"'
            propriedades_formatadas.append(f"{chave}: {valor}")
        return ", ".join(propriedades_formatadas)


    def _corrigir_nome_propriedade(self, propriedades):
        """
        Corrige a propriedade 'nome' para '`nome`' nos nós 'Orientacao'.
        """
        if isinstance(propriedades, dict):
            for chave, valor in propriedades.copy().items():
                if chave == 'nome':
                    propriedades["nome"] = propriedades.pop('nome')  # Escapa a chave 'nome'
                elif chave == 'orientacoes':
                    propriedades['orientacoes'] = propriedades.pop('orientacoes')
                # Chamada recursiva para corrigir em sub-dicionários
                propriedades[chave] = self._corrigir_nome_propriedade(valor)
        elif isinstance(propriedades, list):
            # Chamada recursiva para corrigir em sub-listas
            for i in range(len(propriedades)):
                propriedades[i] = self._corrigir_nome_propriedade(propriedades[i])
        return propriedades


    def _validar_dados(self, curriculo_dict):
        """
        Valida os dados do currículo contra o schema JSON.
        """
        schema = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "Identifica\u00e7\u00e3o": {
                        "type": "object",
                        "properties": {
                            "Nome": {
                                "type": "string"
                            },
                            "ID Lattes": {
                                "type": "string"
                            },
                            "\u00daltima atualiza\u00e7\u00e3o": {
                                "type": "string"
                            }
                        },
                        "required": [
                            "Nome",
                            "ID Lattes",
                            "\u00daltima atualiza\u00e7\u00e3o"
                        ]
                    },
                    "Idiomas": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "Idioma": {
                                    "type": "string"
                                },
                                "Profici\u00eancia": {
                                    "type": "string"
                                }
                            },
                            "required": [
                                "Idioma",
                                "Profici\u00eancia"
                            ]
                        }
                    },
                    "Forma\u00e7\u00e3o": {
                        "type": "object",
                        "properties": {
                            "Acad\u00eamica": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "Ano": {
                                            "type": "string"
                                        },
                                        "Descri\u00e7\u00e3o": {
                                            "type": "string"
                                        }
                                    },
                                    "required": [
                                        "Ano",
                                        "Descri\u00e7\u00e3o"
                                    ]
                                }
                            },
                            "Pos-Doc": {
                                "type": "array"
                            },
                            "Complementar": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "Ano": {
                                            "type": "string"
                                        },
                                        "Descri\u00e7\u00e3o": {
                                            "type": "string"
                                        }
                                    },
                                    "required": [
                                        "Ano",
                                        "Descri\u00e7\u00e3o"
                                    ]
                                }
                            }
                        },
                        "required": [
                            "Acad\u00eamica",
                            "Pos-Doc",
                            "Complementar"
                        ]
                    },
                    "Atua\u00e7\u00e3o Profissional": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "Institui\u00e7\u00e3o": {
                                    "type": "string"
                                },
                                "Ano": {
                                    "type": "string"
                                },
                                "Descri\u00e7\u00e3o": {
                                    "type": "string"
                                },
                                "Outras informa\u00e7\u00f5es": {
                                    "type": "string"
                                }
                            },
                            "required": [
                                "Institui\u00e7\u00e3o",
                                "Ano",
                                "Descri\u00e7\u00e3o",
                                "Outras informa\u00e7\u00f5es"
                            ]
                        }
                    },
                    "Linhas de Pesquisa": {
                        "type": "array"
                    },
                    "\u00c1reas": {
                        "type": "object",
                        "patternProperties": {
                            "^[0-9]+\\.$": {
                                "type": "string"
                            }
                        }
                    },
                    "Produ\u00e7\u00f5es": {
                        "type": "object",
                        "properties": {
                            "Artigos completos publicados em peri\u00f3dicos": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "ano": {
                                            "type": "string"
                                        },
                                        "fator_impacto_jcr": {
                                            "type": "string"
                                        },
                                        "ISSN": {
                                            "type": "string"
                                        },
                                        "titulo": {
                                            "type": "string"
                                        },
                                        "revista": {
                                            "type": "string"
                                        },
                                        "autores": {
                                            "type": "string"
                                        },
                                        "data_issn": {
                                            "type": "string"
                                        },
                                        "DOI": {
                                            "type": "string"
                                        },
                                        "Qualis": {
                                            "type": "string"
                                        }
                                    },
                                    "required": [
                                        "ano",
                                        "fator_impacto_jcr",
                                        "ISSN",
                                        "titulo",
                                        "revista",
                                        "autores",
                                        "data_issn",
                                        "DOI",
                                        "Qualis"
                                    ]
                                }
                            },
                            "Resumos publicados em anais de congressos": {
                                "type": "object",
                                "patternProperties": {
                                    "^[0-9]+\\.$": {
                                        "type": "string"
                                    }
                                }
                            },
                            "Apresenta\u00e7\u00f5es de Trabalho": {
                                "type": "object",
                                "patternProperties": {
                                    "^[0-9]+\\.$": {
                                        "type": "string"
                                    }
                                }
                            },
                            "Outras produ\u00e7\u00f5es bibliogr\u00e1ficas": {
                                "type": "object",
                                "patternProperties": {
                                    "^[0-9]+\\.$": {
                                        "type": "string"
                                    }
                                }
                            },
                            "Entrevistas, mesas redondas, programas e coment\u00e1rios na m\u00eddia": {
                                "type": "object",
                                "patternProperties": {
                                    "^[0-9]+\\.$": {
                                        "type": "string"
                                    }
                                }
                            },
                            "Demais tipos de produ\u00e7\u00e3o t\u00e9cnica": {
                                "type": "object",
                                "patternProperties": {
                                    "^[0-9]+\\.$": {
                                        "type": "string"
                                    }
                                }
                            }
                        },
                        "required": [
                            "Artigos completos publicados em peri\u00f3dicos"
                        ]
                    },
                    "ProjetosPesquisa": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "chave": {
                                    "type": "string"
                                },
                                "titulo_projeto": {
                                    "type": "string"
                                },
                                "descricao": {
                                    "type": "string"
                                }
                            },
                            "required": [
                                "chave",
                                "titulo_projeto",
                                "descricao"
                            ]
                        }
                    },
                    "ProjetosExtens\u00e3o": {
                        "type": "array"
                    },
                    "ProjetosDesenvolvimento": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "chave": {
                                    "type": "string"
                                },
                                "titulo_projeto": {
                                    "type": "string"
                                },
                                "descricao": {
                                    "type": "string"
                                }
                            },
                            "required": [
                                "chave",
                                "titulo_projeto",
                                "descricao"
                            ]
                        }
                    },
                    "ProjetosOutros": {
                        "type": "array"
                    },
                    "Patentes e registros": {
                        "type": "object"
                    },
                    "Bancas": {
                        "type": "object",
                        "properties": {
                            "Participa\u00e7\u00e3o em bancas de trabalhos de conclus\u00e3o": {
                                "type": "object",
                                "patternProperties": {
                                    "^[0-9]+\\.$": {
                                        "type": "string"
                                    }
                                }
                            },
                            "Participa\u00e7\u00e3o em bancas de comiss\u00f5es julgadoras": {
                                "type": "object",
                                "patternProperties": {
                                    "^[0-9]+\\.$": {
                                        "type": "string"
                                    }
                                }
                            }
                        },
                        "required": []
                    },
                    "Orienta\u00e7\u00f5es": {
                        "type": "array"
                    },
                    "JCR2": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "doi": {
                                    "type": [
                                        "string",
                                        "null"
                                    ]
                                },
                                "impact-factor": {
                                    "type": "string"
                                },
                                "original_title": {
                                    "type": "string"
                                }
                            },
                            "required": []
                        }
                    }
                },
                "required": [
                    "Identifica\u00e7\u00e3o",
                    "Idiomas",
                    "Forma\u00e7\u00e3o",
                    "Atua\u00e7\u00e3o Profissional",
                    "Linhas de Pesquisa",
                    "\u00c1reas",
                    "Produ\u00e7\u00f5es",
                    "ProjetosPesquisa",
                    "ProjetosExtens\u00e3o",
                    "ProjetosDesenvolvimento",
                    "ProjetosOutros",
                    "Patentes e registros",
                    "Bancas",
                    "Orienta\u00e7\u00f5es",
                    "JCR2"
                ]
            }
        }

        try:
            validate(instance=curriculo_dict, schema=schema)
        except Exception as e:
            print(f"Erro de validação: {e}")
            # Tratar o erro de validação (ex: log, interromper a ingestão, etc.)
            raise  # Lança a exceção para interromper a ingestão

    def criar_subgrafo_curriculo(self, curriculo_dict):
        """
        Cria um subgrafo para um currículo, incluindo suas informações e 
        relacionamentos com outras entidades.

        Args:
            curriculo_dict: Um dicionário contendo as informações do currículo.

        Returns:
            Um dicionário contendo as informações do subgrafo, com os nós e as arestas.
        """
            # Validação dos dados antes da criação do subgrafo
        # self._validar_dados(curriculo_dict)

        subgrafo = {"nos": [], "arestas": []}

        # --- Adicionar o nó do currículo ---
        curriculo_id = curriculo_dict['Identificação']['ID Lattes']
        # Remove espaços e caracteres especiais do ID Lattes
        curriculo_id = re.sub(r"[^a-zA-Z0-9_]", "", curriculo_id)        
        subgrafo["nos"].append({"tipo": "Curriculo", "propriedades": curriculo_dict['Identificação']})

        # --- Adicionar nós e arestas para os artigos ---
        artigos = curriculo_dict.get('Produções', {}).get('Artigos completos publicados em periódicos', [])
        for artigo in artigos:
            artigo_id = artigo.get('DOI')  # Usando o DOI como ID do artigo
            if artigo_id:
                artigo = self._corrigir_nome_propriedade(artigo)
                subgrafo["nos"].append({"tipo": "Artigo", "propriedades": artigo})
                subgrafo["arestas"].append({"origem": {"IDLattes": curriculo_id}, "destino": {"DOI": artigo_id}, "tipo": "PUBLICOU_ARTIGO"})

        # --- Adicionar nós e arestas para as áreas de atuação ---
        areas = curriculo_dict.get('Áreas', {})
        for area_id, area_descricao in areas.items():
            area_props = {}  # Define um dicionário vazio como valor padrão
            if area_id and area_descricao:
                area_props = {"id": area_id, "descricao": area_descricao}
                subgrafo["nos"].append({"tipo": "Area", "propriedades": {"id": area_id, "descricao": area_descricao}})
                area_props = self._corrigir_nome_propriedade(area_props)
                subgrafo["arestas"].append({"origem": {"IDLattes": curriculo_id}, "destino": {"id": area_id}, "tipo": "PESQUISA_AREA"})

        # --- Adicionar nós e arestas para a formação acadêmica ---
        formacao = curriculo_dict.get('Formação', {}).get('Acadêmica', [])
        for item in formacao:
            formacao_id = item.get('Descrição')  # Usando a Descrição como ID da formação
            if formacao_id:
                item = self._corrigir_nome_propriedade(item)
                subgrafo["nos"].append({"tipo": "FormacaoAcademica", "propriedades": item})
                subgrafo["arestas"].append({"origem": {"IDLattes": curriculo_id}, "destino": {"Descricao": formacao_id}, "tipo": "POSSUI_FORMACAO"})

        # --- Adicionar nós e arestas para o pós-doutorado ---
        posdoc = curriculo_dict.get('Formação', {}).get('Pos-Doc', [])
        for item in posdoc:
            posdoc_id = item.get('Descrição')  # Usando a Descrição como ID do pós-doutorado
            if posdoc_id:
                item = self._corrigir_nome_propriedade(item)
                subgrafo["nos"].append({"tipo": "PosDoutorado", "propriedades": item})
                subgrafo["arestas"].append({"origem": {"IDLattes": curriculo_id}, "destino": {"Descricao": posdoc_id}, "tipo": "POSSUI_POSDOC"})

        # --- Adicionar nós e arestas para a formação complementar ---
        formacao_complementar = curriculo_dict.get('Formação', {}).get('Complementar', [])
        for item in formacao_complementar:
            complementar_id = item.get('Descrição')  # Usando a Descrição como ID da formação complementar
            if complementar_id:
                item = self._corrigir_nome_propriedade(item)
                subgrafo["nos"].append({"tipo": "FormacaoComplementar", "propriedades": item})
                subgrafo["arestas"].append({"origem": {"IDLattes": curriculo_id}, "destino": {"Descricao": complementar_id}, "tipo": "POSSUI_COMPLEMENTAR"})

        # --- Adicionar nós e arestas para a atuação profissional ---
        atuacao_profissional = curriculo_dict.get('Atuação Profissional', [])
        for item in atuacao_profissional:
            atuacao_id = item.get('Instituição') + " - " + item.get('Ano')  # Usando a Instituição e o Ano como ID da atuação
            if atuacao_id:
                item = self._corrigir_nome_propriedade(item)
                subgrafo["nos"].append({"tipo": "AtuacaoProfissional", "propriedades": item})
                subgrafo["arestas"].append({"origem": {"IDLattes": curriculo_id}, "destino": {"id": atuacao_id}, "tipo": "POSSUI_VINCULO"})

        # --- Adicionar nós e arestas para as linhas de pesquisa ---
        linhas_de_pesquisa = curriculo_dict.get('Linhas de Pesquisa', [])
        for item in linhas_de_pesquisa:
            pesquisa_id = item.get('Descrição')  # Usando a Descrição como ID da linha de pesquisa
            if pesquisa_id:
                item = self._corrigir_nome_propriedade(item)
                subgrafo["nos"].append({"tipo": "LinhaPesquisa", "propriedades": item})
                subgrafo["arestas"].append({"origem": {"IDLattes": curriculo_id}, "destino": {"Descricao": pesquisa_id}, "tipo": "PESQUISA_LINHA"})

        # --- Adicionar nós e arestas para os idiomas ---
        idiomas = curriculo_dict.get('Idiomas', [])
        for item in idiomas:
            idioma_id = item.get('Idioma')  # Usando o Idioma como ID do idioma
            if idioma_id:
                item = self._corrigir_nome_propriedade(item)
                subgrafo["nos"].append({"tipo": "Idioma", "propriedades": item})
                subgrafo["arestas"].append({"origem": {"IDLattes": curriculo_id}, "destino": {"Idioma": idioma_id}, "tipo": "DOMINA_IDIOMA"})

        # Livros publicados/organizados ou edições
        livros = curriculo_dict.get('Produções', {}).get('Livros publicados/organizados ou edições', {})
        for livro_id, livro_descricao in livros.items():
            # Corrige as propriedades antes de adicionar o nó ao subgrafo
            livro_props = {"id": livro_id, "descricao": livro_descricao}
            livro_props = self._corrigir_nome_propriedade(livro_props)            
            subgrafo["nos"].append({"tipo": "Livro", "propriedades": {"id": livro_id, "descricao": livro_descricao}})
            subgrafo["arestas"].append({"origem": {"IDLattes": curriculo_id}, "destino": {"id": livro_id}, "tipo": "PUBLICOU_LIVRO"})

        ## Capítulos de livros publicados
        capitulos = curriculo_dict.get('Produções', {}).get('Capítulos de livros publicados', {})
        for capitulo_id, capitulo_descricao in capitulos.items():
            subgrafo["nos"].append({"tipo": "CapituloLivro", "propriedades": {"id": capitulo_id, "descricao": capitulo_descricao}})
            subgrafo["arestas"].append({"origem": {"IDLattes": curriculo_id}, "destino": {"id": capitulo_id}, "tipo": "PUBLICOU_CAPITULO"})

        ## Resumos publicados em anais de congressos
        resumos_congressos = curriculo_dict.get('Produções', {}).get('Resumos publicados em anais de congressos', {})
        for resumo_id, resumo_descricao in resumos_congressos.items():
            subgrafo["nos"].append({"tipo": "ResumoCongresso", "propriedades": {"id": resumo_id, "descricao": resumo_descricao}})
            subgrafo["arestas"].append({"origem": {"ID Lattes": curriculo_id}, "destino": {"id": resumo_id}, "tipo": "PUBLICOU_RESUMO_CONGRESSO"})

        ## Apresentações de Trabalho
        apresentacoes = curriculo_dict.get('Produções', {}).get('Apresentações de Trabalho', {})
        for apresentacao_id, apresentacao_descricao in apresentacoes.items():
            subgrafo["nos"].append({"tipo": "ApresentacaoTrabalho", "propriedades": {"id": apresentacao_id, "descricao": apresentacao_descricao}})
            subgrafo["arestas"].append({"origem": {"ID Lattes": curriculo_id}, "destino": {"id": apresentacao_id}, "tipo": "APRESENTOU_TRABALHO"})

        ## Outras produções bibliográficas
        outras_producoes = curriculo_dict.get('Produções', {}).get('Outras produções bibliográficas', {})
        for producao_id, producao_descricao in outras_producoes.items():
            subgrafo["nos"].append({"tipo": "ProducaoBibliografica", "propriedades": {"id": producao_id, "descricao": producao_descricao}})
            subgrafo["arestas"].append({"origem": {"ID Lattes": curriculo_id}, "destino": {"id": producao_id}, "tipo": "PUBLICOU_PRODUCAO"})

        ## Entrevistas, mesas redondas, programas e comentários na mídia
        entrevistas = curriculo_dict.get('Produções', {}).get('Entrevistas, mesas redondas, programas e comentários na mídia', {})
        for entrevista_id, entrevista_descricao in entrevistas.items():
            subgrafo["nos"].append({"tipo": "Entrevista", "propriedades": {"id": entrevista_id, "descricao": entrevista_descricao}})
            subgrafo["arestas"].append({"origem": {"ID Lattes": curriculo_id}, "destino": {"id": entrevista_id}, "tipo": "PARTICIPOU_ENTREVISTA"})

        ## Demais tipos de produção técnica
        demais_producoes_tecnicas = curriculo_dict.get('Produções', {}).get('Demais tipos de produção técnica', {})
        for producao_tecnica_id, producao_tecnica_descricao in demais_producoes_tecnicas.items():
            subgrafo["nos"].append({"tipo": "ProducaoTecnica", "propriedades": {"id": producao_tecnica_id, "descricao": producao_tecnica_descricao}})
            subgrafo["arestas"].append({"origem": {"ID Lattes": curriculo_id}, "destino": {"id": producao_tecnica_id}, "tipo": "PRODUZIU_TECNICA"})

        # --- Nós e Arestas para os projetos (pesquisa, extensão, etc.) ---
        # Projetos de Pesquisa
        projetos_pesquisa = curriculo_dict.get('ProjetosPesquisa', [])
        for projeto in projetos_pesquisa:
            projeto_id = projeto.get('chave')  # Usando a chave como ID do projeto
            if projeto_id:
                subgrafo["nos"].append({"tipo": "ProjetoPesquisa", "propriedades": projeto})
                subgrafo["arestas"].append({"origem": {"ID Lattes": curriculo_id}, "destino": {"chave": projeto_id}, "tipo": "PARTICIPOU_PROJETO_PESQUISA"})

        # Projetos de Extensão
        projetos_extensao = curriculo_dict.get('ProjetosExtensão', [])
        for projeto in projetos_extensao:
            projeto_id = projeto.get('chave')  # Usando a chave como ID do projeto
            if projeto_id:
                subgrafo["nos"].append({"tipo": "ProjetoExtensao", "propriedades": projeto})
                subgrafo["arestas"].append({"origem": {"ID Lattes": curriculo_id}, "destino": {"chave": projeto_id}, "tipo": "PARTICIPOU_PROJETO_EXTENSAO"})

        # Projetos de Desenvolvimento
        projetos_desenvolvimento = curriculo_dict.get('ProjetosDesenvolvimento', [])
        for projeto in projetos_desenvolvimento:
            projeto_id = projeto.get('chave')  # Usando a chave como ID do projeto
            if projeto_id:
                subgrafo["nos"].append({"tipo": "ProjetoDesenvolvimento", "propriedades": projeto})
                subgrafo["arestas"].append({"origem": {"ID Lattes": curriculo_id}, "destino": {"chave": projeto_id}, "tipo": "PARTICIPOU_PROJETO_DESENVOLVIMENTO"})

        # Projetos Outros
        projetos_outros = curriculo_dict.get('ProjetosOutros', [])
        for projeto in projetos_outros:
            projeto_id = projeto.get('chave')  # Usando a chave como ID do projeto
            if projeto_id:
                subgrafo["nos"].append({"tipo": "ProjetoOutro", "propriedades": projeto})
                subgrafo["arestas"].append({"origem": {"ID Lattes": curriculo_id}, "destino": {"chave": projeto_id}, "tipo": "PARTICIPOU_PROJETO_OUTRO"})

        # --- Nós e arestas para patentes e registros ---
        patentes = curriculo_dict.get('Patentes e registros', {})
        for patente_id, patente_info in patentes.items():
            patente_info = self._corrigir_nome_propriedade(patente_info)
            # Considerando que cada patente_info é um dicionário com informações da patente
            subgrafo["nos"].append({"tipo": "Patente", "propriedades": patente_info})
            subgrafo["arestas"].append({"origem": {"ID Lattes": curriculo_id}, "destino": {"id": patente_id}, "tipo": "POSSUI_PATENTE"})

        # --- Nós e arestas para bancas e Orientações---
        bancas = curriculo_dict.get('Bancas', {})
        
        # Participação em bancas de trabalhos de conclusão
        bancas_trabalhos = bancas.get('Participação em bancas de trabalhos de conclusão', {})
        for banca_id, banca_info in bancas_trabalhos.items():
            banca_props = {"id": banca_id, "descricao": banca_info}
            banca_props = self._corrigir_nome_propriedade(banca_props)            
            subgrafo["nos"].append({"tipo": "BancaTrabalho", "propriedades": {"id": banca_id, "descricao": banca_info}})
            subgrafo["arestas"].append({"origem": {"ID Lattes": curriculo_id}, "destino": {"id": banca_id}, "tipo": "PARTICIPOU_BANCA_TRABALHO"})

        # Participação em bancas de comissões julgadoras
        bancas_comissoes = bancas.get('Participação em bancas de comissões julgadoras', {})
        for banca_id, banca_info in bancas_comissoes.items():
            banca_props = {"id": banca_id, "descricao": banca_info}
            banca_props = self._corrigir_nome_propriedade(banca_props)            
            subgrafo["nos"].append({"tipo": "BancaComissao", "propriedades": {"id": banca_id, "descricao": banca_info}})
            subgrafo["arestas"].append({"origem": {"ID Lattes": curriculo_id}, "destino": {"id": banca_id}, "tipo": "PARTICIPOU_BANCA_COMISSAO"})

        # Orientações
        orientacoes = curriculo_dict.get('Orientações', [])
        for orientacao in orientacoes:
            orientacao_id = orientacao.get('nome')  # Usando o nome como ID da orientação
            if orientacao_id:
                orientacao = self._corrigir_nome_propriedade(orientacao)
                subgrafo["nos"].append({"tipo": "Orientacao", "propriedades": orientacao})
                subgrafo["arestas"].append({"origem": {"ID Lattes": curriculo_id}, "destino": {"nome": orientacao_id}, "tipo": "ORIENTADOR"})

        # --- Nós e arestas para Fator de Impacto JCR---
        # JCR2
        jcr2 = curriculo_dict.get('JCR2', [])
        for item in jcr2:
            jcr2_id = item.get('doi')  # Usando o DOI como ID do JCR2
            if jcr2_id:
                item = self._corrigir_nome_propriedade(item)
                subgrafo["nos"].append({"tipo": "JCR2", "propriedades": item})
                subgrafo["arestas"].append({"origem": {"ID Lattes": curriculo_id}, "destino": {"doi": jcr2_id}, "tipo": "POSSUI_JCI"})

        return subgrafo


    def ingerir_subgrafo(self, subgrafo_dict):
        """
        Ingere um subgrafo no grafo de conhecimento do Neo4j.

        Args:
            subgrafo_dict: Um dicionário contendo as informações do subgrafo,
                           com os nós e as arestas.
        """
        # Extrair nós e arestas do subgrafo
        nos = subgrafo_dict.get("nos", [])
        arestas = subgrafo_dict.get("arestas", [])

        # Adicionar nós ao Neo4j
        for no in nos:
            propriedades = no.get("propriedades", {})
            tipo_no = no.get("tipo")
            query = f"""
                MERGE (n:{tipo_no} {{ {self._formatar_propriedades(propriedades)} }})
                RETURN n
            """
            self.graph.run(query)

        # Adicionar arestas ao Neo4j
        for aresta in arestas:
            no_origem = aresta.get("origem")
            no_destino = aresta.get("destino")
            tipo_aresta = aresta.get("tipo")
            propriedades = aresta.get("propriedades", {})
            query = f"""
                MATCH (n1 {{ {self._formatar_propriedades(no_origem)} }})
                MATCH (n2 {{ {self._formatar_propriedades(no_destino)} }})
                MERGE (n1)-[r:{tipo_aresta} {{ {self._formatar_propriedades(propriedades)} }}]->(n2)
                RETURN r
            """
            self.graph.run(query)


    def gerar_embeddings(self, texto):
        """
        Gera embeddings para um texto usando o modelo SentenceTransformer.

        Args:
            texto: O texto a ser usado para gerar o embedding.

        Returns:
            Um tensor PyTorch com o embedding.
        """
        return self.embedding_model.encode(texto, convert_to_tensor=True)


    def forward(self, x):
        """
        Define o forward pass do KGNN.

        Args:
            x: Os dados de entrada, contendo a lista de nós e seus tipos.

        Returns:
            O resultado do forward pass.
        """

        # 1. Obter embeddings dos nós
        embeddings = self.obter_embeddings_nos(x)

        # 2. Agregar informações dos vizinhos
        # Passar a lista de tipos de nós para a função agregar_informacoes_vizinhos
        embeddings_agregados = self.agregar_informacoes_vizinhos(embeddings, x['tipos_nos'])  

        # 3. Combinar embeddings dos nós com embeddings agregados
        embeddings_combinados = self.combinar_embeddings(embeddings, embeddings_agregados)

        # 4. Aplicar camadas adicionais (opcional)
        embeddings_combinados = self.camadas_adicionais(embeddings_combinados)

        # 5. Retornar os embeddings finais
        return embeddings_combinados


    def obter_embeddings_nos(self, x):
        """
        Obtém os embeddings dos nós de entrada.
        Considera o schema dos dados dos currículos fornecido e permite selecionar quais campos serão usados.

        Args:
            x: Dados de entrada em lista de dicionários conforme schema.

        Returns:
            Um tensor com os embeddings dos nós.
        """

        embeddings = []
        for no in x['nos']:
            propriedades = no['propriedades']
            texto_no = ''

            # --- Flags para selecionar os campos a serem usados ---
            usar_identificacao = True
            usar_idiomas = False
            usar_formacao = True
            usar_atuacao_profissional = True
            usar_linhas_de_pesquisa = True
            usar_areas = True
            usar_producoes = True
            usar_projetos_pesquisa = True
            usar_projetos_extensao = True
            usar_projetos_desenvolvimento = True
            usar_projetos_outros = True
            usar_patentes_e_registros = True
            usar_bancas = True
            usar_orientacoes = True
            usar_jcr2 = False

            # Extrair texto das propriedades do nó, de acordo com o schema e as flags
            if usar_identificacao:
                for chave, valor in propriedades['Identificação'].items():
                    texto_no += valor + ' '

            if usar_idiomas:
                for idioma in propriedades['Idiomas']:
                    for chave, valor in idioma.items():
                        texto_no += valor + ' '

            if usar_formacao:
                for tipo_formacao, formacoes in propriedades['Formação'].items():
                    for formacao in formacoes:
                        if isinstance(formacao, dict):  # Verifica formação (Acadêmica/Complementar)
                            for chave, valor in formacao.items():
                                texto_no += valor + ' '
                        elif isinstance(formacao, str):  # Verifica se é uma string (Pos-Doc)
                            texto_no += formacao + ' '

            if usar_atuacao_profissional:
                for atuacao in propriedades['Atuação Profissional']:
                    for chave, valor in atuacao.items():
                        texto_no += valor + ' '

            if usar_linhas_de_pesquisa:
                for linha_de_pesquisa in propriedades['Linhas de Pesquisa']:
                    texto_no += linha_de_pesquisa + ' '

            if usar_areas:
                for area_id, area_descricao in propriedades['Áreas'].items():
                    texto_no += area_descricao + ' '

            if usar_producoes:
                for tipo_producao, producoes in propriedades['Produções'].items():
                    if isinstance(producoes, list):
                        for producao in producoes:
                            for chave, valor in producao.items():
                                texto_no += valor + ' '
                    elif isinstance(producoes, dict):
                        for k, v in producoes.items():
                            texto_no += v + ' '

            if usar_projetos_pesquisa:
                for projeto in propriedades['ProjetosPesquisa']:
                    for chave, valor in projeto.items():
                        texto_no += valor + ' '

            if usar_projetos_extensao:
                for projeto in propriedades['ProjetosExtensão']:
                    for chave, valor in projeto.items():
                        texto_no += valor + ' '

            if usar_projetos_desenvolvimento:
                for projeto in propriedades['ProjetosDesenvolvimento']:
                    for chave, valor in projeto.items():
                        texto_no += valor + ' '

            if usar_projetos_outros:
                for projeto in propriedades['ProjetosOutros']:
                    for chave, valor in projeto.items():
                        texto_no += valor + ' '

            if usar_patentes_e_registros:
                for patente_id, patente_info in propriedades['Patentes e registros'].items():
                    if isinstance(patente_info, str):
                        texto_no += patente_info + ' '
                    elif isinstance(patente_info, dict):
                        for chave, valor in patente_info.items():
                            texto_no += valor + ' '

            if usar_bancas:
                for tipo_banca, bancas in propriedades['Bancas'].items():
                    for banca_id, banca_info in bancas.items():
                        texto_no += banca_info + ' '

            if usar_orientacoes:
                for orientacao in propriedades['Orientações']:
                    if isinstance(orientacao, dict):
                        for chave, valor in orientacao.items():
                            texto_no += valor + ' '
                    elif isinstance(orientacao, str):
                        texto_no += orientacao + ' '

            if usar_jcr2:
                for jcr2_item in propriedades['JCR2']:
                    for chave, valor in jcr2_item.items():
                        if isinstance(valor, str):
                            texto_no += valor + ' '

            # Gerar embedding do nó
            embedding_no = self.embedding_model.encode(texto_no, convert_to_tensor=True)
            embeddings.append(embedding_no)

        return torch.stack(embeddings)


    def obter_embeddings_vizinhos(self, no_embedding, tipo_no, tipo_relacionamento):
        """
        Obtém os embeddings dos vizinhos de um nó através de um tipo de 
        relacionamento, usando a propriedade do nó como identificador único.

        Args:
            no_embedding: O embedding do nó.
            tipo_no: O tipo do nó.
            tipo_relacionamento: O tipo de relacionamento.

        Returns:
            Uma lista com os embeddings dos vizinhos.
        """

        # Determinar a propriedade do nó a ser usada como identificador
        if tipo_no == 'Curriculo':
            propriedade_id = 'IDLattes'
        elif tipo_no == 'Artigo':
            propriedade_id = 'DOI'
        # ... adicione elif para outros tipos de nós com suas respectivas propriedades
        else:
            raise ValueError(f"Tipo de nó inválido: {tipo_no}")

        # Construir a consulta Cypher dinamicamente
        query = f"""
            MATCH (n:{tipo_no})-[r:{tipo_relacionamento}]-(m)
            WHERE n.{propriedade_id} = $id
            RETURN m
        """

        # Executar a consulta e obter os nós vizinhos
        resultados = self.graph.run(query, id=no_embedding).data()
        vizinhos = []
        for resultado in resultados:
            no_vizinho = resultado['m']

            # Extrair texto das propriedades do nó vizinho
            texto_vizinho = ''
            for chave, valor in no_vizinho.items(): # type: ignore
                if isinstance(valor, str):
                    texto_vizinho += valor + ' '
                elif isinstance(valor, list):
                    for item in valor:
                        if isinstance(item, str):
                            texto_vizinho += item + ' '
                        elif isinstance(item, dict):
                            for k, v in item.items():
                                if isinstance(v, str):
                                    texto_vizinho += v + ' '

            # Gerar embedding do nó vizinho
            embedding_vizinho = self.embedding_model.encode(texto_vizinho, convert_to_tensor=True)
            vizinhos.append(embedding_vizinho)

        return vizinhos
    

    def agregar_informacoes_vizinhos(self, embeddings, tipos_nos):
        """
        Agrega informações dos vizinhos de cada nó.
        Permite selecionar quais tipos de vizinhos serão usados na agregação.

        Args:
            embeddings: Os embeddings dos nós.
            tipos_nos: Uma lista com os tipos dos nós.

        Returns:
            Um tensor com os embeddings agregados dos vizinhos.
        """

        # --- Flags para selecionar os tipos de vizinhos a serem usados ---
        usar_artigos = True
        usar_areas = True
        usar_formacao_academica = True
        usar_pos_doutorado = True
        usar_formacao_complementar = True
        usar_atuacao_profissional = True
        usar_linhas_de_pesquisa = True
        usar_idiomas = True
        usar_livros = True
        usar_capitulos_livros = True
        usar_resumos_congressos = True
        usar_apresentacoes_trabalho = True
        usar_outras_producoes = True
        usar_entrevistas = True
        usar_producoes_tecnicas = True
        usar_projetos_pesquisa = True
        usar_projetos_extensao = True
        usar_projetos_desenvolvimento = True
        usar_projetos_outros = True
        usar_patentes_e_registros = True
        usar_bancas_trabalho = True
        usar_bancas_comissao = True
        usar_orientacoes = True
        usar_jcr2 = True

        embeddings_agregados = []
        for i, no in enumerate(embeddings):
            vizinhos = []
            tipo_no = tipos_nos[i]  # Obter o tipo do nó atual

            # Obter os embeddings dos vizinhos de acordo com as flags
            if usar_artigos:
                vizinhos += self.obter_embeddings_vizinhos(no, tipo_no, 'PUBLICOU_ARTIGO')
            if usar_areas:
                vizinhos += self.obter_embeddings_vizinhos(no, tipo_no, 'PESQUISA_AREA')
            if usar_formacao_academica:
                vizinhos += self.obter_embeddings_vizinhos(no, tipo_no, 'POSSUI_FORMACAO')
            if usar_pos_doutorado:
                vizinhos += self.obter_embeddings_vizinhos(no, tipo_no, 'POSSUI_POSDOC')
            if usar_formacao_complementar:
                vizinhos += self.obter_embeddings_vizinhos(no, tipo_no, 'POSSUI_COMPLEMENTAR')
            if usar_atuacao_profissional:
                vizinhos += self.obter_embeddings_vizinhos(no, tipo_no, 'POSSUI_VINCULO')
            if usar_linhas_de_pesquisa:
                vizinhos += self.obter_embeddings_vizinhos(no, tipo_no, 'PESQUISA_LINHA')
            if usar_idiomas:
                vizinhos += self.obter_embeddings_vizinhos(no, tipo_no, 'DOMINA_IDIOMA')
            if usar_livros:
                vizinhos += self.obter_embeddings_vizinhos(no, tipo_no, 'PUBLICOU_LIVRO')
            if usar_capitulos_livros:
                vizinhos += self.obter_embeddings_vizinhos(no, tipo_no, 'PUBLICOU_CAPITULO')
            if usar_resumos_congressos:
                vizinhos += self.obter_embeddings_vizinhos(no, tipo_no, 'PUBLICOU_RESUMO_CONGRESSO')
            if usar_apresentacoes_trabalho:
                vizinhos += self.obter_embeddings_vizinhos(no, tipo_no, 'APRESENTOU_TRABALHO')
            if usar_outras_producoes:
                vizinhos += self.obter_embeddings_vizinhos(no, tipo_no, 'PUBLICOU_PRODUCAO')
            if usar_entrevistas:
                vizinhos += self.obter_embeddings_vizinhos(no, tipo_no, 'PARTICIPOU_ENTREVISTA')
            if usar_producoes_tecnicas:
                vizinhos += self.obter_embeddings_vizinhos(no, tipo_no, 'PRODUZIU_TECNICA')
            if usar_projetos_pesquisa:
                vizinhos += self.obter_embeddings_vizinhos(no, tipo_no, 'PARTICIPOU_PROJETO_PESQUISA')
            if usar_projetos_extensao:
                vizinhos += self.obter_embeddings_vizinhos(no, tipo_no, 'PARTICIPOU_PROJETO_EXTENSAO')
            if usar_projetos_desenvolvimento:
                vizinhos += self.obter_embeddings_vizinhos(no, tipo_no, 'PARTICIPOU_PROJETO_DESENVOLVIMENTO')
            if usar_projetos_outros:
                vizinhos += self.obter_embeddings_vizinhos(no, tipo_no, 'PARTICIPOU_PROJETO_OUTRO')
            if usar_patentes_e_registros:
                vizinhos += self.obter_embeddings_vizinhos(no, tipo_no, 'POSSUI_PATENTE')
            if usar_bancas_trabalho:
                vizinhos += self.obter_embeddings_vizinhos(no, tipo_no, 'PARTICIPOU_BANCA_TRABALHO')
            if usar_bancas_comissao:
                vizinhos += self.obter_embeddings_vizinhos(no, tipo_no, 'PARTICIPOU_BANCA_COMISSAO')
            if usar_orientacoes:
                vizinhos += self.obter_embeddings_vizinhos(no, tipo_no, 'ORIENTADOR')
            if usar_jcr2:
                vizinhos += self.obter_embeddings_vizinhos(no, tipo_no, 'POSSUI_JCI')

            # Agregar embeddings dos vizinhos (usando a média)
            if vizinhos:
                embeddings_agregados.append(torch.mean(torch.stack(vizinhos), dim=0))
            else:
                # Se nó não tiver vizinhos, usar próprio embedding
                embeddings_agregados.append(no)

        return torch.stack(embeddings_agregados)


    def combinar_embeddings_concat(self, embeddings, embeddings_agregados):
        """
        Combina os embeddings dos nós com os embeddings agregados dos vizinhos, por Concatenação dos embeddings.

        Args:
            embeddings: Os embeddings dos nós.
            embeddings_agregados: Os embeddings agregados dos vizinhos.

        Returns:
            Um tensor com os embeddings combinados.
        """

        embeddings_combinados = torch.cat([embeddings, embeddings_agregados], dim=1)
        return embeddings_combinados


    def combinar_embeddings_soma(self, embeddings, embeddings_agregados):
        """
        Combina os embeddings dos nós com os embeddings agregados dos vizinhos, por Soma dos embeddings.

        Args:
            embeddings: Os embeddings dos nós.
            embeddings_agregados: Os embeddings agregados dos vizinhos.

        Returns:
            Um tensor com os embeddings combinados.
        """

        embeddings_combinados = embeddings + embeddings_agregados
        return embeddings_combinados


    def combinar_embeddings_ponderada(self, embeddings, embeddings_agregados):
        """
        Combina os embeddings dos nós com os embeddings agregados dos vizinhos, por Soma ponderada.

        Args:
            embeddings: Os embeddings dos nós.
            embeddings_agregados: Os embeddings agregados dos vizinhos.

        Returns:
            Um tensor com os embeddings combinados.
        """

        peso_no = torch.nn.Parameter(torch.tensor(0.6))  # Peso inicial o embedding do nó
        peso_vizinhos = torch.nn.Parameter(torch.tensor(0.4))  # Peso inicial embedding dos vizinhos
        embeddings_combinados = peso_no * embeddings + peso_vizinhos * embeddings_agregados
        return embeddings_combinados


    def camadas_adicionais(self, embeddings_combinados):
        """
        Aplica camadas adicionais aos embeddings combinados (opcional).

        Args:
            embeddings_combinados: Os embeddings combinados.

        Returns:
            Um tensor com os embeddings após a aplicação das camadas adicionais.
        """

        # Definição das camadas adicionais
        self.linear1 = nn.Linear(embeddings_combinados.size(1), 128)  # Camada linear 1
        self.relu = nn.ReLU()  # Função de ativação ReLU
        self.linear2 = nn.Linear(128, 64)  # Camada linear 2

        # Aplicação das camadas
        embeddings_combinados = self.linear1(embeddings_combinados)
        embeddings_combinados = self.relu(embeddings_combinados)
        embeddings_combinados = self.linear2(embeddings_combinados)

        return embeddings_combinados