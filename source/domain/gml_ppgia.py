class AnalisadorProducaoPPGIA:
    def __init__(self, dict_list):
        """
        Inicializa a classe com a lista de dicionários de currículos Lattes como entrada.

        Args:
            dict_list (lista): Lista de dicinários aninhados com dados de currículos Lattes.
        """
        self.dict_list = dict_list

    ## PADRONIZAÇÃO DE NOMES DE AUTOR E ANÁLISE DE SIMILARIDADES
    def padronizar_nome(self, linha_texto):
        '''Procura sobrenomes e abreviaturas e monta nome completo
        Recebe: String com todos os sobrenomes e nomes, abreviados ou não
        Retorna: Nome completo no formato padronizado em SOBRENOME AGNOME, Prenomes
        Autor: Marcos Aires (Mar.2022)
        '''
        import unicodedata
        import re
        # print('               Analisando:',linha_texto)
        string = ''.join(ch for ch in unicodedata.normalize('NFKD', linha_texto) if not unicodedata.combining(ch))
        string = string.replace('(Org)','').replace('(Org.)','').replace('(Org).','').replace('.','').replace('\'','')
        string = string.replace(',,,',',').replace(',,',',')
        string = re.sub(r'[0-9]+', '', string)
            
        # Expressões regulares para encontrar padrões de divisão de nomes de autores
        sobrenome_inicio   = re.compile(r'^[A-ZÀ-ú-a-z]+,')                  # Sequência de letras maiúsculas no início da string
        sobrenome_composto = re.compile(r'^[A-ZÀ-ú-a-z]+[ ][A-ZÀ-ú-a-z]+,')  # Duas sequências de letras no início da string, separadas por espaço, seguidas por vírgula
        letra_abrevponto   = re.compile(r'^[A-Z][.]')                        # Uma letra maiúscula no início da string, seguida por ponto
        letra_abrevespaco  = re.compile(r'^[A-Z][ ]')                        # Uma letra maiúscula no início da string, seguida por espaço
        letras_dobradas    = re.compile(r'[A-Z]{2}')                         # Duas letras maiúsculas juntas no início da string, seguida por espaço
        letras_dobradasini = re.compile(r'[A-Z]{2}[ ]')                      # Duas letras maiúsculas juntas no início da string, seguida por espaço
        letras_dobradasfim = re.compile(r'[ ][A-Z]{2}')                      # Duas letras maiúsculas juntas no final da string, precedida por espaço
        letras_duasconsnts = re.compile(r'[B-DF-HJ-NP-TV-XZ]{2}')            # Duas Letras maiúsculas e consoantes juntas
        letras_tresconsnts = re.compile(r'[B-DF-HJ-NP-TV-XZ]{3}')            # Três Letras maiúsculas e consoantes juntas
        
        # Agnomes e preprosições a tratar, agnomes vão maiúsculas para sobrenome e preposições vão para minúsculas nos nomes
        nomes=[]
        agnomes       = ['NETO','JUNIOR','FILHO','SEGUNDO','TERCEIRO']
        preposicoes   = ['da','de','do','das','dos']
        nome_completo = ''
        
        # Ajustar lista de termos, identificar sobrenomes compostos e ajustar sobrenome com ou sem presença de vírgula
        div_sobrenome      = sobrenome_inicio.findall(string)
        div_sbrcomposto    = sobrenome_composto.findall(string)
        
        # print('-'*100)
        # print('                 Recebido:',string)
        
        # Caso haja vírgulas na string, tratar sobrenomes e sobrenomes compostos
        if div_sobrenome != [] or div_sbrcomposto != []:
            # print('CASO_01: Há víruglas na string')
            div = string.split(', ')
            sobrenome     = div[0].strip().upper()
            try:
                div_espaco    = div[1].split(' ')
            except:
                div_espaco    = ['']
            primeiro      = div_espaco[0].strip('.')
            
            # print('     Dividir por vírgulas:',div)
            # print('      Primeira DivVirgula:',sobrenome)
            # print('Segunda DivVrg/DivEspaços:',div_espaco)
            # print('      Primeira DivEspaços:',primeiro)
                
            # Caso primeiro nome sejam somente duas letras maiúsculas juntas, trata-se de duas iniciais
            if len(primeiro)==2 or letras_tresconsnts.findall(primeiro):
                # print('CASO_01.a: Há duas letras ou três letras consoantes juntas, são iniciais')
                primeiro_nome=primeiro[0].strip()
                # print('          C01.a1_PrimNome:',primeiro_nome)
                nomes.append(primeiro[1].strip().upper())
                try:
                    nomes.append(primeiro[2].strip().upper())
                except:
                    pass
            else:
                # print('CASO_01.b: Primeiro nome maior que 2 caracteres')
                primeiro_nome = div_espaco[0].strip().title()
                # print('          C01.a2_PrimNome:',primeiro_nome)
            
            # Montagem da lista de nomes do meio
            for nome in div_espaco:
                # print('CASO_01.c: Para cada nome da divisão por espaços após divisão por vírgula')
                if nome not in nomes and nome.lower()!=primeiro_nome.lower() and nome.lower() not in primeiro_nome.lower() and nome!=sobrenome:   
                    # print('CASO_01.c1: Se o nome não está nem como primeiro nome, nem sobrenomes')
                    # print(nome, len(nome))
                    
                    # Avaliar se é abreviatura seguida de ponto e remover o ponto
                    if len(nome)<=2 and nome.lower() not in preposicoes:
                        # print('    C01.c1.1_Nome<=02:',nome)
                        for inicial in nome:
                            # print(inicial)
                            if inicial not in nomes and inicial not in primeiro_nome:
                                nomes.append(inicial.replace('.','').strip().title())
                    elif len(nome)==3 and nome.lower() not in preposicoes:
                            # print('    C01.c1.2_Nome==03:',nome)
                            for inicial in nome:
                                if inicial not in nomes and inicial not in primeiro_nome:
                                    nomes.append(inicial.replace('.','').strip().title())
                    else:
                        if nome not in nomes and nome!=primeiro_nome and nome!=sobrenome and nome!='':
                            if nome.lower() in preposicoes:
                                nomes.append(nome.replace('.','').strip().lower())
                            else:
                                nomes.append(nome.replace('.','').strip().title())
                            # print(nome,'|',primeiro_nome)
                            
            #caso haja sobrenome composto que não esteja nos agnomes considerar somente primeiro como sobrenome
            if div_sbrcomposto !=[] and sobrenome.split(' ')[1] not in agnomes and sobrenome.split(' ')[0].lower() not in preposicoes:
                # print('CASO_01.d: Sobrenome composto sem agnomes')
                # print(div_sbrcomposto)
                # print('Sobrenome composto:',sobrenome)
                
                nomes.append(sobrenome.split(' ')[1].title())
                sobrenome = sobrenome.split(' ')[0].upper()
                # print('Sobrenome:',sobrenome)
                
                for i in nomes:
                    if i.lower() in sobrenome.lower():
                        nomes.remove(i)
                # print('    Nomes:',nomes)
            
            #caso haja preposição como agnome desconsiderar e passar para final dos nomes
            if div_sbrcomposto !=[] and sobrenome.split(' ')[0].lower() in preposicoes:
                # print('CASO_01.e: Preposição no Sobrenome passar para o final dos nomes')
                # print('   div_sbrcomposto:', div_sbrcomposto)
                # print('Sobrenome composto:',div_sbrcomposto)
                
                nomes.append(div_sbrcomposto[0].split(' ')[0].lower())
                # print('    Nomes:',nomes)
                sobrenome = div_sbrcomposto[0].split(' ')[1].upper().strip(',')
                # print('Sobrenome:',sobrenome)
                
                for i in nomes:
                    # print('CASO_01.e1: Para cada nome avaliar se o sobrenome está na lista')
                    if i.lower() in sobrenome.lower():
                        nomes.remove(i)
                # print('  Nomes:',nomes)
            
            # print('Ao final do Caso 01')
            # print('    Sobrenome com vírgula:',sobrenome, len(sobrenome),'letras')
            # print('Primeiro nome com vírgula:',primeiro_nome, len(primeiro_nome),'letras')
            # print('           Lista de nomes:',nomes, len(nomes),'nomes')
            
        # Caso não haja vírgulas na string considera sobrenome o último nome da string dividida com espaço vazio
        else:
            # print('CASO_02: Não há víruglas na string')
            try:
                div = string.split(' ')
                # print('      Divisões por espaço:',div)
                
                if div[-1] in agnomes: # nome final é um agnome
                    sobrenome     = div[-2].upper().strip()+' '+div[-1].upper().strip()
                    for i in div[1:-2]:
                        if i not in sobrenome and i not in preposicoes:
                            nomes.append(i.title().strip())
                        if i in preposicoes:
                            nomes.append(i.lower().strip())
                else:
                    if len(div[-1]) > 2:
                        sobrenome     = div[-1].upper().strip()
                        primeiro_nome = div[1].title().strip()
                        for i in div[1:-1]:
                            if i != sobrenome and i not in preposicoes:
                                nomes.append(i.title().strip())
                            if i in preposicoes:
                                nomes.append(i.lower().strip())
                    else:
                        sobrenome     = div[-2].upper().strip()
                        for i in div[-1]:
                            nomes.append(i.title())
                        primeiro_nome = nomes[0].title().strip()
                        for i in div[1:-1]:
                            if i != sobrenome and i not in preposicoes:
                                nomes.append(i.title().strip())
                            if i in preposicoes:
                                nomes.append(i.lower().strip())
            except:
                sobrenome = div[-1].upper().strip()
                for i in div[1:-1]:
                        if i != sobrenome and i not in preposicoes:
                            nomes.append(i.title().strip())
                        if i in preposicoes:
                            nomes.append(i.lower().strip())
                
            if sobrenome.lower() != div[0].lower().strip():
                primeiro_nome=div[0].title().strip()
            else:
                primeiro_nome=''
            
            # print('Ao final do Caso 02')
            # print('    Sobrenome sem vírgula:',sobrenome, len(sobrenome),'letras')
            # print('Primeiro nome sem vírgula:',primeiro_nome, len(primeiro_nome),'letras')
            # print('Nomes do meio sem vírgula:',nomes, len(nomes),'nomes')
        
        # Encontrar e tratar como abreviaturas termos com apenas uma ou duas letras iniciais juntas, com ou sem ponto
        for j in nomes:
            # print('CASO_03: Avaliar cada nome armazenado na variável nomes')
            # Procura padrões com expressões regulares na string
            div_sobrenome      = sobrenome_inicio.findall(j)
            div_sbrcomposto    = sobrenome_composto.findall(j)
            div_abrevponto     = letra_abrevponto.findall(j)
            div_abrevespaco    = letra_abrevespaco.findall(j)
            div_ltrdobradasini = letras_dobradasini.findall(j)
            div_ltrdobradasfim = letras_dobradasfim.findall(j)
            div_ltrdobradas    = letras_dobradas.findall(j)
            tamanho=len(j)
            # print('\n', div_ltrdobradasini, div_ltrdobradasfim, tamanho, 'em:',j,len(j))
            
            #caso houver abreviatura com uma letra em maiúscula nos nomes
            if div_abrevponto !=[] or tamanho==1:
                # print('CASO_03.1: Há abreviaturas uma letra maiúscula nos nomes')
                nome = j.replace('.','').strip()
                if nome not in nomes and nome != sobrenome and nome != primeiro_nome:
                    # print('CASO_03.1a: Há abreviaturas uma letra maiúscula nos nomes')
                    nomes.append(nome.upper())
            
            #caso houver duas inicias juntas em maiúsculas
            elif div_ltrdobradasini !=[] or div_ltrdobradasfim !=[] or div_ltrdobradas !=[] :
                # print('CASO_03.2: Há abreviaturas uma letra maiúscula nos nomes')
                for letra in j:
                    # print('CASO_03.2a: Avaliar cada inicial do nome')
                    if letra not in nomes and letra != sobrenome and letra != primeiro_nome:
                        # print('CASO_03.2a.1: Se não estiver adicionar inicial aos nomes')
                        nomes.append(letra.upper())
            
            #caso haja agnomes ao sobrenome
            elif sobrenome in agnomes:
                # print('CASO_03.3: Há agnomes nos sobrenomes')
                sobrenome = nomes[-1].upper()+' '+sobrenome
                # print(sobrenome.split(' '))
                # print('Sobrenome composto:',sobrenome)
                for i in nomes:
                    if i.lower() in sobrenome.lower():
                        nomes.remove(i)
                # print('Nomes do meio:',nomes)
                
            else:
                # print('CASO_03.4: Não há agnomes nos sobrenomes')
                if j not in nomes and j not in sobrenome and j != primeiro_nome:
                    if len(nomes) == 1:
                        nomes.append(j.upper())
                    elif 1 < len(nomes) <= 3:
                        nomes.append(j.lower())
                    else:
                        nomes.append(j.title())
            
            # print('Ao final do Caso 03')
            # print('    Sobrenome com vírgula:',sobrenome, len(sobrenome),'letras')
            # print('Primeiro nome com vírgula:',primeiro_nome, len(primeiro_nome),'letras')
            # print('Nomes do meio com vírgula:',nomes, len(nomes),'nomes')
            
        nomes_meio=' '.join([str for str in nomes]).strip()
        # print('        Qte nomes do meio:',nomes,len(nomes))
        
        if primeiro_nome.lower() == sobrenome.lower():
            # print('CASO_04: Primeiro nome é igual ao sobrenome')
            try:
                primeiro_nome=nomes_meio.split(' ')[0]
            except:
                pass
            try:
                nomes_meio.replace(sobrenome,'')
            except:
                pass
        
            # print('Ao final do caso 04')
            # print('    Sobrenome com vírgula:',sobrenome, len(sobrenome),'letras')
            # print('Primeiro nome com vírgula:',primeiro_nome, len(primeiro_nome),'letras')
            # print('Nomes do meio com vírgula:',nomes, len(nomes),'nomes')
        
        # Caso sobrenome seja só de 1 letra passá-lo para nomes e considerar o próximo nome como sobrenome
        for i in range(len(div)):
            if len(sobrenome)==1 or sobrenome.lower() in preposicoes:
                # print('CASO_05: Mudar sobrenomes até o adequado')
                div    = string.split(', ')
                # print('Divisão por vírgulas:',div)
                avaliar0       = div[0].split(' ')[0].strip()
                if 1< len(avaliar0) < 3:
                    # print('CASO_05.1: 1 < Sobrenome < 3 fica em minúsculas')
                    sbrn0          = avaliar0.lower()
                else:
                    # print('CASO_05.2: Sobrenome de tamanho 1 ou maior que 3 fica em maiúsculas')
                    sbrn0          = avaliar0.title()
                # print('sbrn0:',sbrn0, len(sbrn0))
                
                try:
                    avaliar1=div[0].split(' ')[1].strip()
                    # print('avaliar0',avaliar0)
                    # print('avaliar1',avaliar1)
                    if 1 < len(avaliar1) <=3:
                        sbrn1     = avaliar1.lower()
                    else:
                        sbrn1     = avaliar1.title()
                    # print('sbrn1:',sbrn1, len(sbrn1))

                except:
                    pass

                if div != []:
                    # print('CASO_05.3: Caso haja divisão por vírgulas na string')
                    try:
                        div_espaco     = div[1].split(' ')
                    except:
                        div_espaco     = div[0].split(' ')
                    sobrenome      = div_espaco[0].strip().upper()
                    try:
                        primeiro_nome  = div_espaco[1].title().strip()
                    except:
                        primeiro_nome  = div_espaco[0].title().strip()
                    if len(sbrn0) == 1:
                        # print('CASO_05.3a: Avalia primeiro sobrenome de tamanho 1')
                        # print('Vai pros nomes:',str(sbrn0).title())
                        nomes_meio = nomes_meio+str(' '+sbrn0.title())
                        # print('   NomesMeio:',nomes_meio)

                    elif 1 < len(sbrn0) <= 3:
                        # print('CASO_05.3b: Avalia primeiro sobrenome 1< tamanho <=3')
                        # print('Vão pros nomes sbrn0:',sbrn0, 'e sbrn1:',sbrn1)

                        div_tresconsoantes = letras_tresconsnts.findall(sobrenome)
                        if div_tresconsoantes != []:
                            # print('CASO_05.4: Três consoantes como sobrenome')
                            for letra in sobrenome:
                                nomes.append(letra)

                            if len(sobrenome) >2:
                                sobrenome=nomes[0]
                            else:
                                sobrenome=nomes[1]
                            nomes.remove(sobrenome)
                            primeiro_nome=nomes[0]
                            nomes_meio=' '.join([str for str in nomes[1:]]).strip()
                            nome_completo=sobrenome.upper()+', '+nomes_meio                
                        
                        try:                       
                            # print(' 05.3b    Lista de Nomes:',nomes_meio)
                            nomes_meio=nomes_meio.replace(sbrn0,'')
                            # print(' 05.3b ReplaceSobrenome0:',nomes_meio)
                            nomes_meio=nomes_meio.replace(sbrn1,'')
                            # print(' 05.3b ReplaceSobrenome1:',nomes_meio)
                        except Exception as e:
                            # print('   Erro ReplaceSobrenome:',e)
                            pass
                        try:
                            nomes_meio.replace(primeiro_nome.title(),'')
                            nomes_meio.replace(primeiro_nome.lower(),'')
                            nomes_meio.replace(primeiro_nome,'')
                            # print(' 05.3b Replace PrimNome:',nomes_meio)
                        except Exception as e:
                            print('Erro no try PrimeiroNome:',e)
                            pass
                        nomes_meio = nomes_meio.replace(sobrenome,'')
                        try:
                            for n,i in enumerate(avaliar1):
                                nomes.append(i.upper())
                                sbrn1     = avaliar1[0]
                            else:
                                sbrn1     = avaliar1.title()
                            # print('sbrn1:',sbrn1, len(sbrn1))
                            nomes_meio = nomes_meio+str(' '+sbrn0)+str(' '+sbrn1)
                        except:
                            nomes_meio = nomes_meio+str(' '+sbrn0)
                        nomes      = nomes_meio.strip().strip(',').split(' ')
                        # print(' 05.3b NomesMeio:',nomes_meio)
                        # print(' 05.3b     Nomes:',nome)

                    else:
                        # print('CASO_05.3c: Avalia primeiro sobrenome >3')
                        nomes_meio = nomes_meio+str(' '+div[0].strip().title())
                        nomes      = nomes_meio.strip().split(' ')
                        # print(' 05.3c NomesMeio:',nomes_meio)
                        # print(' 05.3c     Nomes:',nomes)

                    nomes_meio=nomes_meio.replace(sobrenome,'').replace(',','').strip()
                    nomes_meio=nomes_meio.replace(primeiro_nome,'').strip()

                # print('Ao final do caso 05')
                # print('    Sobrenome com vírgula:',sobrenome, len(sobrenome),'letras')
                # print('Primeiro nome com vírgula:',primeiro_nome, len(primeiro_nome),'letras')
                # print('Nomes do meio com vírgula:',nomes, len(nomes),'nomes')
        
        if sobrenome != '' and primeiro_nome !='':
            nome_completo=sobrenome.upper().replace(',','')+', '+primeiro_nome.replace(',','')+' '+nomes_meio.replace(sobrenome,'').replace(',','')
        elif sobrenome != '':
            nome_completo=sobrenome.upper().replace(',','')+', '+nomes_meio.replace(sobrenome,'').replace(',','')
        else:
            nome_completo=sobrenome.upper()
        
    #     print('Após ajustes finais')
    #     print('     Sobrenome:',sobrenome)
    #     print(' Primeiro Nome:',primeiro_nome)
    #     print('         Nomes:',nomes)
    #     print('     NomesMeio:',nomes_meio)        
            
    #     print('                Resultado:',nome_completo)
        
        return nome_completo.strip()

    def padronizar_nome_antigo(self, linha_texto):
        '''Procura sobrenomes e abreviaturas e monta nome completo
        Recebe: String com todos os sobrenomes e nomes, abreviados ou não
        Retorna: Nome completo no formato padronizado em SOBRENOME AGNOME, Prenomes
        Autor: Marcos Aires (Mar.2022)
        '''
        import unicodedata
        import re
        # print('               Analisando:',linha_texto)
        string = ''.join(ch for ch in unicodedata.normalize('NFKD', linha_texto) if not unicodedata.combining(ch))
        string = string.replace('(Org)','').replace('(Org.)','').replace('(Org).','').replace('.','').replace('\'','')
        string = string.replace(',,,',',').replace(',,',',')
        string = re.sub(r'[0-9]+', '', string)
            
        # Expressões regulares para encontrar padrões de divisão de nomes de autores
        sobrenome_inicio   = re.compile(r'^[A-ZÀ-ú-a-z]+,')                  # Sequência de letras maiúsculas no início da string
        sobrenome_composto = re.compile(r'^[A-ZÀ-ú-a-z]+[ ][A-ZÀ-ú-a-z]+,')  # Duas sequências de letras no início da string, separadas por espaço, seguidas por vírgula
        letra_abrevponto   = re.compile(r'^[A-Z][.]')                        # Uma letra maiúscula no início da string, seguida por ponto
        letra_abrevespaco  = re.compile(r'^[A-Z][ ]')                        # Uma letra maiúscula no início da string, seguida por espaço
        letras_dobradas    = re.compile(r'[A-Z]{2}')                         # Duas letras maiúsculas juntas no início da string, seguida por espaço
        letras_dobradasini = re.compile(r'[A-Z]{2}[ ]')                      # Duas letras maiúsculas juntas no início da string, seguida por espaço
        letras_dobradasfim = re.compile(r'[ ][A-Z]{2}')                      # Duas letras maiúsculas juntas no final da string, precedida por espaço
        letras_duasconsnts = re.compile(r'[B-DF-HJ-NP-TV-XZ]{2}')            # Duas Letras maiúsculas e consoantes juntas
        letras_tresconsnts = re.compile(r'[B-DF-HJ-NP-TV-XZ]{3}')            # Três Letras maiúsculas e consoantes juntas
        
        # Agnomes e preprosições a tratar, agnomes vão maiúsculas para sobrenome e preposições vão para minúsculas nos nomes
        nomes=[]
        agnomes       = ['NETO','JUNIOR','FILHO','SEGUNDO','TERCEIRO']
        preposicoes   = ['da','de','do','das','dos']
        nome_completo = ''
        
        # Ajustar lista de termos, identificar sobrenomes compostos e ajustar sobrenome com ou sem presença de vírgula
        div_sobrenome      = sobrenome_inicio.findall(string)
        div_sbrcomposto    = sobrenome_composto.findall(string)
        
        # print('-'*100)
        # print('                 Recebido:',string)
        
        # Caso haja vírgulas na string, tratar sobrenomes e sobrenomes compostos
        if div_sobrenome != [] or div_sbrcomposto != []:
            print('CASO_01: Há víruglas na string')
            div = string.split(', ')
            sobrenome     = div[0].strip().upper()
            try:
                div_espaco    = div[1].split(' ')
            except:
                div_espaco    = ['']
            primeiro      = div_espaco[0].strip('.')
            
            # print('     Dividir por vírgulas:',div)
            # print('      Primeira DivVirgula:',sobrenome)
            # print('Segunda DivVrg/DivEspaços:',div_espaco)
            # print('      Primeira DivEspaços:',primeiro)
                
            # Caso primeiro nome sejam somente duas letras maiúsculas juntas, trata-se de duas iniciais
            if len(primeiro)==2:
                print('CASO_01.a: Há duas Iniciais')
                primeiro_nome=primeiro[0].strip()
                # print('          C01.a1_PrimNome:',primeiro_nome)
                nomes.append(primeiro[1].strip())
            else:
                print('CASO_01.b: Primeiro nome maior que 2 caracteres')
                primeiro_nome = div_espaco[0].strip().title()
                # print('          C01.a2_PrimNome:',primeiro_nome)
            
            # Montagem da lista de nomes do meio
            for nome in div_espaco:
                print('CASO_01.c: Para cada nome da divisão por espaços após divisão por vírgula')
                print(nome, len(nome), len(nome)==3, nome.lower() not in preposicoes)
                if nome not in nomes and nome.lower()!=primeiro_nome.lower() and nome.lower() not in primeiro_nome.lower() and nome!=sobrenome:   
                    print('CASO_01.c1: Se o nome não está nem como primeiro nome, nem sobrenomes')
                    
                    # Avaliar se é abreviatura seguida de ponto e remover o ponto
                    if len(nome)<=2 and nome.lower() not in preposicoes:
                        print('    C01.c1a_Nome<=02:',nome)
                        for inicial in nome:
                            # print(inicial)
                            if inicial not in nomes and inicial not in primeiro_nome:
                                nomes.append(inicial.replace('.','').strip().title())
                    else:
                        print('    C01.c1b_Nome>3:',nome)
                        if nome not in nomes and nome!=primeiro_nome and nome!=sobrenome and nome!='':
                            if nome.lower() in preposicoes:
                                nomes.append(nome.replace('.','').strip().lower())
                            else:
                                nomes.append(nome.replace('.','').strip().title())
                            # print(nome,'|',primeiro_nome)
                
                if (len(nome)==3 and nome.lower() not in preposicoes):
                    print('    C01.c2_Nome==03:',nome, nomes)
                    print('div_espaco:',div_espaco)
                    for inicial in nome:
                        if inicial not in nomes:
                            nomes.append(inicial.replace('.','').strip().title())      
                    primeiro_nome=nomes[0]
                    nomes.pop(0)
                            
            #caso haja sobrenome composto que não esteja nos agnomes considerar somente primeiro como sobrenome
            if div_sbrcomposto !=[] and sobrenome.split(' ')[1] not in agnomes and sobrenome.split(' ')[0].lower() not in preposicoes:
                print('CASO_01.d: Sobrenome composto sem agnomes')
                print(div_sbrcomposto)
                print('Sobrenome composto:',sobrenome)
                
                nomes.append(sobrenome.split(' ')[1].title())
                sobrenome = sobrenome.split(' ')[0].upper()
                # print('Sobrenome:',sobrenome)
                
                for i in nomes:
                    if i.lower() in sobrenome.lower():
                        nomes.remove(i)
                # print('    Nomes:',nomes)
            
            #caso haja preposição como agnome desconsiderar e passar para final dos nomes
            if div_sbrcomposto !=[] and sobrenome.split(' ')[0].lower() in preposicoes:
                print('CASO_01.e: Preposição no Sobrenome passar para o final dos nomes')
                print('   div_sbrcomposto:', div_sbrcomposto)
                print('Sobrenome composto:',div_sbrcomposto)
                
                nomes.append(div_sbrcomposto[0].split(' ')[0].lower())
                # print('    Nomes:',nomes)
                sobrenome = div_sbrcomposto[0].split(' ')[1].upper().strip(',')
                # print('Sobrenome:',sobrenome)
                
                for i in nomes:
                    print('CASO_01.e1: Para cada nome avaliar se o sobrenome está na lista')
                    if i.lower() in sobrenome.lower():
                        nomes.remove(i)
                # print('  Nomes:',nomes)
            
            print('Ao final do Caso 01')
            print('    Sobrenome com vírgula:',sobrenome, len(sobrenome),'letras')
            print('Primeiro nome com vírgula:',primeiro_nome, len(primeiro_nome),'letras')
            print('           Lista de nomes:',nomes, len(nomes),'nomes')
            
        # Caso não haja vírgulas na string considera sobrenome o último nome da string dividida com espaço vazio
        else:
            # print('CASO_02: Não há víruglas na string')
            try:
                div = string.split(' ')
                # print('      Divisões por espaço:',div)
                
                if div[-1] in agnomes: # nome final é um agnome
                    sobrenome     = div[-2].upper().strip()+' '+div[-1].upper().strip()
                    for i in div[1:-2]:
                        if i not in sobrenome and i not in preposicoes:
                            nomes.append(i.title().strip())
                        if i in preposicoes:
                            nomes.append(i.lower().strip())
                else:
                    if len(div[-1]) > 2:
                        sobrenome     = div[-1].upper().strip()
                        primeiro_nome = div[1].title().strip()
                        for i in div[1:-1]:
                            if i != sobrenome and i not in preposicoes:
                                nomes.append(i.title().strip())
                            if i in preposicoes:
                                nomes.append(i.lower().strip())
                    else:
                        sobrenome     = div[-2].upper().strip()
                        for i in div[-1]:
                            nomes.append(i.title())
                        primeiro_nome = nomes[0].title().strip()
                        for i in div[1:-1]:
                            if i != sobrenome and i not in preposicoes:
                                nomes.append(i.title().strip())
                            if i in preposicoes:
                                nomes.append(i.lower().strip())
            except:
                sobrenome = div[-1].upper().strip()
                for i in div[1:-1]:
                        if i != sobrenome and i not in preposicoes:
                            nomes.append(i.title().strip())
                        if i in preposicoes:
                            nomes.append(i.lower().strip())
                
            if sobrenome.lower() != div[0].lower().strip():
                primeiro_nome=div[0].title().strip()
            else:
                primeiro_nome=''
            
            # print('Ao final do Caso 02')
            # print('    Sobrenome sem vírgula:',sobrenome, len(sobrenome),'letras')
            # print('Primeiro nome sem vírgula:',primeiro_nome, len(primeiro_nome),'letras')
            # print('Nomes do meio sem vírgula:',nomes, len(nomes),'nomes')
        
        # Encontrar e tratar como abreviaturas termos com apenas uma ou duas letras iniciais juntas, com ou sem ponto
        for j in nomes:
            print('CASO_03: Avaliar cada nome armazenado na variável nomes')
            # Procura padrões com expressões regulares na string
            div_sobrenome      = sobrenome_inicio.findall(j)
            div_sbrcomposto    = sobrenome_composto.findall(j)
            div_abrevponto     = letra_abrevponto.findall(j)
            div_abrevespaco    = letra_abrevespaco.findall(j)
            div_ltrdobradasini = letras_dobradasini.findall(j)
            div_ltrdobradasfim = letras_dobradasfim.findall(j)
            div_ltrdobradas    = letras_dobradas.findall(j)
            tamanho=len(j)
            # print('\n', div_ltrdobradasini, div_ltrdobradasfim, tamanho, 'em:',j,len(j))
            
            #caso houver abreviatura com uma letra em maiúscula nos nomes
            if div_abrevponto !=[] or tamanho==1:
                print('CASO_03.1: Há abreviaturas uma letra maiúscula nos nomes')
                nome = j.replace('.','').strip()
                if nome not in nomes and nome != sobrenome and nome != primeiro_nome:
                    print('CASO_03.1a: Há abreviaturas uma letra maiúscula nos nomes')
                    nomes.append(nome.upper())
            
            #caso houver duas inicias juntas em maiúsculas
            elif div_ltrdobradasini !=[] or div_ltrdobradasfim !=[] or div_ltrdobradas !=[] :
                print('CASO_03.2: Há abreviaturas uma letra maiúscula nos nomes')
                for letra in j:
                    print('CASO_03.2a: Avaliar cada inicial do nome')
                    if letra not in nomes and letra != sobrenome and letra != primeiro_nome:
                        print('CASO_03.2a.1: Se não estiver adicionar inicial aos nomes')
                        nomes.append(letra.upper())
            
            #caso haja agnomes ao sobrenome
            elif sobrenome in agnomes:
                print('CASO_03.3: Há agnomes nos sobrenomes')
                sobrenome = nomes[-1].upper()+' '+sobrenome
                # print(sobrenome.split(' '))
                # print('Sobrenome composto:',sobrenome)
                for i in nomes:
                    if i.lower() in sobrenome.lower():
                        nomes.remove(i)
                print('Nomes do meio:',nomes)
                
            else:
                print('CASO_03.4: Não há agnomes nos sobrenomes')
                if (j in nomes and len(j)==1) or j not in nomes and j not in sobrenome and j != primeiro_nome:
                    if len(nomes) == 1:
                        nomes.append(j.upper())
                    elif 1 < len(nomes) <= 3:
                        nomes.append(j.lower())
                    else:
                        nomes.append(j.title())
            
        print('Ao final do Caso 03')
        print('    Sobrenome com vírgula:',sobrenome, len(sobrenome),'letras')
        print('Primeiro nome com vírgula:',primeiro_nome, len(primeiro_nome),'letras')
        print('Nomes do meio com vírgula:',nomes, len(nomes),'nomes')
            
        nomes_meio=' '.join([str for str in nomes]).strip()
        # print('        Qte nomes do meio:',nomes,len(nomes))
        
        if primeiro_nome.lower() == sobrenome.lower():
            print('CASO_04: Primeiro nome é igual ao sobrenome')
            try:
                primeiro_nome=nomes_meio.split(' ')[0]
            except:
                pass
            try:
                nomes_meio.replace(sobrenome,'')
            except:
                pass
        
            # print('Ao final do caso 04')
            # print('    Sobrenome com vírgula:',sobrenome, len(sobrenome),'letras')
            # print('Primeiro nome com vírgula:',primeiro_nome, len(primeiro_nome),'letras')
            # print('Nomes do meio com vírgula:',nomes, len(nomes),'nomes')
        
        # Caso sobrenome só de 1 letra passá-lo para nomes e considerar o próximo nome como sobrenome
        for i in range(len(div)):
            if len(sobrenome)==1 or sobrenome.lower() in preposicoes:
                print('CASO_05: Mudar sobrenomes até o adequado')
                div    = string.split(', ')
                # print('Divisão por vírgulas:',div)
                avaliar0       = div[0].split(' ')[0].strip()
                if 1< len(avaliar0) < 3:
                    print('CASO_05.1: 1 < Sobrenome < 3 fica em minúsculas')
                    sbrn0          = avaliar0.lower()
                else:
                    print('CASO_05.2: Sobrenome de tamanho 1 ou maior que 3 fica em maiúsculas')
                    sbrn0          = avaliar0.title()
                # print('sbrn0:',sbrn0, len(sbrn0))

                try:
                    avaliar1=div[0].split(' ')[1].strip()
                    # print('avaliar0',avaliar0)
                    # print('avaliar1',avaliar1)
                    if 1 < len(avaliar1) <=3:
                        sbrn1     = avaliar1.lower()
                    else:
                        sbrn1     = avaliar1.title()
                    # print('sbrn1:',sbrn1, len(sbrn1))

                except:
                    pass

                if div != []:
                    print('CASO_05.3: Caso haja divisão por vírgulas na string')
                    try:
                        div_espaco     = div[1].split(' ')
                    except:
                        div_espaco     = div[0].split(' ')
                    sobrenome      = div_espaco[0].strip().upper()
                    try:
                        primeiro_nome  = div_espaco[1].title().strip()
                    except:
                        primeiro_nome  = div_espaco[0].title().strip()
                    if len(sbrn0) == 1:
                        print('CASO_05.3a: Avalia primeiro sobrenome de tamanho 1')
                        # print('Vai pros nomes:',str(sbrn0).title())
                        nomes_meio = nomes_meio+str(' '+sbrn0.title())
                        # print('   NomesMeio:',nomes_meio)

                    elif 1 < len(sbrn0) <= 3:
                        print('CASO_05.3b: Avalia primeiro sobrenome 1< tamanho <=3')
                        # print('Vão pros nomes sbrn0:',sbrn0, 'e sbrn1:',sbrn1)

                        div_tresconsoantes = letras_tresconsnts.findall(sobrenome)
                        if div_tresconsoantes != []:
                            print('CASO_05.4: Três consoantes como sobrenome')
                            for letra in sobrenome:
                                nomes.append(letra)

                            if len(sobrenome) >2:
                                sobrenome=nomes[0]
                            else:
                                sobrenome=nomes[1]
                            nomes.remove(sobrenome)
                            primeiro_nome=nomes[0]
                            nomes_meio=' '.join([str for str in nomes[1:]]).strip()
                            nome_completo=sobrenome.upper()+', '+nomes_meio                
                        
                        try:                       
                            # print(' 05.3b    Lista de Nomes:',nomes_meio)
                            nomes_meio=nomes_meio.replace(sbrn0,'')
                            # print(' 05.3b ReplaceSobrenome0:',nomes_meio)
                            nomes_meio=nomes_meio.replace(sbrn1,'')
                            # print(' 05.3b ReplaceSobrenome1:',nomes_meio)
                        except Exception as e:
                            # print('   Erro ReplaceSobrenome:',e)
                            pass
                        try:
                            nomes_meio.replace(primeiro_nome.title(),'')
                            nomes_meio.replace(primeiro_nome.lower(),'')
                            nomes_meio.replace(primeiro_nome,'')
                            # print(' 05.3b Replace PrimNome:',nomes_meio)
                        except Exception as e:
                            # print('Erro no try PrimeiroNome:',e)
                            pass
                        nomes_meio = nomes_meio.replace(sobrenome,'')
                        try:
                            for n,i in enumerate(avaliar1):
                                nomes.append(i.upper())
                                sbrn1     = avaliar1[0]
                            else:
                                sbrn1     = avaliar1.title()
                            # print('sbrn1:',sbrn1, len(sbrn1))
                            nomes_meio = nomes_meio+str(' '+sbrn0)+str(' '+sbrn1)
                        except:
                            nomes_meio = nomes_meio+str(' '+sbrn0)
                        nomes      = nomes_meio.strip().strip(',').split(' ')
                        # print(' 05.3b NomesMeio:',nomes_meio)
                        # print(' 05.3b     Nomes:',nome)

                    else:
                        print('CASO_05.3c: Avalia primeiro sobrenome >3')
                        nomes_meio = nomes_meio+str(' '+div[0].strip().title())
                        nomes      = nomes_meio.strip().split(' ')
                        # print(' 05.3c NomesMeio:',nomes_meio)
                        # print(' 05.3c     Nomes:',nomes)

                    nomes_meio=nomes_meio.replace(sobrenome,'').replace(',','').strip()
                    nomes_meio=nomes_meio.replace(primeiro_nome,'').strip()

                # print('Ao final do caso 05')
                # print('    Sobrenome com vírgula:',sobrenome, len(sobrenome),'letras')
                # print('Primeiro nome com vírgula:',primeiro_nome, len(primeiro_nome),'letras')
                # print('Nomes do meio com vírgula:',nomes, len(nomes),'nomes')
        
        if sobrenome != '' and primeiro_nome !='':
            nome_completo=sobrenome.upper().replace(',','')+', '+primeiro_nome.replace(',','')+' '+nomes_meio.replace(sobrenome,'').replace(',','')
        elif sobrenome != '':
            nome_completo=sobrenome.upper().replace(',','')+', '+nomes_meio.replace(sobrenome,'').replace(',','')
        else:
            nome_completo=sobrenome.upper()
        
        # print('Após ajustes finais')
        # print('     Sobrenome:',sobrenome)
        # print(' Primeiro Nome:',primeiro_nome)
        # print('         Nomes:',nomes)
        # print('     NomesMeio:',nomes_meio)        
            
        # print('                Resultado:',nome_completo)
        
        return nome_completo.strip()

    def iniciais_nome(self, linha_texto):
        '''Função para retornar sobrenome+iniciais dos nomes, na forma: SOBRENOME, X Y Z
        Recebe: String com nome
        Retorna: Tupla com nome e sua versão padronizada em sobrenome+agnomes em maiúsculas, seguida de vírgula e iniciais dos nomes 
        Autor: Marcos Aires (Mar.2022)
        '''
        import unicodedata
        import re
        # print('               Analisando:',linha_texto)
        string = ''.join(ch for ch in unicodedata.normalize('NFKD', linha_texto) if not unicodedata.combining(ch))
        string = string.replace('(Org)','').replace('(Org.)','').replace('(Org).','').replace('.','')
            
        # Expressões regulares para encontrar padrões de divisão de nomes de autores
        sobrenome_inicio   = re.compile(r'^[A-ZÀ-ú-a-z]+,')                 # Sequência de letras maiúsculas no início da string
        sobrenome_composto = re.compile(r'^[A-ZÀ-ú-a-z]+[ ][A-ZÀ-ú-a-z]+,') # Duas sequências de letras no início da string, separadas por espaço, seguidas por vírgula
        letra_abrevponto   = re.compile(r'^[A-Z][.]')                       # Uma letra maiúscula no início da string, seguida por ponto
        letra_abrevespaco  = re.compile(r'^[A-Z][ ]')                       # Uma letra maiúscula no início da string, seguida por espaço
        letras_dobradas    = re.compile(r'[A-Z]{2}')                        # Duas letras maiúsculas juntas no início da string, seguida por espaço
        letras_dobradasini = re.compile(r'[A-Z]{2}[ ]')                     # Duas letras maiúsculas juntas no início da string, seguida por espaço
        letras_dobradasfim = re.compile(r'[ ][A-Z]{2}')                     # Duas letras maiúsculas juntas no final da string, precedida por espaço
            
        nomes=[]
        agnomes       = ['NETO','JUNIOR','FILHO','SEGUNDO','TERCEIRO']
        preposicoes   = ['da','de','do','das','dos','DA','DE','DOS','DAS','DOS','De']
        nome_completo = ''
        
        # Ajustar lista de termos, identificar sobrenomes compostos e ajustar sobrenome com ou sem presença de vírgula
        div_sobrenome      = sobrenome_inicio.findall(string)
        div_sbrcomposto    = sobrenome_composto.findall(string)
        
        # Caso haja vírgulas na string, tratar sobrenomes e sobrenomes compostos
        if div_sobrenome != [] or div_sbrcomposto != []:
            div   = string.split(', ')
            sobrenome     = div[0].strip().upper()
            try:
                div_espaco    = div[1].split(' ')
            except:
                div_espaco  = ['']
            primeiro      = div_espaco[0].strip('.')
            
            # Caso primeiro nome sejam somente duas letras maiúsculas juntas, trata-se de duas iniciais
            if len(primeiro)==2:
                primeiro_nome=primeiro[0].strip()
                nomes.append(primeiro[1].strip())
            else:
                primeiro_nome = div_espaco[0].strip().title()
            
            # Montagem da lista de nomes do meio
            for nome in div_espaco:
                if nome not in nomes and nome.lower()!=primeiro_nome.lower() and nome.lower() not in primeiro_nome.lower() and nome!=sobrenome:   
                    # print(nome, len(nome))
                    
                    # Avaliar se é abreviatura seguida de ponto e remover o ponto
                    if len(nome)<=2 and nome.lower() not in preposicoes:
                        for inicial in nome:
                            # print(inicial)
                            if inicial not in nomes and inicial not in primeiro_nome:
                                nomes.append(inicial.replace('.','').strip().title())
                    else:
                        if nome not in nomes and nome!=primeiro_nome and nome!=sobrenome and nome!='':
                            if nome.lower() in preposicoes:
                                nomes.append(nome.replace('.','').strip().lower())
                            else:
                                nomes.append(nome.replace('.','').strip().title())
                            # print(nome,'|',primeiro_nome)
                            
            #caso haja sobrenome composto que não esteja nos agnomes considerar somente primeiro como sobrenome
            if div_sbrcomposto !=[] and sobrenome.split(' ')[1] not in agnomes:
                # print(div_sbrcomposto)
                # print('Sobrenome composto:',sobrenome)
                nomes.append(sobrenome.split(' ')[1].title())
                sobrenome = sobrenome.split(' ')[0].upper()
                # print('Sobrenome:',sobrenome.split(' '))
                for i in nomes:
                    if i.lower() in sobrenome.lower():
                        nomes.remove(i)
                # print('Nomes do meio:',nomes)
            
            # print('    Sobrenome com vírgula:',sobrenome, len(sobrenome),'letras')
            # print('Primeiro nome com vírgula:',primeiro_nome, len(primeiro_nome),'letras')
            # print('Nomes do meio com vírgula:',nomes, len(nomes),'nomes')
            
        # Caso não haja vírgulas na string considera sobrenome o último nome da string dividida com espaço vazio
        else:
            try:
                div       = string.split(' ')
                if div[-2] in agnomes:
                    sobrenome = div[-2].upper()+' '+div[-1].strip().upper()
                    for i in nomes[1:-2]:
                        if i not in sobrenome and i not in preposicoes:
                            nomes.append(i.strip().title())
                        if i in preposicoes:
                            nomes.append(i.strip().lower())
                else:
                    sobrenome = div[-1].strip().upper()
                    for i in div[1:-1]:
                        if i not in sobrenome and i not in preposicoes:
                            nomes.append(i.strip().title())
                        if i in preposicoes:
                            nomes.append(i.strip().lower())
            except:
                sobrenome = div[-1].strip().upper()
                for i in div[1:-1]:
                        if i not in sobrenome and i not in preposicoes:
                            nomes.append(i.strip().title())
                        if i in preposicoes:
                            nomes.append(i.strip().lower())
                
            if sobrenome.lower() != div[0].strip().lower():
                primeiro_nome=div[0].strip().title()
            else:
                primeiro_nome=''
            
            # print('    Sobrenome sem vírgula:',sobrenome)
            # print('Primeiro nome sem vírgula:',primeiro_nome)
            # print('Nomes do meio sem vírgula:',nomes)
        
        # Encontrar e tratar como abreviaturas termos com apenas uma ou duas letras iniciais juntas, com ou sem ponto
        for j in nomes:
            # Procura padrões com expressões regulares na string
            div_sobrenome      = sobrenome_inicio.findall(j)
            div_sbrcomposto    = sobrenome_composto.findall(j)
            div_abrevponto     = letra_abrevponto.findall(j)
            div_abrevespaco    = letra_abrevespaco.findall(j)
            div_ltrdobradasini = letras_dobradasini.findall(j)
            div_ltrdobradasfim = letras_dobradasfim.findall(j)
            div_ltrdobradas    = letras_dobradas.findall(j)
            tamanho=len(j)
            # print('\n', div_ltrdobradasini, div_ltrdobradasfim, tamanho, 'em:',j,len(j))
            
            #caso houver abreviatura com uma letra em maiúscula nos nomes
            if div_abrevponto !=[] or tamanho==1:
                cada_nome = j.replace('.','').strip()
                if cada_nome not in nomes and cada_nome != sobrenome and nome != primeiro_nome:
                    nomes.append(cada_nome)
            
            #caso houver duas inicias juntas em maiúsculas
            elif div_ltrdobradasini !=[] or div_ltrdobradasfim !=[] or div_ltrdobradas !=[] :
                for letra in j:
                    if letra not in nomes and letra != sobrenome and letra != primeiro_nome:
                        nomes.append(letra)
            
            #caso haja agnomes ao sobrenome
            elif sobrenome in agnomes:
                sobrenome = nomes[-1].upper()+' '+sobrenome
                # print(sobrenome.split(' '))
                # print('Sobrenome composto:',sobrenome)
                for i in nomes:
                    if i.lower() in sobrenome.lower():
                        nomes.remove(i)
                # print('Nomes do meio:',nomes)
                
            else:
                if j not in nomes and j not in sobrenome and j != primeiro_nome:
                    nomes.append(j)
        
        nomes_meio=' '.join([str[0] for str in nomes]).strip()
        # print('Qte nomes do meio',len(nomes),nomes)
        if sobrenome != '' and primeiro_nome !='':
            sobrenome_iniciais = sobrenome+', '+primeiro_nome[0]+' '+nomes_meio
        elif sobrenome != '':
            sobrenome_iniciais = sobrenome
        
        return sobrenome_iniciais.strip()

    def similares(self, lista_autores, lista_grupo, limite_jarowinkler, distancia_levenshtein):
        """Função para aplicar padronização no nome de autor da lista de pesquisadores e buscar similaridade na lista de coautores
        Recebe: Lista de pesquisadores do grupo em análise gerada pela lista de nomes dos coautores das publicações em análise
        Utiliza: get_jaro_distance(), editdistance()
        Retorna: Lista de autores com fusão de nomes cuja similaridade esteja dentro dos limites definidos nesta função
        Autor: Marcos Aires (Fev.2022)
        
        Refazer: Inserir crítica de, mantendo sequência ordem alfabética, retornar no final nome mais extenso em caso de similaridade;
        """
        from pyjarowinkler.distance import get_jaro_distance
        from IPython.display import clear_output
        import editdistance
        import numpy as np
        import time
        
        t0=time.time()
        
        # limite_jarowinkler=0.85
        # distancia_levenshtein=6
        similares_jwl=[]
        similares_regras=[]
        similares=[]
        tempos=[]
        
        count=0
        t1=time.time()
        for i in lista_autores:
            count+=1
            if count > 0:
                tp=time.time()-t1
                tmed=tp/count*2
                tempos.append(tp)
        #     print("Analisar similaridades com: ", nome_padronizado)
            
            count1=0
            for nome in lista_autores:
                if count1 > 0:
                    resta=len(lista_autores)-count
                    print(f'Analisando {count1:3}/{len(lista_autores)} resta analisar {resta:3} nomes. Previsão de término em {np.round(tmed*resta/60,1)} minutos')
                else:
                    print(f'Analisando {count1:3}/{len(lista_autores)} resta analisar {len(lista_autores)-count1} nomes.')
                
                t2=time.time()
                count1+=1            

                try:
                    similaridade_jarowinkler = get_jaro_distance(i, nome)
                    print(f'{i:40} | {nome:40} | Jaro-Winkler: {np.round(similaridade_jarowinkler,2):4} Levenshtein: {editdistance.eval(i, nome)}')
                    similaridade_levenshtein = editdistance.eval(i, nome)

                    # inferir similaridade para nomes que estejam acima do limite ponderado definido, mas não idênticos e não muito distantes em edição
                    if  similaridade_jarowinkler > limite_jarowinkler and similaridade_jarowinkler!=1 and similaridade_levenshtein < distancia_levenshtein:
                        # Crítica no nome mais extenso como destino no par (origem, destino)
                        
                        similares_jwl.append((i,nome))

                except:
                    pass

                clear_output(wait=True)
        
        # Conjunto de regras de validação de similaridade
        # Monta uma lista de nomes a serem retirados antes de montar a lista de troca
        trocar=[]
        retirar=[]
        for i in similares_jwl:
            sobrenome_i = i[0].split(',')[0]
            sobrenome_j = i[1].split(',')[0]

            try:
                iniciais_i  = self.iniciais_nome(i[0]).split(',')[1].strip()
            except:
                iniciais_i  = ''

            try:
                iniciais_j  = self.iniciais_nome(i[1]).split(',')[1].strip()
            except:
                iniciais_j  = ''

            try:
                primnome_i = i[0].split(',')[1].strip().split(' ')[0].strip()
            except:
                primnome_i = ''

            try:
                primnome_j = i[1].split(',')[1].strip().split(' ')[0].strip()
            except:
                primnome_j = ''    

            try:
                inicial_i = i[0].split(',')[1].strip()[0]
            except:
                inicial_i = ''

            try:
                resto_i   = i[0].split(',')[1].strip().split(' ')[0][1:]
            except:
                resto_i   = ''

            try:
                inicial_j = i[1].split(',')[1].strip()[0]
            except:
                inicial_j = ''

            try:
                resto_j   = i[1].split(',')[1].strip().split(' ')[0][1:]
            except:
                resto_j = ''

            # Se a distância de edição entre os sobrenomes
            if editdistance.eval(sobrenome_i, sobrenome_j) > 2 or inicial_i!=inicial_j:
                retirar.append(i)
            else:
                if primnome_i!=primnome_j and len(primnome_i)>1:
                    retirar.append(i)
                if primnome_i!=primnome_j and len(primnome_i)>1 and len(primnome_j)>1:
                    retirar.append(i)
                if resto_i!=resto_j and resto_i!='':
                    retirar.append(i)
                if len(i[1]) < len(i[0]):
                    retirar.append(i)
                if len(iniciais_i) != len(iniciais_j):
                    retirar.append(i)

        for i in similares_jwl:
            if i not in retirar:
                trocar.append(i)

            if self.iniciais_nome(i[0]) in self.iniciais_nome(i[1]) and len(i[0]) < len(i[1]):
                trocar.append(i)

            if self.iniciais_nome(i[0]) == self.iniciais_nome(i[1]) and len(i[0]) < len(i[1]):
                trocar.append(i)

        
        lista_extra = [
                        # ('ALBUQUERQUE, Adriano B', 'ALBUQUERQUE, Adriano Bessa'),
                        # ('ALBUQUERQUE, Adriano', 'ALBUQUERQUE, Adriano Bessa'),
                        # ('COELHO, Andre L V', 'COELHO, Andre Luis Vasconcelos'),
                        # ('DUARTE, Joao B F', 'DUARTE, Joao Batista Furlan'),
                        # ('FILHO, Raimir H','HOLANDA FILHO, Raimir'),
                        # ('FILHO, Raimir','HOLANDA FILHO, Raimir'),
                        # ('FORMIGO, A','FORMICO, Maria Andreia Rodrigues'),
                        # ('FORMICO, A','FORMICO, Maria Andreia Rodrigues'),
                        # ('FURLAN, J B D', 'FURLAN, Joao Batista Duarte'),
                        # ('FURTADO, Elizabeth', 'FURTADO, Maria Elizabeth Sucupira'),
                        # ('FURTADO, Elizabeth S', 'FURTADO, Maria Elizabeth Sucupira'),
                        # ('FURTADO, Elizabeth Sucupira','FURTADO, Maria Elizabeth Sucupira'),
                        # ('FURTADO, M E S', 'FURTADO, Maria Elizabeth Sucupira'),
                        # ('FURTADO, Vasco', 'FURTADO, Joao Jose Vasco Peixoto'),
                        # ('FURTADO, J P', 'FURTADO, Joao Jose Vasco Peixoto'),
                        # ('FURTADO, J V P', 'FURTADO, Joao Jose Vasco Peixoto'),
                        # ('FURTADO, Vasco', 'FURTADO, Joao Jose Vasco Peixoto'),
                        # ('FURTADO, Elizabeth','FURTADO, Maria Elizabeth Sucupira'),
                        # ('HOLANDA, Raimir', 'HOLANDA FILHO, Raimir'),
                        # ('LEITE, G S', 'LEITE, Gleidson Sobreira'),
                        # ('PEQUENO, T H C', 'PEQUENO, Tarcisio Haroldo Cavalcante'),
                        # ('PEQUENO, Tarcisio','PEQUENO, Tarcisio Haroldo Cavalcante'),
                        # ('PEQUENO, Tarcisio Cavalcante', 'PEQUENO, Tarcisio Haroldo Cavalcante'),
                        # ('PINHEIRO, Placido R', 'PINHEIRO, Placido Rogerio'),
                        # ('PINHEIRO, Vladia', 'PINHEIRO, Vladia Celia Monteiro'),
                        # ('RODRIGUES, M A F', 'RODRIGUES, Maria Andreia Formico'),
                        # ('RODRIGUES, Andreia', 'RODRIGUES, Maria Andreia Formico'),
                        # ('JOAO, Batista F Duarte,', 'FURLAN, Joao Batista Duarte'),
                        # ('MACEDO, Antonio Roberto M de', 'MACEDO, Antonio Roberto Menescal de'),
                        # ('MACEDO, D V', 'MACEDO, Daniel Valente'),
                        # ('MENDONCA, Nabor C', 'MENDONCA, Nabor das Chagas'),
                        # ('PEQUENO, Tarcisio', 'PEQUENO, Tarcisio Haroldo Cavalcante'),
                        # ('PEQUENO, Tarcisio H', 'PEQUENO, Tarcisio Haroldo Cavalcante'),
                        # ('PINHEIRO, Mirian C D', 'PINHEIRO, Miriam Caliope Dantas'),
                        # ('PINHEIRO, Mirian Caliope Dantas', 'PINHEIRO, Miriam Caliope Dantas'),
                        # ('PINHEIRO, P G C D', 'PINHEIRO, Pedro Gabriel Caliope Dantas'),
                        # ('PINHEIRO, Pedro G C', 'PINHEIRO, Pedro Gabriel Caliope Dantas'),
                        # ('PINHEIRO, Placido R', 'PINHEIRO, Placido Rogerio'),
                        # ('PINHEIRO, Vladia', 'PINHEIRO, Vladia Celia Monteiro'),
                        # ('ROGERIO, Placido Pinheiro', 'PINHEIRO, Placido Rogerio'),
                        # ('REBOUCRAS FILHO, Pedro', 'REBOUCAS FILHO, Pedro Pedrosa'),
                        # ('SAMPAIO, A', 'SAMPAIO, Americo Tadeu Falcone'),
                        # ('SAMPAIO, Americo', 'SAMPAIO, Americo Tadeu Falcone'),
                        # ('SAMPAIO, Americo Falcone', 'SAMPAIO, Americo Tadeu Falcone'),
                        # ('SUCUPIRA, Elizabeth Furtado','FURTADO, Maria Elizabeth Sucupira'),
                    ]
        
        trocar=trocar+lista_extra
        trocar.sort()
        
        return trocar


    def extrair_variantes(self, df_dadosgrupo):
        ''' Utiliza campo de Nome em Citações do currículo como filtro para obter variantes do nome de cada membro
        Recebe: Dataframe com os dados brutos do grupo de pesquisa agrupados; lista de nomes de pesquisadores de interesse
        Retorna: Lista de tuplas com pares a serem trocados da variante pelo nome padronizado na forma (origem, destino)
        '''
        filtro1   = 'Nome'
        lista_nomes = df_dadosgrupo[(df_dadosgrupo.ROTULOS == filtro1)]['CONTEUDOS'].values

        variantes=[]
        filtro='Nome em citações bibliográficas'
        variantes=df_dadosgrupo[(df_dadosgrupo.ROTULOS == filtro)]['CONTEUDOS'].to_list()

        trocar=[]
        for j in range(len(variantes)):
            padrao_destino = self.padronizar_nome(lista_nomes[j])
            trocar.append((lista_nomes[j], padrao_destino))
            for k in variantes[j]:
                padrao_origem = self.padronizar_nome(k)
                trocar.append((k, padrao_destino))
                trocar.append((padrao_origem, padrao_destino))
        
        return trocar


    def inferir_variantes(self, nome):
        ''' Quebra um nome inicialmente por vírgula para achar sobrenomes, e depois por ' ' para achar nomes
        Recebe: Par de nomes a comparar, nome1 é nome padronizado na função padronizar_nome(), nome2 é o que será analisado
        Utiliza: Função padronizar_nome(nome)
        Retorna: Lista de tuplas, no formato (origem, destino), com variantes de nome a serem trocadas pela forma padronizada
        Autor: Marco Aires (Fev.2022)
        '''
        trocar = []
        nomes  = []
        try:
            div0  = nome.split(',').strip()
            sobrenome=div0[0]
            try:
                div1 = div0[1].split(' ').strip()
                for i in div1:
                    nomes.append(i)
            except:
                pass
            
        except:
            pass
        
        trocar.append((nome, self.iniciais_nome(nome)))
        
        return trocar


    def comparar_nomes(self, nome1,nome2):
        ''' Compara dois nomes por seus sobrenomes e iniciais do primeiro nome
        Recebe: Par de nomes a comparar, nome1 é nome padronizado na função padronizar_nome(), nome2 é o que será analisado
        Utiliza: Função padronizar_nome(nome)
        Retorna: Lista de tuplas, no formato (origem, destino), com variantes de nome a serem trocadas pela forma padronizada
        Autor: Marco Aires (Fev.2022)
        '''
        trocar=[]
        qte_nomes1=0
        nome_padronizado1 = self.padronizar_nome(nome1)
        sobrenome1        = nome_padronizado1.split(',')[0]
        if sobrenome1!='':
            qte_nomes1+=1
        primeiro_nome1    = nome_padronizado1.split(',')[1].split(' ')[0]
        if primeiro_nome1!='':
            qte_nomes1+=1
        inicial_primnome1 = primeiro_nome1[0]
        demais_nomes1     = nome_padronizado1.split(',')[1].split(' ')[1:]
        qte_nomes1=qte_nomes1+len(demais_nomes1)
        
        qte_nomes2=0
        nome_padronizado2 = self.padronizar_nome(nome2)
        sobrenome2        = nome_padronizado2.split(',')[0]
        if sobrenome2!='':
            qte_nomes2+=1    
        primeiro_nome2    = nome_padronizado2.split(',')[1].split(' ')[0]
        if primeiro_nome2!='':
            qte_nomes2+=1
        inicial_primnome2 = primeiro_nome2[0]
        demais_nomes2     = nome_padronizado2.split(',')[1].split(' ')[1:]
        qte_nomes2=qte_nomes2+len(demais_nomes2)
        
        if sobrenome1==sobrenome2 and primeiro_nome1==primeiro_nome2:
            trocar.append((nome1,nome_padronizado2))

        if sobrenome1==sobrenome2 and primeiro_nome1==primeiro_nome2:
            trocar.append((nome1,nome_padronizado2))
            
        return trocar