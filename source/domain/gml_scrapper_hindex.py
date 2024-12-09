from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
# from chromedriver_manager import ChromeDriverManager # não usar esta forma
from bs4 import BeautifulSoup
import time
import random
import pandas as pd

class GoogleScholarScraper:
    def __init__(self, affiliation_terms=None):
        chrome_options = Options()
        user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
            "Mozilla/5.0 (X11; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/115.0"
        ]
        chrome_options.add_argument(f"user-agent={random.choice(user_agents)}")
        service = Service(ChromeDriverManager().install())
        self.driver = webdriver.Chrome(service=service, options=chrome_options)
        self.wait = WebDriverWait(self.driver, 10)
        self.affiliation_terms = affiliation_terms or []
        self.ausentes=[]

    def match_affiliation(self, affiliation_text):
        if not self.affiliation_terms:
            return True
        return any(term.lower() in affiliation_text.lower() for term in self.affiliation_terms)

    def get_researchers_by_institution(self, institution_name):
        """Extrai lista de pesquisadores de uma instituição usando paginação"""
        researchers = []
        start = 0
        has_next_page = True
        
        try:
            while has_next_page:
                # Pausa aleatória entre 3 e 7 segundos antes de cada nova requisição
                time.sleep(random.uniform(3, 7))
                
                # Constrói a URL com o parâmetro de paginação correto
                if start == 0:
                    search_url = f"https://scholar.google.com/citations?view_op=search_authors&mauthors={institution_name.replace(' ', '+')}&hl=en"
                else:
                    search_url = f"https://scholar.google.com/citations?view_op=search_authors&mauthors={institution_name.replace(' ', '+')}&hl=en&start={start}"
                
                self.driver.get(search_url)
                
                # Pausa após carregar a página
                time.sleep(2)
                
                # Aguarda carregamento dos resultados
                profiles = self.wait.until(
                    EC.presence_of_all_elements_located((By.CLASS_NAME, "gs_ai_chpr"))
                )
                
                if not profiles:
                    break
                    
                for profile in profiles:
                    # Pausa curta entre processamento de cada perfil
                    time.sleep(random.uniform(0.5, 1))
                    
                    researcher_data = {
                        "nome": "",
                        "afiliacao": "",
                        "email": "",
                        "areas": [],
                        "citacoes": 0
                    }
                    
                    try:
                        name_element = profile.find_element(By.CLASS_NAME, "gs_ai_name")
                        researcher_data["nome"] = name_element.text
                        researcher_data["afiliacao"] = profile.find_element(By.CLASS_NAME, "gs_ai_aff").text
                        
                        if institution_name.lower() in researcher_data["afiliacao"].lower():
                            # Extrai email verificado
                            try:
                                email_element = profile.find_element(By.CLASS_NAME, "gs_ai_eml")
                                researcher_data["email"] = email_element.text.replace("Verified email at ", "")
                            except:
                                pass
                            
                            # Extrai áreas de interesse
                            try:
                                areas = profile.find_elements(By.CLASS_NAME, "gs_ai_one_int")
                                researcher_data["areas"] = [area.text for area in areas]
                            except:
                                pass
                            
                            # Extrai citações
                            try:
                                citations = profile.find_element(By.CLASS_NAME, "gs_ai_cby").text
                                researcher_data["citacoes"] = int(citations.split()) if 'Cited by' in citations else 0
                            except:
                                pass
                            
                            researchers.append(researcher_data)
                    except:
                        continue
                
                # Verifica se existe próxima página
                try:
                    next_button = self.driver.find_element(By.CSS_SELECTOR, ".gs_btnPR:not(.gs_dis)")
                    if 'disabled' in next_button.get_attribute('class'):
                        has_next_page = False
                    else:
                        start += 10
                except:
                    has_next_page = False
            
            return researchers
            
        except Exception as e:
            print(f"Erro ao extrair pesquisadores da instituição {institution_name}: {e}")
            return researchers

    def get_researcher_data(self, researcher_name, verbose=True):
        search_url = f"https://scholar.google.com/citations?hl=en&view_op=search_authors&mauthors={researcher_name.replace(' ', '+')}"
        self.driver.get(search_url)

        try:
            # Esperar até que a lista de resultados apareça
            self.wait.until(EC.presence_of_element_located((By.CLASS_NAME, "gsc_1usr")))
            
            # Encontrar todos os perfis na página
            profiles = self.driver.find_elements(By.CLASS_NAME, "gs_ai_chpr")
            
            for profile in profiles:
                try:
                    # Extrair nome e afiliação do perfil
                    name = profile.find_element(By.CLASS_NAME, "gs_ai_name").text
                    affiliation = profile.find_element(By.CLASS_NAME, "gs_ai_aff").text
                    
                    # Verificar se o nome e a afiliação correspondem
                    if researcher_name.lower() in name.lower() and self.match_affiliation(affiliation):
                        if verbose:
                            print(f"{researcher_name}")
                            print(f"  {affiliation}")                            
                        initial_data = {
                            "nome": name,
                            "afiliacao": affiliation,
                            "dominio": "",
                            "citacoes": 0,
                            "areas": []
                        }
                        
                        # Extrair dominio de email verificado
                        try:
                            dominio = profile.find_element(By.CLASS_NAME, "gs_ai_eml").text
                            if verbose:
                                print(f"  Afiliação confirmada em {dominio}")
                            initial_data["dominio"] = dominio.replace("Verified email at ", "")
                        except:
                            pass
                            
                        # Extrair número de citações
                        try:
                            citations = profile.find_element(By.CLASS_NAME, "gs_ai_cby").text
                            # if verbose:
                            #     print(f"  Total de citações: {citations}")                            
                            initial_data["citacoes"] = int(''.join(filter(str.isdigit, citations)))
                        except:
                            pass
                            
                        # Extrair áreas de interesse
                        try:
                            areas = profile.find_elements(By.CLASS_NAME, "gs_ai_one_int")
                            if verbose:
                                lista_areas = [area.text for area in areas]
                                print(f"  Áreas de atuação: {lista_areas}")
                            initial_data["areas"] = [area.text for area in areas]
                        except:
                            pass

                        # Clicar no nome do pesquisador para obter dados adicionais
                        try:
                            # Localizar diretamente o link dentro da div gs_ai_name
                            name_link = profile.find_element(By.CSS_SELECTOR, ".gs_ai_name a")
                            name_link.click()                            
                        except:
                            print("  Erro ao abrir página do pesquisador")
                        
                        # Aguardar carregamento do perfil completo
                        self.wait.until(EC.presence_of_element_located((By.ID, "gsc_rsb_st")))
                        
                        # Extrair dados adicionais do perfil
                        profile_data = self.extract_profile_stats()
                        initial_data.update(profile_data)
                        if verbose:
                            print("="*125)                        
                        return initial_data
                    
                except Exception as e:
                    print(f"  Erro ao processar perfil: {e}")
                    continue
            
            print(f"Nenhum perfil encontrado para {researcher_name} com a afiliação especificada")
            return None

        except Exception as e:
            print(f"Não disponível: {researcher_name}")
            self.ausentes.append(researcher_name)
            return None

    def safe_int_conversion(self, text):
        """Converter texto para inteiro de forma segura"""
        try:
            # Remove espaços e verifica se está vazio
            text = text.strip()
            if not text:
                return 0
            return int(text)
        except (ValueError, AttributeError):
            return 0

    def extract_profile_stats(self):
        """Extrair estatísticas do perfil completo"""
        try:
            # Localizar a tabela de métricas
            stats_table = self.wait.until(EC.presence_of_element_located((By.ID, "gsc_rsb_st")))
            
            # Extrair as células com as métricas
            metric_cells = stats_table.find_elements(By.CLASS_NAME, "gsc_rsb_std")
            
            # As células estão organizadas em ordem: 
            # [0-1]: Citações (All, Since 2019)
            # [2-3]: h-index (All, Since 2019)
            # [4-5]: i10-index (All, Since 2019)           
            stats = {
                "total_citacoes": self.safe_int_conversion(metric_cells[0].text),
                "citacoes_desde_2019": self.safe_int_conversion(metric_cells[1].text),
                "indice_h": self.safe_int_conversion(metric_cells[2].text),
                "indice_h_desde_2019": self.safe_int_conversion(metric_cells[3].text),
                "indice_i10": self.safe_int_conversion(metric_cells[4].text),
                "indice_i10_desde_2019": self.safe_int_conversion(metric_cells[5].text)
            }
            
            return stats
            
        except Exception as e:
            print(f"Erro ao extrair estatísticas do perfil: {e}")
            return {
                "total_citacoes": 0,
                "citacoes_desde_2019": 0,
                "indice_h": 0,
                "indice_h_desde_2019": 0,
                "indice_i10": 0,
                "indice_i10_desde_2019": 0
            }

    # def extract_profile_stats(self):
    #     """Extrai estatísticas do perfil completo"""
    #     try:
    #         # Localizar a tabela de métricas
    #         stats_table = self.wait.until(EC.presence_of_element_located((By.ID, "gsc_rsb_st")))
            
    #         # Extrair as células com as métricas
    #         metric_cells = stats_table.find_elements(By.CLASS_NAME, "gsc_rsb_std")
            
    #         # As células estão organizadas em ordem: 
    #         # [0-1]: Citações (All, Since 2019)
    #         # [2-3]: h-index (All, Since 2019)
    #         # [4-5]: i10-index (All, Since 2019)
    #         stats = {
    #             "total_citacoes": int(metric_cells[0].text),
    #             "citacoes_desde_2019": int(metric_cells[1].text),
    #             "indice_h": int(metric_cells[2].text),
    #             "indice_h_desde_2019": int(metric_cells[3].text),
    #             "indice_i10": int(metric_cells[4].text),
    #             "indice_i10_desde_2019": int(metric_cells[5].text)
    #         }
            
    #         return stats
            
    #     except Exception as e:
    #         print(f"  Erro ao extrair estatísticas do perfil: {e}")
    #         return {
    #             "total_citacoes": 0,
    #             "citacoes_desde_2019": 0,
    #             "indice_h": 0,
    #             "indice_h_desde_2019": 0,
    #             "indice_i10": 0,
    #             "indice_i10_desde_2019": 0
    #         }

    def close(self):
        self.driver.quit()

def main(researchers_list, affiliation_terms=None):
    scraper = GoogleScholarScraper(affiliation_terms)
    results = []

    for researcher in researchers_list:
        data = scraper.get_researcher_data(researcher)
        if data:
            results.append(data)
        time.sleep(2)

    scraper.close()
    
    # Cria um DataFrame e ordena por citações
    if results:
        df = pd.DataFrame(results)
        df_sorted = df.sort_values(by='citacoes', ascending=False)
        df_sorted.to_csv('pesquisadores_metricas.csv', index=False)
        return df_sorted
    
    return None

if __name__ == "__main__":
    researchers_list = [
        "Roberto Nicolete",
        "Jaime Ribeiro Filho"
    ]
    affiliation_terms = ["Fundação Oswaldo Cruz", "Fiocruz"]
    
    results = main(researchers_list, affiliation_terms)
    print(results)