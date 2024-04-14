import os
import json
import html
import logging
import requests
from pprint import pprint
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By

class FiocruzCearaScraper:
    def __init__(self, base_url, base_repo_dir):
        self.base_repo_dir = base_repo_dir
        self.folder_utils = os.path.join(base_repo_dir, 'utils')
        self.folder_assets = os.path.join(base_repo_dir, 'assets')        
        self.folder_domain = os.path.join(base_repo_dir, 'source', 'domain')
        self.folder_data_input = os.path.join(base_repo_dir, 'data', 'input')
        self.folder_data_output = os.path.join(base_repo_dir, 'data', 'output')            
        self.base_url = base_url
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
        }

    def get_html(self, url):
        response = requests.get(url, headers=self.headers)
        if response.status_code == 200:
            # print("Resposta obtida com sucesso:",response.status_code)
            return BeautifulSoup(response.content, 'html.parser')
        else:
            print("Resposta:",response.status_code)
            return None

    def scrape(self):
        driver = webdriver.Chrome()  # você precisa configurar o WebDriver para o navegador que você está usando
        driver.get(self.url)

        thematic_areas = []

        try:
            # Espera até que os elementos de interesse estejam disponíveis na página
            thematic_area_elements = WebDriverWait(driver, 10).until(
                EC.presence_of_all_elements_located((By.CLASS_NAME, "thematic-area"))
            )

            for thematic_area_element in thematic_area_elements:
                area_name = thematic_area_element.find_element(By.TAG_NAME, 'h5').text.strip()

                # Encontra o botão dentro do elemento de área temática e clica nele
                button = thematic_area_element.find_element(By.TAG_NAME, 'button')
                button.click()

                # Espera até que o conteúdo colapsável esteja visível
                collapse_content = WebDriverWait(driver, 10).until(
                    EC.visibility_of_element_located((By.CSS_SELECTOR, button.get_attribute('data-target')))
                )

                description = collapse_content.find_element(By.TAG_NAME, 'p').text.strip()
                link = collapse_content.find_element(By.TAG_NAME, 'a').get_attribute('href')

                thematic_areas.append({
                    'name': area_name,
                    'description': description,
                    'link': link
                })

                # Fecha o conteúdo colapsado
                button.click()
                
        finally:
            driver.quit()  # Garante que o navegador seja fechado mesmo em caso de exceção

        return json.dumps(thematic_areas, indent=4)

    def extract_details_link(self, base_link):
        response = requests.get(base_link)
        soup = BeautifulSoup(response.text, 'html.parser')
        if soup:
            details_link = soup.find('a', class_='btn')['href']
            return details_link

    def scrape_areas(self):
        page_content = self.get_html(self.base_url)
        if not page_content:
            return []

        areas = []

        content_section = page_content.find('section', class_='fiocruz-areas')
        conteiner_section = content_section.find('section', class_='container')
        if content_section:
            
            area_section = content_section.find('div', class_='container')
            if area_section:
                articles = area_section.find_all('article', class_='team-cards')
                for article in articles:
                    area_name = article.find('div', class_='team-name').get_text(strip=True) if article.find('div', class_='team-name') else ''
                    team_count = article.find('div', class_='dept-name').get_text(strip=True) if article.find('div', class_='dept-name') else ''
                    image_url = article.find('img')['src'] if article.find('img') else ''

                    area_url = article.find('a')['href'] if article.find('a') else ''
                    area_details = self.extract_research_area_details(area_url)

                    area_info = {
                        'area_name': area_name,
                        'team_count': team_count,
                        'image_url': image_url,
                        'details': area_details
                    }
                    areas.append(area_info)

        return areas

    def scrape_main_page_quantitative_data(self):
        page_content = self.get_html(self.base_url)
        if not page_content:
            return {}

        data = {}

        header_infos = page_content.find('div', id='header-infos')
        if header_infos:
            items = header_infos.find_all('div', class_='item')
            for item in items:
                category_name = item.find('div', class_='text').get_text(strip=True) if item.find('div', class_='text') else ''
                count = item.find('div', class_='count').get_text(strip=True) if item.find('div', class_='count') else ''
                data[category_name] = count

        return data

    def scrape_priority_scientific_areas(self):
        page_content = self.get_html(self.base_url)
        if not page_content:
            return []

        priority_areas = []

        content_section = page_content.find('div', id='content')
        if content_section:
            
            priority_section = content_section.find('div', class_='post-section priority')
            if priority_section:
                articles = priority_section.find_all('article', class_='team-cards')
                for article in articles:
                    area_name = article.find('div', class_='team-name').get_text(strip=True) if article.find('div', class_='team-name') else ''
                    team_count = article.find('div', class_='dept-name').get_text(strip=True) if article.find('div', class_='dept-name') else ''
                    image_url = article.find('img')['src'] if article.find('img') else ''

                    area_url = article.find('a')['href'] if article.find('a') else ''
                    area_details = self.extract_research_area_details(area_url)

                    area_info = {
                        'area_name': area_name,
                        'team_count': team_count,
                        'image_url': image_url,
                        'details': area_details
                    }
                    priority_areas.append(area_info)

        return priority_areas

    def extract_research_area_details(self, url):
        page_content = self.get_html(url)
        if not page_content:
            return {}

        details = {}

        # Extracting the main title of the research area
        title_div = page_content.find('div', id='header-infos')
        if title_div:
            details['title'] = title_div.find('h1').get_text(strip=True) if title_div.find('h1') else ''

        # Extracting research directors
        directors = []
        for member in page_content.find_all('div', class_='member'):
            name = member.find('div', class_='member-name').get_text(strip=True) if member.find('div', class_='member-name') else ''
            position = member.find('div', class_='member-position').get_text(strip=True) if member.find('div', class_='member-position') else ''
            role = member.find('div', class_='member-head').get_text(strip=True) if member.find('div', class_='member-head') else ''

            directors.append({'name': name,
                            'position': position,
                            'role': role
                            })
        
        details['directors'] = directors

        about_section = page_content.find('article', id='about')
        if about_section:
            # Extracting the 'About' section including specific h2 contents
            h2_elements = about_section.find_all('h2')
            for h2 in h2_elements:
                h2_title = h2.get_text(strip=True).lower()
                # Find the next sibling of h2 which contains the content
                h2_content = h2.find_next_sibling().get_text(strip=True) if h2.find_next_sibling() else ''
                details[h2_title] = h2_content

            # Extracting the 'About' section including specific h3 contents
            h3_elements = about_section.find_all('h3')
            for h3 in h3_elements:
                h3_title = h3.get_text(strip=True).lower()
                # Find the next sibling of h3 which contains the content
                h3_content = h3.find_next_sibling().get_text(strip=True) if h3.find_next_sibling() else ''
                details[h3_title] = h3_content

            # Extracting the 'About' section including specific h4 contents
            h4_elements = about_section.find_all('h4')
            for h4 in h4_elements:
                h4_title = h4.get_text(strip=True).lower()
                # Find the next sibling of h4 which contains the content
                h4_content = h4.find_next_sibling().get_text(strip=True) if h4.find_next_sibling() else ''
                details[h4_title] = h4_content

        return details


    def scrape_centers(self):
        centers_page = self.get_html(f"{self.base_url}/en/centers/")
        if not centers_page:
            return []
        
        centers = []
        
        for article in centers_page.find_all('article'):
            centers_info = {}
            title_section = article.find('h4', class_='center-all')
            if title_section:
                centers_info['title'] = title_section.get_text(strip=True)

            head_name_section = article.find('div', class_='head-name')
            if head_name_section:
                centers_info['head_name'] = head_name_section.get_text(strip=True)

            link = article.find('a', class_='team')
            if link and link.get('href'):
                centers_info['link'] = link.get('href')
                centers_info['aditional_info'] = self.scrape_additional_info(link.get('href'))

            home_title_section = article.find('h2')
            if home_title_section:
                platform_center = self.extract_titles_and_heads_by_section(centers_page)
                if platform_center:
                    centers_info['relations'] = platform_center

            if centers_info:
                centers.append(centers_info)

        return centers
    
    def extract_section_data(self, team_grid):
        articles = []
        for article in team_grid.find_all('article', class_='item filterable post-list mini'):
            article_info = {}
            title = article.find('h4')
            head_name = article.find('h5')
            if title:
                article_info['title'] = title.get_text(strip=True)
            if head_name:
                article_info['head_name'] = head_name.get_text(strip=True)
            articles.append(article_info)

        return articles

    def scrape_centers_data(self):
        centers_page = self.get_html(f"{self.base_url}/en/centers/")
        if not centers_page:
            return []
        
        centers = []

        # Iterating through each 'h2' element
        for h2 in centers_page.find_all('h2', class_='home-title'):
            center_title = h2.get_text(strip=True)
            team_grid = h2.find_next_sibling('div', class_='team-grid')

            if team_grid:
                articles = self.extract_centers_section_data(team_grid)
                center_info = {
                    'center_title': center_title,
                    'teams': articles
                }
                centers.append(center_info)

        return centers

    def extract_centers_section_data(self, team_grid):
        articles = []
        for article in team_grid.find_all('article', class_='item filterable post-list mini'):
            article_info = {
                'title': article.find('h4').get_text(strip=True) if article.find('h4') else '',
                'head_name': article.find('h5').get_text(strip=True) if article.find('h5') else ''
            }
            articles.append(article_info)

        return articles

    def scrape_department_data(self):
        page_content = self.get_html(self.base_url + "/en/departments/")
        # print(page_content.prettify())
        # print(len(page_content))
        if not page_content:
            return []

        departments = []

        for article in page_content.find_all('article', class_='team-cards'):
            department_name = article.h4.get_text(strip=True) if article.h4 else ''
            head_name = article.find('div', class_='head-name').get_text(strip=True) if article.find('div', class_='head-name') else ''
            team_count = article.find('div', class_='label').get_text(strip=True) if article.find('div', class_='label') else ''
            department_url = article.a['href'] if article.a else ''
            # print(department_url)
            department_info = {
                'department_name': department_name,
                'head_name': head_name,
                'team_count': team_count,
                'url': department_url
            }

            if department_url:
                department_info['aditional_info'] = self.scrape_departments_additional_info(department_url)

            departments.append(department_info)

        return departments

    def scrape_teams_data(self):
        teams_page = self.get_html(f"{self.base_url}/en/teams-heads/")
        if not teams_page:
            return []
        team_grid = teams_page.find('div', id='infinit', class_='team-grid')
        teams = []

        # Iterating through each 'h2' element
        for article in team_grid.find_all('article', class_='item'):
            if 'invisible' not in article['class']:  # Skip invisible items
                title = article.find('h4').get_text(strip=True) if article.find('h4') else ''
                head_name = article.find('h5').get_text(strip=True) if article.find('h5') else ''
                url = article.find('a')['href'] if article.find('a') else ''

                team_info = {
                    'title': title,
                    'head_name': head_name,
                    'url': url
                }
                teams.append(team_info)

        return teams

    def scrape_department_teams_data(self, page_content):
        teams_data = []
        for team_card in page_content.find_all('article', class_='team-cards'):
            team_name = team_card.h4.get_text(strip=True) if team_card.h4 else ''
            head_name = team_card.find('div', class_='head-name').get_text(strip=True) if team_card.find('div', class_='head-name') else ''
            team_url = team_card.a['href'] if team_card.a else ''
            teams_data.append({
                'team_name': team_name,
                'head_name': head_name,
                'url': team_url
            })
        return teams_data

    def scrape_departments_additional_info(self, url):
        page_content = self.get_html(url)
        if not page_content:
            return {}

        info = {}
        mid_content = page_content.find('div', id='mid-content')
        
        if mid_content:
            # Extract quantities
            for block in mid_content.find_all('div', class_='squ-block'):
                text = block.get_text(strip=True).replace("Scroll down","")
                if "Pub." in text:
                    info['qte_publication'] = text.replace("Pub.","").strip()
                elif "Team" in text:
                    info['qte_teams'] = text.replace("Team","").replace("s","").strip()
                elif "Keywords" in text:
                    info['qte_keywords'] = text.replace("Keywords","").strip()
                elif "Member" in text:
                    info['qte_members'] = text.replace("Member","").replace("s","").strip()   
                elif "Projects" in text:
                    info['qte_projects'] = text.replace("Project","").replace("s","").strip()
                elif "Tools" in text:
                    info['qte_tools'] = text.replace("Tool","").replace("s","").strip()                    
                elif "Software" in text:
                    info['qte_keywords'] = text.replace("Software","").replace("s","").replace("*","").strip()

            # Extração dos dados dos times
            teams_section = page_content.find('article', id='teams')
            if teams_section:
                info['teams'] = self.scrape_department_teams_data(teams_section)

            # Extract projects data
            projects_data = self.scrape_project_section(mid_content)
            if projects_data:
                info['projects'] = projects_data

            # Extract transversal projects data
            transversal_projects_data = self.scrape_transversal_project_section(mid_content)
            if transversal_projects_data:
                info['transversal_projects'] = transversal_projects_data

            # Extract section contents
            for section_id in ['about', 'teams', 'teams-secondary', 'nrcs', 'team', 'publications']:
                section = mid_content.find('article', id=section_id)
                if section:
                    title = section.find('h2')
                    content = section.find('article', class_='entry-content')
                    if title and content:
                        info[section_id] = {
                            'title': title.get_text(strip=True),
                            'content': content.get_text(strip=True)
                        }

        return info

    def scrape_platforms(self):
        page_content = self.get_html(self.base_url + "/en/platforms/")
        # print(page_content.prettify())
        # print(len(page_content))
        if not page_content:
            return []

        platforms = []
        # for article in page_content.find_all('article', class_='filterable team-cards'):
        for article in page_content.find_all('article'):
            platform_info = {}
            title_section = article.find('h4', class_='center-all')
            if title_section:
                platform_info['title'] = title_section.get_text(strip=True)

            head_name_section = article.find('div', class_='head-name')
            if head_name_section:
                platform_info['head_name'] = head_name_section.get_text(strip=True)

            link = article.find('a', class_='team')
            if link and link.get('href'):
                platform_info['link'] = link.get('href')
                platform_info['aditional_info'] = self.scrape_platforms_additional_info(link.get('href'))

            platforms.append(platform_info)

        return platforms

    def scrape_platforms_additional_info(self, url):
        page_content = self.get_html(url)
        if not page_content:
            return {}

        info = {}
        mid_content = page_content.find('div', id='mid-content')
        
        if mid_content:
            # Extract quantities
            for block in mid_content.find_all('div', class_='squ-block'):
                text = block.get_text(strip=True).replace("Scroll down","")
                if "Pub." in text:
                    info['qte_publication'] = text.replace("Pub.","").strip()
                elif "Keywords" in text:
                    info['qte_keywords'] = text.replace("Keywords","").strip()
                elif "Member" in text:
                    info['qte_members'] = text.replace("Member","").replace("s","").strip()   
                elif "Projects" in text:
                    info['qte_projects'] = text.replace("Project","").replace("s","").strip()
                elif "Tools" in text:
                    info['qte_tools'] = text.replace("Tool","").replace("s","").strip()                    
                elif "Software" in text:
                    info['qte_keywords'] = text.replace("Software","").replace("s","").replace("*","").strip()

            # Extract projects data
            projects_data = self.scrape_project_section(mid_content)
            if projects_data:
                info['projects'] = projects_data

            # Extract transversal projects data
            transversal_projects_data = self.scrape_transversal_project_section(mid_content)
            if transversal_projects_data:
                info['transversal_projects'] = transversal_projects_data

            # Extract section contents
            for section_id in ['about', 'members', 'software', 'fundings', 'partners', 'publications']:
                section = mid_content.find('article', id=section_id)
                if section:
                    title = section.find('h2')
                    content = section.find('div', class_='entry-content')
                    if title and content:
                        info[section_id] = {
                            'title': title.get_text(strip=True),
                            'content': content.get_text(strip=True)
                        }

        return info

    def scrape_transversal_project_section(self, page_content):
        projects_section = page_content.find('article', id='transversal-project')
        if not projects_section:
            return []

        transversal_projects = []
        for project_item in projects_section.find_all('div', class_='list-item'):
            project_data = self.extract_project_data(project_item.find('div', class_='rcontent'))
            if project_data:
                transversal_projects.append(project_data)

        return transversal_projects

    def extract_transversal_project_data(self, project_section):
        if not project_section:
            return None

        project_data = {}

        # Extração do título do projeto
        title = project_section.find('h3')
        if title:
            project_data['title'] = title.get_text(strip=True)

        # Extração do nome do líder do projeto
        head_name = project_section.find('div', class_='head-name')
        if head_name:
            project_data['head_name'] = head_name.get_text(strip=True)

        # Extração do status do projeto
        status = project_section.find('div', class_='status')
        if status:
            project_data['status'] = status.get_text(strip=True)

        # Extração do número de membros
        members = project_section.find('div', class_='members')
        if members:
            project_data['members'] = members.get_text(strip=True)

        # Extração da descrição
        description = project_section.find('div', class_='description')
        if description:
            project_data['description'] = description.get_text(strip=True)

        return project_data

    def scrape_project_section(self, page_content):
        projects_section = page_content.find('article', id='projects')
        if not projects_section:
            return []

        projects = []
        for project_item in projects_section.find_all('div', class_='list-item'):
            project_data = self.extract_project_data(project_item.find('div', class_='rcontent'))
            if project_data:
                projects.append(project_data)

        return projects

    def extract_project_data(self, project_section):
        if not project_section:
            return None

        project_data = {}

        # Extração do título do projeto
        title = project_section.find('h3')
        if title:
            project_data['title'] = title.get_text(strip=True)

        # Extração do nome do líder do projeto
        head_name = project_section.find('div', class_='head-name')
        if head_name:
            project_data['head_name'] = head_name.get_text(strip=True)

        # Extração do status do projeto
        status = project_section.find('div', class_='status')
        if status:
            project_data['status'] = status.get_text(strip=True)

        # Extração do número de membros
        members = project_section.find('div', class_='members')
        if members:
            project_data['members'] = members.get_text(strip=True)

        # Extração da descrição
        description = project_section.find('div', class_='description')
        if description:
            project_data['description'] = description.get_text(strip=True)

        return project_data

    def scrape_teams_section(self, page_content):
        teams_section = page_content.find('article', id='teams')
        if not teams_section:
            return []

        teams = []
        for teams_item in teams_section.find_all('article', class_='team-cards'):
            teams_data = self.extract_team_data(teams_item.find('div', class_='rcontent'))
            if teams_data:
                teams.append(teams_data)

        return teams

    def extract_team_data(self, project_section):
        if not project_section:
            return None

        teams_data = {}

        # Extração do título do team
        title = project_section.find('h4')
        if title:
            teams_data['title'] = title.get_text(strip=True)

        # Extração do nome do líder do projeto
        head_name = project_section.find('div', class_='head-name')
        if head_name:
            teams_data['head_name'] = head_name.get_text(strip=True)

        # Extração do label do team
        label = project_section.find('div', class_='label')
        if label:
            teams_data['label'] = label.get_text(strip=True)

        # # Extração do número de membros
        # members = project_section.find('div', class_='members')
        # if members:
        #     teams_data['members'] = members.get_text(strip=True)

        # # Extração da descrição
        # description = project_section.find('div', class_='description')
        # if description:
        #     teams_data['description'] = description.get_text(strip=True)

        return teams_data

    @staticmethod
    def save_to_json(data, filename):
        """
        Save the given data to a JSON file.

        :param data: List of dictionaries to be saved.
        :param filename: Name of the file where the data will be saved.
        """
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            print(f"Data successfully saved to {filename}")
        except Exception as e:
            print(f"Error saving data: {e}")

    def clean_pprint_output(self, output):
        # Remove unwanted characters introduced by pprint
        pprinted = pprint(output, width=200)
        try:
            if "('" in pprinted:
                output = pprinted.replace("('", "")
            if "'" in pprinted:    
                output = pprinted.replace("'", "")
            if "')" in pprinted:    
                output = pprinted.replace("')", "")
        except:
            pass
        # Handle any other pprint artifacts as needed
        return output

    def inserir_logotipo(self, imagem, alinhamento="center"):
        """
        Insere um logotipo em um html.

        Args:
            imagem: O caminho para o arquivo .png do logotipo.
            alinhamento: O alinhamento do logotipo no cabecalho de página. Os valores possíveis são "left", "center" e "right".

        Returns:
            O código html do logotipo.
        """

        if alinhamento not in ("left", "center", "right"):
            raise ValueError("O alinhamento deve ser 'left', 'center' ou 'right'.")

        return html.escape(f"""
            <img src="{imagem}" alt="Logotipo" align="{alinhamento}" width="300" height="200">
        """)

    def inserir_logotipos(self, logotipo_esquerdo=None, logotipo_centro=None, logotipo_direito=None):
        """
        Insere três logotipos em um html.

        Args:
            logotipo_esquerdo: O caminho para o arquivo .png do logotipo esquerdo.
            logotipo_centro: O caminho para o arquivo .png do logotipo central.
            logotipo_direito: O caminho para o arquivo .png do logotipo direito.

        Returns:
            O código html dos logotipos.
        """

        html = ""

        if logotipo_esquerdo is not None:
            html += self.inserir_logotipo(logotipo_esquerdo, "left")

        if logotipo_centro is not None:
            html += self.inserir_logotipo(logotipo_centro, "center")

        if logotipo_direito is not None:
            html += self.inserir_logotipo(logotipo_direito, "right")

        return html

    def generate_pasteur_report_html(self):
        # Use StringIO to capture print output
        from io import StringIO
        import sys

        separator = 80
        json_file = {}

        # Set up logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

        # Capture the output
        old_stdout = sys.stdout
        sys.stdout = report_output = StringIO()

        # Generate the report content
        logo_esq = os.path.join(self.folder_assets,'logo_fioce.png')
        logo_dir = os.path.join(self.folder_assets,'logo_pasteur.png')
        print(self.inserir_logotipos(logo_esq, None, logo_dir))                
        print("<h1><center><b>Coordenação de Pesquisa da Fiocruz Ceará</b></center></h1>")
        print("<h2><center><b>Estruturação em pesquisa do Instituto Pasteur</b></center></h2>")
        logging.info("Obtendo os dados do site do Instituto Pasteur, aguarde...")
        main_data = self.scrape_main_page_quantitative_data()
        print(f"<h2><center><b>{len(main_data)} seções de dados principais extraídas</b></center></h2>")
        logging.info(f"{len(main_data)} seções de dados principais extraídas")
        for i, j in main_data.items():
            print(f"<center>{j} {i}</center>")

        print(f"\n{'='*separator}\n")  

        logging.info("Obtendo áreas prioritárias de pesquisa, aguarde...")
        priority_research = self.scrape_priority_scientific_areas()
        json_file['priority_research'] = priority_research

        print(f"<h2><center><b>{len(priority_research)} áreas prioritárias em pesquisa</b></center></h2>")
        logging.info(f"{len(priority_research)} áreas prioritárias em pesquisa extraídas")
        for i in priority_research:
            print(f"<center>{i.get('team_count')} em <b>{i.get('area_name')}</b></center>")
        print()

        print(f"\n{'='*separator}\n")  

        for i in priority_research:
            titulo = i.get('details').get('title')
            diretores = i.get('details').get('directors')
            print(f"Área: <b>{titulo.upper()}</b>")
            for d in diretores:
                print(f"<center>{d.get('role')}: {d.get('name')} ({d.get('position')})</center>")    
            descricao = i.get('details').get('about')
            proposito = i.get('details').get('aims')
            indicador = i.get('details').get('measures')
            objetivos = i.get('details').get('achievements and future objectives')
            if descricao:
                if proposito:
                    descricao = descricao.replace(proposito,"").replace('AIMS', "").replace('Aims', "")
                if objetivos:
                    descricao = descricao.replace(objetivos,"").replace('ACHIEVEMENTS and FUTURE OBJECTIVES', '')
                if indicador:
                    descricao = descricao.replace(indicador,"").replace('MEASURES', "").replace('Measures', "")
                print()
                print("<b>Descrição da área:</b>")
                print(descricao)
            if proposito:
                print()
                print("<b>Propósito da área:</b>")
                print(proposito)
            if objetivos:
                print()
                print("<b>Objetivos da área:</b>")
                print(objetivos)
            if indicador:
                print()
                print("<b>Medidas da área:</b>")
                print(indicador)

            print(f"\n{'='*separator}\n")   

        logging.info("Obtendo os centros de referência, aguarde...")
        centers_data = self.scrape_centers_data()
        heads_centers_data = self.scrape_centers()
        json_file['centers_data'] = centers_data
        json_file['heads_centers_data'] = heads_centers_data

        print(f"<h2><center>{len(centers_data)} centros extraídos</h2></center>")
        logging.info(f"{len(centers_data)} centros extraídos")
        for i in heads_centers_data:
            print(f"<b><center>{i.get('title')}</b></center>\n<center>{i.get('head_name')}</center>")
        print(f"\n<h4>Associação dos Times com os Centros</h4>")
        for i in centers_data:
            print(f"\n<center><h2>{i.get('center_title')}</h2> ({len(i.get('teams')):02} times associados)</center>")
            for team in i.get('teams'):
                print(f"  {team.get('head_name'):>35}: <b>{team.get('title')}</b>")

        print(f"\n{'='*separator}\n")  

        logging.info("Obtendo os departamentos, aguarde...")
        departments_data = self.scrape_department_data()
        json_file['departments_data'] = departments_data

        print(f"<h2><center>{len(departments_data)} Departamentos extraídos</h2></center>")
        logging.info(f"{len(departments_data)} departamentos extraídos")
        for i in departments_data:
            print(f"  {i.get('head_name'):>25}: <b>{i.get('department_name')}</b>")
        print("<h4>Associação dos Times com os Departamentos</h4>")
        print('-'*150)
        for i in departments_data:
            print(f"\n<center><h2>{i.get('department_name')}</h2> ({len(i.get('aditional_info').get('teams')):02} times associados)</center>")
            for team in i.get('aditional_info').get('teams'):
                print(f"  {team.get('head_name'):>35}: <b>{team.get('team_name')}</b>")
            print('-'*150)

        print(f"\n{'='*separator}\n")

        logging.info("Obtendo as plataformas, aguarde...")
        platforms_data = self.scrape_platforms()
        json_file['platforms_data'] = platforms_data

        print(f"<h2><center>{len(platforms_data)} Plataformas extraídas</h2></center>")
        logging.info(f"{len(platforms_data)} plataformas extraídas")
        for i in platforms_data:
            print(f"  {i.get('head_name'):>25}: <b>{i.get('title')}</b>")

        print("<h4>Associação de Projetos com as Plaaformas</h4>")
        for i in platforms_data:
            platform_title = i.get('title')
            print('-'*150)
            print(f"<b>{platform_title.upper()}</b>")
            try:
                project_list = i.get('aditional_info').get('projects')
                for j in project_list:
                    title = j.get('title')
                    head = j.get('head_name')
                    status = j.get('status')
                    description = j.get('description')
                    print(f"\n  [{status}] {title} ({head})")
                    print(f"            {description}")
            except:
                try:
                    project_list = i.get('aditional_info').get('transversal_projects')
                    for j in project_list:
                        print(f"[Transversal Project] {j.get('title')}")
                except:
                    print(f"            Projetos não encontrados para esta plataforma")

        # Reset stdout so further print statements go to the console again
        sys.stdout = old_stdout

        # Get the report content as a string
        report_content = report_output.getvalue()

        # Convert the report content to HTML
        html_content = self.convert_to_html(report_content)

        # Save the HTML content to a file
        filename = 'report_pasteur_research.html'
        filepath = os.path.join(self.folder_data_output,filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)

        # Close the StringIO object
        report_output.close()
        logging.info("Relatório concluído!")
        logging.info(f"Salvo em: {filepath}")

        return json_file
    
    @staticmethod
    def convert_to_html(text):
        # Replace line breaks with <br>, and any other transformations needed
        html_content = text.replace('\n', '<br>')
        return f"<html><body>{html_content}</body></html>"