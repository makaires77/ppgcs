from flask import Flask, render_template, jsonify, json, request
from flask import send_from_directory
from flask_cors import CORS
import os

app = Flask(__name__)
os.environ['FLASK_RUN_PORT'] = '8080'
app.static_folder = 'static'
CORS(app)

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(app.root_path, '/static/favicon.ico') 

@app.route('/')
def index():
    # Página inicial com links para os dois templates
    return render_template('index.html')

@app.route('/static/assets/images/<filename>')
def serve_image(filename):
    return send_from_directory('static/assets/images', filename)

@app.route('/templates/graph_revistas_capes.html')
def graph_revistas_capes():
    # Renderizar link para report no breadcrumb
    return render_template('graph_revistas_capes.html', show_render_button=True)

@app.route('/pasteur_fr_report')
def pasteur_fr_report():
    # Renderizar link para report no breadcrumb
    return render_template('report_pasteur_fr.html', show_render_button=True)

@app.route('/fiocruz_ce_report')
def fiocruz_ce_report():
    # Renderizar link para report no breadcrumb
    return render_template('report_fiocruz_ce.html', show_render_button=True)

@app.route('/i9c_gp_nobc')
def i9c_gp_nobc():
    # Renderizar template na região dinâmica sem trazer o breadcrumb
    return render_template('innomap_processes_no_breadcrumb.html', show_render_button=True)

@app.route('/i9c_mp_nobc')
def i9c_mp_nobc():
    # Renderizar template na região dinâmica sem trazer o breadcrumb
    return render_template('innomap_macroprocesses_no_breadcrumb.html', show_render_button=True)

@app.route('/i9c_gp01')
def i9c_gp01():
    return render_template('i9c_gp01.html', show_render_button=True)
    
@app.route('/i9c_gp02')
def i9c_gp02():
    return render_template('i9c_gp02.html', show_render_button=True)

@app.route('/i9c_gp03')
def i9c_gp03():
    return render_template('i9c_gp03.html', show_render_button=True)

@app.route('/i9c_gp')
def i9c_gp():
    # Verificar se a solicitação é AJAX para desabilitar breadcrumb
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        # Se for AJAX, renderiza o template sem o breadcrumb
        return render_template('innomap_processes_no_breadcrumb.html', show_render_button=True)
    else:
        # Para solicitações normais, inclui o breadcrumb
        return render_template('innomap_processes.html')

@app.route('/i9c_mp')
def i9c_mp():
    # Verificar se a solicitação é AJAX para desabilitar breadcrumb
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        return render_template('innomap_macroprocesses_no_breadcrumb.html', show_render_button=True)
    else:
        return render_template('innomap_macroprocesses.html')

@app.route('/data/json/roadmap.json')
def serve_roadmap_json():
    # Servir o arquivo JSON diretamente do diretório especificado
    return send_from_directory('static/data/json', 'roadmap.json')

@app.route('/api/graphdata', methods=['GET'])
def get_graph_data():
    # Carregar dados do JSON e retorna como resposta da API
    directory = os.path.join(app.root_path, 'static/data/json')
    with open(os.path.join(directory, 'roadmap.json'), 'r') as file:
        data = json.load(file)
    return jsonify(data)

@app.route('/api/update-graph', methods=['POST'])
def update_graph():
    try:
        # Carrega o JSON existente
        directory = os.path.join(app.root_path, 'static/data/json')
        json_path = os.path.join(directory, 'roadmap.json')
        with open(json_path, 'r+') as file:
            data = json.load(file)

            # Extrair dados do novo nó do corpo da solicitação
            new_node = request.json

            # Certificar extrair/validar campos 'size','color'
            # Definir valores padrão/validar entrada
            size = new_node.get('size', 10)  # valor padrão
            color = new_node.get('color', 'blue')  # vr padrão
            
            # Adicionar campos 'size' e 'color' ao novo nó
            new_node_data = {
                "id": new_node['id'],
                "label": new_node['label'],
                "title": new_node['title'],
                "row": new_node['row'],
                "size": size,
                "color": color
            }

            # Inserir o novo nó nos dados existentes
            data['nodes'].append(new_node_data)
            
            # Voltar início do arquivo para sobrescrevê-lo
            file.seek(0)
            # Atualiza o arquivo JSON
            json.dump(data, file, indent=4)
            # Trunca o arquivo para o novo tamanho
            file.truncate()
        
        return jsonify({"success": True, "message": "Graph updated successfully."}), 200
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

@app.route('/graph_hierarquico')
def graph_hierarquico():
    return render_template('graph_hierarquico.html')

if __name__ == '__main__':
    app.run(debug=True)
