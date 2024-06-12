// Definir margens
const margin = { top: 0, right: 0, bottom: 0, left: 0 };

// Variável para armazenar o tipo de layout selecionado
let layoutType = 'hierarchical'; // Valor padrão

// Variável para indicar se o input de arquivo está ativo
let fileInputActive = false;

// Variável para indicar se o grafo já foi carregado (inicialmente true)
let graphLoaded = true;

// Variável para armazenar o nome do arquivo JSON carregado
let currentJsonFileName = null;

// Variável para armazenar os dados do grafo
let graphData = null;

// Botão para renderizar grafo no HTML
document.addEventListener('DOMContentLoaded', function() {
    // Ouvinte de evento para o botão "Escolher Grafo"
    const renderBtn = document.getElementById('renderGraph');
    const jsonFileInput = document.getElementById('jsonFile'); 
    const newFileInput = document.createElement('input'); // Cria o input apenas uma vez
    newFileInput.type = 'file';
    newFileInput.accept = '.json';
    newFileInput.style.display = 'none';
    document.body.appendChild(newFileInput); // Adiciona o input ao DOM

    if (renderBtn) {
        renderBtn.addEventListener('click', () => {
            newFileInput.value = null;  // Limpa o valor do input antes de abrir a janela
            newFileInput.click();
        });
    }
    
    // Adiciona o ouvinte de evento ao newFileInput
    newFileInput.addEventListener('change', handleFileChange);

    // Adicionar ouvintes de evento para os botões de layout (se existirem)
    const hierarchicalLayoutBtn = document.getElementById('hierarchicalLayoutBtn');
    if (hierarchicalLayoutBtn) {
        hierarchicalLayoutBtn.addEventListener('click', () => {
            layoutType = 'hierarchical';
            if (graphData) {
                renderGraph(graphData, layoutType);
            }
        });
    }

    const groupedLayoutBtn = document.getElementById('groupedLayoutBtn');
    if (groupedLayoutBtn) {
        groupedLayoutBtn.addEventListener('click', () => {
            layoutType = 'grouped';
            if (graphData) {
                renderGraph(graphData, layoutType);
            }
        });
    }

    // Efeito de transição entre páginas
    const links = document.querySelectorAll('.ajax-link');
    links.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            const href = this.getAttribute('href');
            loadContent(href);
        });
    });
});

// Função para lidar com a mudança do arquivo (movida para fora do ouvinte do botão)
function handleFileChange(e) {
    const file = e.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = function(event) {
            graphData = JSON.parse(event.target.result);
            renderGraph(graphData);
            resizeAndRenderGraph();
        };
        reader.readAsText(file);
    }

    // Remove o input de arquivo após o carregamento
    document.body.removeChild(e.target);
}

// Função loadContent modificada para carregar os dados do JSON após o carregamento da página
function loadContent(href) {
    fetch(href)
        .then(response => response.text())
        .then(html => {
            const mainContent = document.getElementById('main-content');
            mainContent.innerHTML = html;
            mainContent.style.animation = 'none';
            mainContent.offsetHeight; // Força o navegador a reflow/repaint
            mainContent.style.animation = '';
            mainContent.style.animation = 'fadeIn 0.5s ease-out';

            // Observer para verificar quando o scatterplot é adicionado
            const observer = new MutationObserver(mutations => {
                mutations.forEach(mutation => {
                    if (mutation.addedNodes && mutation.addedNodes.length > 0) {
                        mutation.addedNodes.forEach(node => {
                            if (node.id === 'scatterplot') {
                                resizeAndRenderGraph();
                                observer.disconnect(); // Desconecta o observer após encontrar o scatterplot
                            }
                        });
                    }
                });
            });

            observer.observe(mainContent, { childList: true, subtree: true });
        })
        .catch(error => {
            console.error('Error loading the page: ', error);
        });
}

// Lidar com o redimensionamento e renderização do grafo
function resizeAndRenderGraph() {
    // Verifique se o container existe antes de prosseguir
    const container = document.getElementById('scatterplot');
    if (!container) {
        console.error("scatterplot element not found!");
        return;
    }

    const width = container.clientWidth - margin.left - margin.right;
    const height = container.clientHeight - margin.top - margin.bottom;

    renderGraph(graphData, layoutType, width, height); // Passa width e height para renderGraph
}

// Efeito de transição entre páginas
document.addEventListener('DOMContentLoaded', function() {
    const links = document.querySelectorAll('.ajax-link');
    links.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            const href = this.getAttribute('href');
            loadContent(href);
        });
    });

    // Inicializar o botão de renderização na interface HTML
    initializeRenderGraphButton();
});

function adjustLayout() {
    // Ajustar o layout ou reaplicar estilos
    const mainContent = document.getElementById('main-content');
    // Exemplo: ajustar a altura baseado no conteúdo
    mainContent.style.height = 'auto'; 
    // Outros ajuste conforme necessário aqui
}

function initializeRenderGraphButton() {
    const renderBtn = document.getElementById('renderGraph');
    if (renderBtn) {
        renderBtn.style.display = 'block';
        renderBtn.onclick = () => {
            if (!fileInputActive) {
                fileInputActive = true;

                // Cria um novo input de arquivo
                const newFileInput = document.createElement('input');
                newFileInput.type = 'file';
                newFileInput.accept = '.json';
                newFileInput.style.display = 'none';

                // Adiciona o ouvinte de evento ao novo input
                newFileInput.addEventListener('change', handleFileChange);

                document.body.appendChild(newFileInput);
                newFileInput.click();
                graphLoaded = false; // Reinicia graphLoaded aqui
            }
        };
    }
}

// Carregar dados e renderizar o grafo armazenado em JSON
async function loadGraphData(filePath) {
    const response = await fetch(filePath);
    graphData = await response.json(); // Atualiza a variável global

    // Agora chamar resizeAndRenderGraph aqui, após carregar os dados
    resizeAndRenderGraph();
}

function renderGraph(graphData, layoutType = 'hierarchical') {
    let svg = d3.select("#scatterplot").select("svg");
    const bounds = document.getElementById('scatterplot').getBoundingClientRect();
    const width = bounds.width;
    const height = bounds.height;
    if (svg.empty()) {
        svg = d3.select("#scatterplot")
                .append("svg")
                .attr("width", width)
                .attr("height", height);
    }

    svg.selectAll("*").remove(); // Limpa o SVG antes de desenhar

    const maxRow = d3.max(graphData.nodes, d => d.row);

    const xScale = d3.scaleBand()
                     .domain(graphData.nodes.map(d => d.id))
                     .range([15, width-15])
                     .paddingInner(1);

    const yScale = d3.scalePoint()
                     .domain(d3.range(1, maxRow + 1))
                     .range([0, height])
                     .padding(0.25);

    const nodePositions = {};
    graphData.nodes.forEach(node => {
        nodePositions[node.id] = {
            x: xScale(node.id) + xScale.bandwidth() / 2, // Centralizar nó na horizontal
            y: yScale(node.row) // Centralizar nó na vertical
        };
    });

    // Desenhar arestas como curvas de Bézier com cores
    svg.selectAll(".link")
       .data(graphData.links)
       .enter().append("path")
       .attr("d", d => {
           const source = nodePositions[d.source];
           const target = nodePositions[d.target];
           return `M${source.x},${source.y}C${(source.x + target.x) / 2},${source.y} ${(source.x + target.x) / 2},${target.y} ${target.x},${target.y}`;
       })
       .attr("fill", "none")
       .attr("stroke", d => d.color) // Usa a cor especificada no JSON
       .attr("stroke-width", 2);

    // Desenhar nós e definir a variável nodes para uso posterior com tooltips e labels
    const nodes = svg.selectAll(".node")
                     .data(graphData.nodes)
                     .enter().append("circle")
                     .attr("class", "node")
                     .attr("cx", d => nodePositions[d.id].x)
                     .attr("cy", d => nodePositions[d.id].y)
                     .attr("r", d => d.size) // Usa o tamanho especificado no JSON
                     .style("fill", d => d.color); // Usa a cor especificada no JSON

    // Criar elemento div para tooltip escondido por padrão
    const tooltip = d3.select("body").append("div")
        .attr("class", "tooltip")
        .style("opacity", 0)
        .style("position", "absolute")
        .style("text-align", "left")
        .style("width", "auto")
        .style("height", "auto")
        .style("padding", "8px")
        .style("font", "12px sans-serif")
        .style("background", "lightsteelblue")
        .style("border", "0px")
        .style("border-radius", "8px")
        .style("pointer-events", "none");

    // Adicionar eventos para o tooltip diretamente nos nós
    nodes.on("mouseover", function(event, d) {
             tooltip.transition()
                    .duration(200)
                    .style("opacity", .9);
             tooltip.html(`ID: ${d.id}<br/>Label: ${d.label}<br/>Title: ${d.title}<br/>Row: ${d.row}`)
                    .style("left", (event.pageX) + 20 + "px")
                    .style("top", (event.pageY) + "px");
         })
         .on("mouseout", function() {
             tooltip.transition()
                    .duration(500)
                    .style("opacity", 0);
         });

    // Adicionar rótulos (labels) diretamente aos nós
    svg.selectAll(".node-label")
       .data(graphData.nodes)
       .enter().append("text")
       .attr("x", d => nodePositions[d.id].x)
       .attr("y", d => nodePositions[d.id].y + 2    )
       .text(d => d.label)
       .attr("text-anchor", "middle")
       .attr("alignment-baseline", "central")
       .attr("font-size", "12px")
       .attr("fill", "black");
}

// Renderizar agrupando distribui nós em camadas conforme valor "row" do JSON
function renderGroupedGraph(graphData) {
    let svg = d3.select("#scatterplot").select("svg");
    const bounds = document.getElementById('scatterplot').getBoundingClientRect();
    const width = bounds.width;
    const height = bounds.height;
    if (svg.empty()) {
        svg = d3.select("#scatterplot")
                .append("svg")
                .attr("width", width)
                .attr("height", height);
    }

    svg.selectAll("*").remove(); // Limpa o SVG antes de desenhar

    // Função para mapear o valor de "row" para a nova camada conforme as regras específicas
    function mapRowToNewLayer(row) {
        if ([9, 8, 7].includes(row)) return 4; // Camada correspondente a .line-4 no HTML
        if ([6, 5, 4].includes(row)) return 3; // Camada correspondente a .line-3 no HTML
        if ([3, 2, 1].includes(row)) return 2; // Camada correspondente a .line-2 no HTML
        return row; // Mantém a original caso não se encaixe nas regras acima
    }

    // Atualiza os valores de 'row' dos nós conforme as regras de mapeamento
    graphData.nodes.forEach(node => {
        node.mappedRow = mapRowToNewLayer(node.row); // nova propriedade para evitar modificar o 'row' original
    });

    // Define uma escala para as camadas mapeadas
    const yScale = d3.scalePoint()
                     .domain([2, 3, 4]) // Camadas mapeadas
                     .range([height - (height * 0.2), height * 0.45, height * 0.1]) // Ajusta conforme a distribuição desejada
                     .padding(0.5);

    // Escala para posicionamento horizontal dos nós
    const xScale = d3.scaleBand()
                     .domain(graphData.nodes.map(d => d.id))
                     .range([0, width])
                     .padding(0.1);

    // Desenha os nós
    svg.selectAll(".node")
       .data(graphData.nodes)
       .enter().append("circle")
       .attr("class", "node")
       .attr("cx", d => xScale(d.id))
       .attr("cy", d => yScale(d.mappedRow))
       .attr("r", 10)
       .style("fill", "blue");

    // Desenha as arestas como curvas de Bézier
    const link = d3.linkHorizontal()
                   .x(d => d.x)
                   .y(d => d.y);

    svg.selectAll(".link")
       .data(graphData.links)
       .enter().append("path")
       .attr("class", "link")
       .attr("d", d => link({
           source: nodePositions[d.source],
           target: nodePositions[d.target]
       }))
       .attr("fill", "none")
       .attr("stroke", "black")
       .attr("stroke-width", 1);

    // Cria um elemento div para o tooltip que é escondido por padrão
    const tooltip = d3.select("body").append("div")
        .attr("class", "tooltip")
        .style("opacity", 0)
        .style("position", "absolute")
        .style("text-align", "left")
        .style("width", "auto")
        .style("height", "auto")
        .style("padding", "8px")
        .style("font", "12px sans-serif")
        .style("background", "lightsteelblue")
        .style("border", "0px")
        .style("border-radius", "8px")
        .style("pointer-events", "none");

    // Desenha os nós e adiciona eventos para o tooltip
    const nodes = svg.selectAll(".node")
                     .data(graphData.nodes)
                     .enter().append("circle")
                     .attr("class", "node")
                     .attr("cx", d => nodePositions[d.id].x)
                     .attr("cy", d => nodePositions[d.id].y)
                     .attr("r", 16)
                     .style("fill", "blue")
                     .on("mouseover", function(event, d) {
                         tooltip.transition()
                                .duration(100)
                                .style("opacity", .9);
                         tooltip.html(`ID: ${d.id}<br/>Label: ${d.label}<br/>Title: ${d.title}<br/>Row: ${d.row}`)
                                .style("left", (event.pageX) + "px")
                                .style("top", (event.pageY - 28) + "px");
                     })
                     .on("mouseout", function(d) {
                         tooltip.transition()
                                .duration(200)
                                .style("opacity", 0);
                     });

    // Adiciona rótulos (labels) abaixo e centralizados com cada nó
    svg.selectAll(".node-label")
       .data(graphData.nodes)
       .enter().append("text")
       .attr("class", "node-label")
       .attr("x", d => nodePositions[d.id].x)
       .attr("y", d => nodePositions[d.id].y +4) // Centralizar texto no nó
       .text(d => d.label)
       .attr("text-anchor", "middle") // Centraliza o texto horizontalmente
       .attr("alignment-baseline", "hanging") // Ajusta a linha de base do texto
       .attr("font-size", "10px")
       .attr("fill", "white");
}

// Configura o evento de redimensionamento usando D3
d3.select(window).on('resize', resizeAndRenderGraph);

// Adicionar novos nós
function showAddNodeForm(layerIndex) {
    // Armazena o índice da camada para uso posterior
    document.getElementById("addNodeForm").setAttribute("data-layer", layerIndex);
    document.getElementById("addNodeForm").style.display = "block";
}

async function updateJsonOnServer(newNode) {
    const response = await fetch('/path/to/server/endpoint', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(newNode),
    });

    if (response.ok) {
        console.log("Nó adicionado com sucesso");
        // Fechar o formulário e re-renderizar o gráfico, se necessário
        closeForm();
    } else {
        console.error("Falha ao adicionar nó");
    }
}

// ANTIGO e FUTURO
// Chama a função de carregamento de dados
// loadGraphData();

// TO-DO 1: 
// Outras estratégias de agrupamento e distribuição para renderizar
function renderHierarchicalCluster(graphData) {
    // Posicionamento determinado por layout cluster e escala yScale.
    const scatterplot = document.getElementById('scatterplot');
    const bounds = scatterplot.getBoundingClientRect();
    const width = bounds.width - margin.left - margin.right;
    const height = bounds.height - margin.top - margin.bottom;
    let svg = d3.select("#scatterplot").select("svg");
    if (svg.empty()) {
        svg = d3.select("#scatterplot")
                .append("svg")
                .attr("viewBox", `0 0 ${width + margin.left + margin.right} ${height + margin.top + margin.bottom}`)
                .append("g")
                .attr("transform", `translate(${margin.left},${margin.top})`);
    } else {
        svg.attr("viewBox", `0 0 ${width + margin.left + margin.right} ${height + margin.top + margin.bottom}`);
        svg.select("g").attr("transform", `translate(${margin.left},${margin.top})`);
    }

    // Limpa o SVG antes de desenhar
    svg.selectAll("g > *").remove();

    // Preparação dos dados para hierarquia
    const root = d3.stratify()
                   .id(d => d.id)
                   .parentId(d => {
                       const link = graphData.links.find(l => l.target === d.id);
                       return link ? link.source : null;
                   })(graphData.nodes);

    const cluster = d3.cluster().size([height, width]);
    cluster(root);

    // Escala de pontos posiciona verticalmente nós de acordo com suas camadas
    const yScale = d3.scalePoint()
                     .domain(graphData.nodes.map(d => d.label).sort(d3.ascending))
                     .range([0, height])
                     .padding(0.5);

    // Desenha as arestas
    svg.selectAll(".link")
        .data(root.links())
        .enter().append("path")
        .attr("class", "link")
        .attr("d", d3.linkHorizontal()
            .x(d => d.target.y)
            .y(d => yScale(d.target.data.label)))
        .attr("stroke", "#aaa");

    // Desenha os nós
    svg.selectAll(".node")
        .data(root.descendants())
        .enter().append("circle")
        .attr("class", "node")
        .attr("cx", d => d.y)
        .attr("cy", d => yScale(d.data.label))
        .attr("r", 15)
        .style('fill', '#333');
}

function renderCluster(graphData) {
    // Supõe-se que o #scatterplot seja um SVG já definido no HTML.
    const svg = d3.select("#scatterplot").select("svg"),
          width = +svg.attr("width"),
          height = +svg.attr("height");

    // Transforma a lista linear de nós e links em uma hierarquia.
    const root = d3.stratify()
                   .id(d => d.id)
                   .parentId(d => {
                       const link = graphData.links.find(l => l.target === d.id);
                       return link ? link.source : null;
                   })
                   (graphData.nodes);

    // Cria o layout de cluster.
    const cluster = d3.cluster()
                      .size([width, height - 160]); // Ajuste o tamanho cluster

    // Aplica o layout ao root da hierarquia.
    cluster(root);

    // Cria os links como caminhos SVG.
    const link = svg.selectAll(".link")
                    .data(root.descendants().slice(1))
                    .enter().append("path")
                    .attr("class", "link")
                    .attr("d", d => {
                        return "M" + d.y + "," + d.x
                             + "C" + (d.parent.y + 50) + "," + d.x
                             + " " + (d.parent.y + 50) + "," + d.parent.x
                             + " " + d.parent.y + "," + d.parent.x;
                    });

    // Cria os nós como círculos SVG.
    const node = svg.selectAll(".node")
                    .data(root.descendants())
                    .enter().append("g")
                    .attr("class", d => "node" + (d.children ? " node--internal" : " node--leaf"))
                    .attr("transform", d => "translate(" + d.y + "," + d.x + ")");

    node.append("circle")
        .attr("r", 4.5);

    node.append("text")
        .attr("dy", 3)
        .attr("x", d => d.children ? -8 : 8)
        .style("text-anchor", d => d.children ? "end" : "start")
        .text(d => d.id);
}

function renderForceLayout(graphData) {
    const svg = d3.select("#scatterplot").append("svg")
                  .attr("width", "100%")
                  .attr("height", "100%");

    const svgWidth = document.getElementById('scatterplot').clientWidth;
    const svgHeight = document.getElementById('scatterplot').clientHeight;

    // Define os limites para as colunas baseando-se na quantidade máxima de relações
    const maxRelations = Math.max(...graphData.map(node => node.relationships.length));
    const colWidth = svgWidth / maxRelations;
    const rowHeight = svgHeight / 9; // 9 camadas

    graphData.forEach(node => {
        // Encontra a posição Y baseada no label (camada)
        const y = (node.row - 1) * rowHeight + (rowHeight / 2);
        
        // Determina a posição X de forma dinâmica.
        // Simplificação assumindo distribuição igualitária dentro de cada camada.
        const relationshipsCount = node.relationships.length;
        const x = (relationshipsCount * colWidth) / 2;

        // TO-DO; lógica mais sofisticada, quando necessário, para distribuir horizontalmente com base na topologia.

        // Renderiza o nó como um círculo
        svg.append("circle")
           .attr("cx", x)
           .attr("cy", y)
           .attr("r", 10) // Raio do círculo
           .style("fill", "red"); // Cor do círculo
    });

    if (layoutType === 'force') {
        // Inicializa a simulação de força com os nós e as ligações (links)
        const simulation = d3.forceSimulation(graphData.nodes)
            .force('link', d3.forceLink(graphData.links).id(d => d.id))
            .force('charge', d3.forceManyBody())
            .force('center', d3.forceCenter(svgWidth / 2, svgHeight / 2));

        // Cria os elementos SVG para os nós e atualiza a posição em cada tick
        const node = svg.selectAll('circle')
            .data(graphData.nodes)
            .enter().append('circle')
            .attr('r', 10)
            .style('fill', 'red');

        simulation.on('tick', () => {
            node.attr('cx', d => d.x)
                .attr('cy', d => d.y);
        });
    }    
}

// TO-DO 2: 
// Refatorar renderizar sequencial o grafo, aplicável a todos estilos
async function renderGraphSequentially() {
    // Supondo que graphData já esteja carregado
    if (!graphData) {
        await loadGraphData(); // Carrega os dados se ainda não estiverem carregados
    }

    let svg = d3.select("#scatterplot").select("svg");
    const bounds = document.getElementById('scatterplot').getBoundingClientRect();
    const width = bounds.width;
    const height = bounds.height;

    if (svg.empty()) {
        svg = d3.select("#scatterplot")
                .append("svg")
                .attr("width", width)
                .attr("height", height);
    }

    svg.selectAll("*").remove(); // Limpa o SVG antes de desenhar

    // Ordena os nós e arestas por ID
    const nodes = graphData.nodes.sort((a, b) => a.id - b.id);
    const links = graphData.links.sort((a, b) => a.source - b.source);

    // Define escalas para o posicionamento
    const xScale = d3.scaleBand()
                     .domain(nodes.map(d => d.id))
                     .range([15, width - 15])
                     .paddingInner(1);
    const yScale = d3.scalePoint()
                     .domain([...new Set(nodes.map(d => d.row))]) // Único valor de 'row'
                     .range([15, height - 15])
                     .padding(0.5);

    // Função auxiliar para desenhar um nó
    const drawNode = (node) => {
        svg.append("circle")
           .attr("cx", xScale(node.id))
           .attr("cy", yScale(node.row))
           .attr("r", 5)
           .style("fill", "blue");
    };

    // Função auxiliar para desenhar uma aresta
    const drawLink = (link) => {
        const sourceNode = nodes.find(node => node.id === link.source);
        const targetNode = nodes.find(node => node.id === link.target);

        if (!sourceNode || !targetNode) return; // Ignora se não encontrar nós

        svg.append("line")
           .attr("x1", xScale(sourceNode.id))
           .attr("y1", yScale(sourceNode.row))
           .attr("x2", xScale(targetNode.id))
           .attr("y2", yScale(targetNode.row))
           .attr("stroke", "grey");
    };

    // Renderiza os nós e arestas sequencialmente
    nodes.forEach((node, index) => {
        setTimeout(() => drawNode(node), 100 * index);
    });

    setTimeout(() => { // Começa a desenhar as arestas após todos os nós
        links.forEach((link, index) => {
            setTimeout(() => drawLink(link), 100 * (index + nodes.length));
        });
    }, 100 * nodes.length);
}
// Chamada da função de renderização sequencial (está renderizando invertido)
// renderGraphSequentially();
