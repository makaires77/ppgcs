import requests
import bs4


def main():
    # Coletar os dados do site
    url = "https://research.pasteur.fr/en/platforms/"
    response = requests.get(url)
    html = response.content
    soup = bs4.BeautifulSoup(html, "html.parser")

    # Criar os dados para o gráfico
    market_data = []
    for market in soup.find_all("div", class_="market"):
        name = market.find("h2").text
        trends = []
        for trend in market.find_all("ul"):
            trends.append([trend.find("li").text, trend.find("i").text])
        market_data.append({"name": name, "trends": trends})

    product_data = []
    for product in soup.find_all("div", class_="product"):
        name = product.find("h2").text
        features = []
        for feature in product.find_all("ul"):
            features.append([feature.find("li").text, feature.find("i").text])
        product_data.append({"name": name, "features": features})

    technology_data = []
    for technology in soup.find_all("div", class_="technology"):
        name = technology.find("h2").text
        advances = []
        for advance in technology.find_all("ul"):
            advances.append([advance.find("li").text, advance.find("i").text])
        technology_data.append({"name": name, "advances": advances})

    # Criar o gráfico
    treemap = nivo.Treemap({
        "data": {
            "market": market_data,
            "product": product_data,
            "technology": technology_data,
        },
        "margin": {"top": 50, "bottom": 50},
        "padding": 10,
        "colors": {
            "market": "#1ABC9C",
            "product": "#2ECC71",
            "technology": "#3498DB",
        },
    })

    # Renderizar o gráfico
    treemap.render("treemap")

if __name__ == "__main__":
    main()
