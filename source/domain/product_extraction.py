class ProductExtraction:
    def __init__(self, products_file, model_name="scibert-scivocab-uncased"):
        self.products_file = products_file
        self.model = SentenceTransformer(model_name)

    def load_products(self):
        with open(self.products_file, "r") as f:
            return json.load(f)

    def extract_product_features(self, product_data):
        features = []
        # Extrair características dos produtos (agravos, áreas terapêuticas, etc.)
        # ...
        return features

    def vectorize_product_features(self, features):
        product_vectors = self.model.encode(features)
        return product_vectors