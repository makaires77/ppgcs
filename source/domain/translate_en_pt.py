import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

class TranslatorEnPt:
    def __init__(self):
        # Nome do modelo
        model_name = "unicamp-dl/translation-en-pt-t5"

        # Carregar modelo e tokenizer
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)

        # Verificar e usar GPU se disponível, senão usar CPU
        # self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = "cpu"
        self.model.to(self.device)

    def translate(self, text):
        # O modelo T5 requer que você pré-fixe a string com "translate English to Portuguese: "
        input_text = "translate English to Portuguese: " + text
        inputs = self.tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)

        # Mover inputs para o mesmo dispositivo que o modelo
        inputs = inputs.to(self.device)

        # Gerar a tradução
        outputs = self.model.generate(inputs, max_length=512, num_beams=4, early_stopping=True)
        translated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Retornar texto traduzido
        return translated_text

# Código para criar uma instância da classe e traduzir um texto
# translator = TranslatorEnPt()
# english_text = "This is a test sentence in English."
# translated_text = translator.translate(english_text)
# print(translated_text)