#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

#include "sentencepiece_processor.h"  // SentencePiece Processor
#include "sentencepiece_model.pb.h"  // SentencePiece Model
#include "transformer.h"             // Modelo Transformer
#include "utils.h"                   // Funções utilitárias

// Função para tokenizar uma frase com SentencePiece
std::vector<int> tokenize_sentence(const std::string& sentence, const sentencepiece::SentencePieceProcessor& sp) {
    std::vector<int> ids;
    sp.Encode(sentence, &ids);
    return ids;
}

// Kernel CUDA
__global__ void encode_competences_kernel(
    const char** sentences,
    int num_sentences,
    int max_seq_length,
    float* output_embeddings,
    int embedding_dim,
    const sentencepiece::SentencePieceProcessor& sp,
    Transformer& model
) {
    int sentence_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (sentence_id < num_sentences) {
        // Tokenizar a frase
        std::vector<int> input_ids = tokenize_sentence(sentences[sentence_id], sp);

        // Tratar sequências maiores que o máximo
        if (input_ids.size() > max_seq_length) {
            input_ids.resize(max_seq_length - 1);  // Remover o último token para adicionar [SEP]
            input_ids.push_back(model.config.sep_token_id);
        } else {
            // Preencher com zeros até o tamanho máximo da sequência
            while (input_ids.size() < max_seq_length) {
                input_ids.push_back(model.config.pad_token_id);
            }
        }

        // Converter para tensor PyTorch na GPU
        torch::Tensor input_tensor = torch::from_blob(input_ids.data(), {1, max_seq_length}, torch::kInt32).to(torch::kCUDA);
        torch::Tensor attention_mask = (input_tensor != model.config.pad_token_id).to(torch::kFloat32).to(torch::kCUDA);

        // Codificar a frase
        auto model_output = model.forward(input_tensor, attention_mask);
        auto embeddings = model_output.pooler_output; // Utilizar o pooler_output do transformer

        // Copiar o embedding para a memória de saída
        cudaMemcpy(output_embeddings + sentence_id * embedding_dim, embeddings.data_ptr<float>(), embedding_dim * sizeof(float), cudaMemcpyDeviceToDevice);
    }
}

// Função Python para chamar o kernel CUDA
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // Carregar o SentencePieceProcessor
    sentencepiece::SentencePieceProcessor sp;
    auto status = sp.Load(model_name + "/tokenizer.model");
    if (!status.ok()) {
        throw std::runtime_error("Erro ao carregar o SentencePiece model.");
    }

    // Carregar o modelo Transformer
    Transformer model(model_name);
    model.to(torch::kCUDA);

    m.def("encode_competences", &encode_competences, "Encode competences using a pre-trained multilingual model");
}