#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

#include "sentencepiece_processor.h"
#include "sentencepiece_model.pb.h"
#include <vector>
#include <string>
#include "transformers/models/bert/modeling_bert.h"

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
    transformers::BertModel model
) {
    int sentence_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (sentence_id < num_sentences) {
        // Tokenizar a frase
        std::vector<int> input_ids = tokenize_sentence(sentences[sentence_id], sp);

        // Tratar sequências maiores que o máximo
        if (input_ids.size() > max_seq_length) {
            input_ids.resize(max_seq_length - 1);
            input_ids.push_back(model.config.sep_token_id);
        }
        else {
            // Preencher com zeros
            while (input_ids.size() < max_seq_length) {
                input_ids.push_back(model.config.pad_token_id);
            }
        }

        // Converter para tensor PyTorch na GPU
        torch::Tensor input_tensor = torch::from_blob(input_ids.data(), { 1, max_seq_length }, torch::kInt32).to(torch::kCUDA);
        torch::Tensor attention_mask = (input_tensor != model.config.pad_token_id).to(torch::kFloat32).to(torch::kCUDA);

        // Codificar a frase
        auto model_output = model.forward(input_tensor, attention_mask);
        auto embeddings = model_output.pooler_output;

        // Copiar o embedding para a memória de saída
        cudaMemcpy(output_embeddings + sentence_id * embedding_dim, embeddings.data_ptr<float>(), embedding_dim * sizeof(float), cudaMemcpyDeviceToDevice);
    }
}

// Função Python para chamar o kernel CUDA
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("encode_competences", [](
        std::vector<std::string> sentences,
        const std::string& model_name,
        int max_seq_length = 128
        ) {
            // Carregar o SentencePieceProcessor
            sentencepiece::SentencePieceProcessor sp;
            auto status = sp.Load(model_name + "/tokenizer.model");
            if (!status.ok()) {
                throw std::runtime_error("Erro ao carregar o SentencePiece model.");
            }

            // Carregar o modelo Transformer
            transformers::BertModel model(model_name);
            model.to(torch::kCUDA);

            // Converter as frases para char**
            const char** sentences_ptr = new const char* [sentences.size()];
            for (int i = 0; i < sentences.size(); i++) {
                sentences_ptr[i] = sentences[i].c_str();
            }

            // Alocar memória na GPU para os embeddings de saída
            int embedding_dim = model.config.hidden_size;
            torch::Tensor output_embeddings = torch::empty({ sentences.size(), embedding_dim }, torch::kFloat32).to(torch::kCUDA);

            // Configurar blocos e threads
            int threads_per_block = 256;  // Ajuste conforme necessário
            int blocks_per_grid = (sentences.size() + threads_per_block - 1) / threads_per_block;

            // Chamar o kernel CUDA
            encode_competences_kernel << <blocks_per_grid, threads_per_block >> > (
                sentences_ptr, sentences.size(), max_seq_length, output_embeddings.data_ptr<float>(), embedding_dim, sp, model
                );

            // Sincronizar para garantir que a execução do kernel termine
            cudaDeviceSynchronize();
            delete[] sentences_ptr; // Liberar a memória alocada

            return output_embeddings;
        });
}
