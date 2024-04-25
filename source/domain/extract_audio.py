import pyaudio
import wave
import numpy as np

def extrair_trecho_mp3(arquivo_mp3, tempo_inicio, tempo_fim, extracted_name):
  """
  Extrai um trecho de um arquivo MP3 e salva em um novo arquivo WAV.

  Args:
    arquivo_mp3: Caminho para o arquivo MP3.
    tempo_inicio: Tempo de início do trecho em segundos.
    tempo_fim: Tempo de fim do trecho em segundos.

  Returns:
    Nenhum valor. Salva o trecho extraído em um novo arquivo WAV.

  Ex.: 
  https://www.youtube.com/watch?v=O2F91Up9fT8&ab_channel=StarJedi951
  (pathfilename, 16, 38, Theme_EmperorArrives)
  https://www.youtube.com/watch?v=xaBiygfkudk&ab_channel=Rezolak
  (pathfilename, 01, 02, ObiWan_HelloThere)
  (pathfilename, 15, 18, ObiWan_PrepareLighSaber)
  (pathfilename, 22, 25, ObiWan_RotateLighSaber)
  (pathfilename, 34, 35, ObiWan_YourMove)
  (pathfilename, 4:44, 4:46, ObiWan_SoUncivilized)
  https://www.youtube.com/watch?v=63EAJJakvEU&ab_channel=IliaTS
  (pathfilename, 47, 48, ObiWan_ThereYouAre)
  (pathfilename, 47, 48, ObiWan_ThereYouAre)
  (pathfilename, 57, 58, ObiWan_AreYouReady)
  (pathfilename, 1:01, 1:04, Anakin_AreYou)
  (pathfilename, 3:10, 3:18, Yoda_FearHateSuffering)
  """

  # Abrir arquivo MP3
  with wave.open(arquivo_mp3, 'rb') as wf:
    # Obter informações do arquivo
    formato = wf.get_format()
    num_canais = wf.getnchannels()
    taxa_amostragem = wf.getframerate()
    largura_bits = wf.getsampwidth()
    num_quadros = wf.getnframes()

    # Ler dados do arquivo
    dados_mp3 = wf.readframes(num_quadros)

    # Converter dados MP3 para dados WAV
    dados_wav = np.frombuffer(dados_mp3, dtype=np.int16)

    # Calcular índices dos quadros do trecho
    indice_inicio = int(tempo_inicio * taxa_amostragem)
    indice_fim = int(tempo_fim * taxa_amostragem)

    # Extrair trecho dos dados
    dados_trecho = dados_wav[indice_inicio:indice_fim]

    # Criar novo arquivo WAV para o trecho
    with wave.open(f'{extracted_name}.wav', 'wb') as wf_trecho:
      # Definir configurações do arquivo WAV
      wf_trecho.setnchannels(num_canais)
      wf_trecho.setframerate(taxa_amostragem)
      wf_trecho.setsampwidth(largura_bits)

      # Escrever dados do trecho no arquivo WAV
      wf_trecho.writeframes(dados_trecho)