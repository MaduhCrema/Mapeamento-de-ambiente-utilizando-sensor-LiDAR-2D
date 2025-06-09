# Mapeamento de um ambiente interno utilizando sensor LiDAR 2D
## Modo de uso
### 1.Rodar o arquivo read_data.py no servidor, que deve estar conectado ao sensor
### 2.Para coletar os dados no cliente, rodar o arquivo collect_data.py, a onde irá salvar o a nuvem de dados .csv
### 3.Após a coleta dos mapas do ambiente interno e de salvar as distâncias/deslocamentos em X e Y e rotação dos pontos do sensor para a coleta dos dados do ambiente de cada leitura:
#### 4.Executar o arquivo landmarks.py, utilizando os delocamentos que deve ser informado sobre a segunda nuvem de dados, e combinar os dois arquivos. Em seguinda caso tenha uma terceira nuvem, combinar esse arquivo com a leitura que acabamos de juntar das duas primeiras nuvens, e realizar esse passo de acordo com a quantidade de leituras realizadas (não esquecer de informar o deslocamento do segundo arquivo)
![alt text](https://github.com/MaduhCrema/Mapeamento-um-ambiente-utilizando-sensor-LiDAR-2D/blob/master/WhatsApp%20Image%202025-06-09%20at%2010.16.16.jpeg)
[![Clique para baixar e ver o vídeo - Funcionamento do sensor](https://github.com/MaduhCrema/Mapeamento-um-ambiente-utilizando-sensor-LiDAR-2D/blob/master/video%20sensor%20funfando.mp4)
