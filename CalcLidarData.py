import math

# Definindo a classe LidarData para armazenar os dados do lidar
class LidarData:
    def __init__(self, FSA, LSA, CS, Speed, TimeStamp, Confidence_i, Angle_i, Distance_i):
        self.FSA = FSA
        self.LSA = LSA
        self.CS = CS
        self.Speed = Speed
        self.TimeStamp = TimeStamp
        self.Confidence_i = Confidence_i
        self.Angle_i = Angle_i
        self.Distance_i = Distance_i

# Função para calcular os dados do lidar a partir de uma string
def CalcLidarData(str):
    # Removendo espaços em branco da string
    str = str.replace(' ', '')

    # Convertendo dados hexadecimais para valores numéricos
    Speed = int(str[2:4] + str[0:2], 16) / 100
    FSA = float(int(str[6:8] + str[4:6], 16)) / 100
    LSA = float(int(str[-8:-6] + str[-10:-8], 16)) / 100
    TimeStamp = int(str[-4:-2] + str[-6:-4], 16)
    CS = int(str[-2:], 16)

    Confidence_i = list()
    Angle_i = list()
    Distance_i = list()

    # Calculando o passo angular com base nos dados FSA e LSA
    if LSA - FSA > 0:
        angleStep = float(LSA - FSA) / (12)
    else:
        angleStep = float((LSA + 360) - FSA) / (12)

    counter = 0
    circle = lambda deg: deg - 360 if deg >= 360 else deg

    # Iterando sobre os dados hexadecimais e convertendo para valores numéricos
    for i in range(0, 6 * 12, 6):
        Distance_i.append(int(str[8 + i + 2:8 + i + 4] + str[8 + i:8 + i + 2], 16) / 100)
        Confidence_i.append(int(str[8 + i + 4:8 + i + 6], 16))
        Angle_i.append(-(circle(angleStep * counter + FSA) * math.pi / 180.0))
        counter += 1
    
    # Criando uma instância da classe LidarData com os dados calculados
    lidarData = LidarData(FSA, LSA, CS, Speed, TimeStamp, Confidence_i, Angle_i, Distance_i)
    return lidarData
