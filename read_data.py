import serial
from CalcLidarData import CalcLidarData
import math
import socket
import json

def capture_lidar_data(conn):
    ser = serial.Serial(port='/dev/ttyUSB0',
                        baudrate=230400,
                        timeout=5.0,
                        bytesize=8,
                        parity='N',
                        stopbits=1)

    tmpString = ""
    angles = list()
    distances = list()
    read_count = 0

    map_data = {'Ângulo': [], 'Distância': [], 'Coordenada X': [], 'Coordenada Y': []}

    while read_count < 300:  # Parar após 100 leituras
        loopFlag = True
        flag2c = False

        while loopFlag:
            b = ser.read()
            tmpInt = int.from_bytes(b, 'big')

            if tmpInt == 0x54:
                tmpString += b.hex() + " "
                flag2c = True
                continue
            elif tmpInt == 0x2c and flag2c:
                tmpString += b.hex()

                if not len(tmpString[0:-5].replace(' ', '')) == 90:
                    tmpString = ""
                    loopFlag = False
                    flag2c = False
                    continue

                lidarData = CalcLidarData(tmpString[0:-5])
                angles.extend(lidarData.Angle_i)
                distances.extend(lidarData.Distance_i)

                for j in range(len(angles)):
                    x = distances[j] * math.cos(angles[j])
                    y = distances[j] * math.sin(angles[j])

                    map_data['Ângulo'].append(angles[j])
                    map_data['Distância'].append(distances[j])
                    map_data['Coordenada X'].append(x)
                    map_data['Coordenada Y'].append(y)

                tmpString = ""
                loopFlag = False
                read_count += 1
            else:
                tmpString += b.hex() + " "
            flag2c = False

        if read_count % 10 == 0:  # Enviar dados a cada 10 leituras
            conn.send((json.dumps(map_data) + '\n').encode())
            map_data = {'Ângulo': [], 'Distância': [], 'Coordenada X': [], 'Coordenada Y': []}

    conn.close()  # Fechar a conexão após 100 leituras

def start_server(host, port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.bind((host, port))
        server_socket.listen(1)
        print("Servidor escutando em {}:{}".format(host, port))

        conn, addr = server_socket.accept()
        print("Conexão estabelecida com:", addr)

        capture_lidar_data(conn)

if __name__ == "__main__":
    HOST = '127.0.0.1'
    PORT = 65432
    start_server(HOST, PORT)