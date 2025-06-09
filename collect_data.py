import csv
import socket
import json
import psutil
import time
import os


def save_data_to_csv(angles, distances, coord_x, coord_y, filename='leituras-1v-porta-fechada-sem-objetos-p3.csv'):
    with open(filename, mode='a', newline='') as file:
        fieldnames = ['Ângulo', 'Distância', 'Coordenada X', 'Coordenada Y']
        writer = csv.writer(file)
        for angle, distance, cord_x, cord_y in zip(angles, distances, coord_x, coord_y):
            writer.writerow([angle, distance, cord_x, cord_y])


def log_metrics(metrics, filename='client_metrics-leituras-3v-porta-fechada-sem-objetos-p3.json'):
    if not os.path.exists(filename):
        with open(filename, 'w') as file:
            json.dump([], file)

    with open(filename, 'r+') as file:
        data = json.load(file)
        data.append(metrics)
        file.seek(0)
        json.dump(data, file, indent=4)


def receive_data(host, port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
        client_socket.connect((host, port))
        print("Conectado ao servidor em {}:{}".format(host, port))

        data = ""
        start_time = time.time()
        try:
            while True:
                chunk = client_socket.recv(1024).decode()
                if not chunk:
                    break
                data += chunk
                while '\n' in data:
                    json_str, data = data.split('\n', 1)
                    map_data = json.loads(json_str)
                    angles = map_data['Ângulo']
                    distances = map_data['Distância']
                    coord_x = map_data['Coordenada X']
                    coord_y = map_data['Coordenada Y']
                    save_data_to_csv(angles, distances, coord_x, coord_y)
        finally:
            end_time = time.time()
            total_time = end_time - start_time
            cpu_usage = psutil.cpu_percent()
            memory_info = psutil.virtual_memory()._asdict()
            metrics = {
                'Timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                'Total Processing Time (s)': total_time,
                'CPU Usage (%)': cpu_usage,
                'Memory Usage': memory_info
            }
            log_metrics(metrics)


if __name__ == "__main__":
    HOST = '192.168.3.1'
    PORT = 65432
    receive_data(HOST, PORT)
