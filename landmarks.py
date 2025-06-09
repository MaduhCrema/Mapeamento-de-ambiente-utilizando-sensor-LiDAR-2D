import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KDTree
def analyze_landmark_persistence(landmarks1, landmarks2, matches, max_movement=1.0):
    """
    Analisa a persistência de landmarks entre duas leituras consecutivas

    Args:
        landmarks1, landmarks2: arrays de landmarks das duas leituras
        matches: lista de matches entre landmarks
        max_movement: movimento máximo esperado entre leituras

    Returns:
        dict com métricas de persistência
    """
    # Taxa de persistência geral
    persistence_rate = len(matches) / min(len(landmarks1), len(landmarks2))

    # Verificar quais landmarks mantiveram posições similares
    stable_matches = 0
    distances = []
    for match in matches:
        l1 = landmarks1[match[0]]
        l2 = landmarks2[match[1]]
        dist = np.linalg.norm(l1 - l2)
        distances.append(dist)
        if dist < max_movement:
            stable_matches += 1

    stability_rate = stable_matches / len(matches) if matches else 0

    return {
        'total_landmarks_1': len(landmarks1),
        'total_landmarks_2': len(landmarks2),
        'matches_count': len(matches),
        'persistence_rate': persistence_rate,
        'stability_rate': stability_rate,
        'average_movement': np.mean(distances) if distances else 0,
        'max_movement': np.max(distances) if distances else 0
    }
def evaluate_wall_continuity(landmarks, wall_segments, max_gap=1.0):
    """
    Avalia a continuidade das paredes detectadas usando múltiplos critérios
    
    Args:
        landmarks: array de landmarks
        wall_segments: lista de segmentos de parede (índices dos landmarks)
        max_gap: distância máxima esperada entre pontos consecutivos
        
    Returns:
        dict com métricas de continuidade para cada parede
    """
    wall_metrics = []
    
    for segment_idx, segment in enumerate(wall_segments):
        segment_landmarks = landmarks[segment]
        
        # 1. Análise de gaps entre pontos consecutivos
        distances = np.sqrt(np.sum(np.diff(segment_landmarks, axis=0)**2, axis=1))
        max_gap_found = np.max(distances)
        mean_gap = np.mean(distances)
        gap_std = np.std(distances)
        gap_uniformity = 1 - (gap_std / mean_gap if mean_gap > 0 else 0)
        
        # 2. Regressão linear para avaliar alinhamento
        x = segment_landmarks[:, 0]
        y = segment_landmarks[:, 1]
        
        # Calcula a linha de melhor ajuste
        coeffs = np.polyfit(x, y, 1)
        line_y = np.polyval(coeffs, x)
        
        # Calcula o R² para medir qualidade do ajuste linear
        residuals = y - line_y
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r_squared = 1 - (ss_res / ss_tot if ss_tot > 0 else 0)
        
        # 3. Análise de ângulos entre pontos consecutivos
        vectors = np.diff(segment_landmarks, axis=0)
        angles = np.arctan2(vectors[:, 1], vectors[:, 0])
        angle_diffs = np.abs(np.diff(angles))
        angle_consistency = 1 - (np.mean(angle_diffs) / np.pi)
        
        # Combina as métricas em um score de continuidade
        continuity_score = (
            0.4 * r_squared +              # Peso para linearidade
            0.3 * gap_uniformity +         # Peso para uniformidade dos gaps
            0.3 * angle_consistency        # Peso para consistência dos ângulos
        )
        
        wall_metrics.append({
            'segment_idx': segment_idx,
            'num_points': len(segment),
            'length': np.linalg.norm(segment_landmarks[-1] - segment_landmarks[0]),
            'max_gap': max_gap_found,
            'mean_gap': mean_gap,
            'gap_uniformity': gap_uniformity,
            'linearity': r_squared,
            'angle_consistency': angle_consistency,
            'continuity_score': continuity_score,
            'start_point': segment_landmarks[0],
            'end_point': segment_landmarks[-1]
        })
    
    # Calcula métricas globais
    if wall_metrics:
        global_metrics = {
            'mean_continuity_score': np.mean([m['continuity_score'] for m in wall_metrics]),
            'min_continuity_score': min(m['continuity_score'] for m in wall_metrics),
            'max_continuity_score': max(m['continuity_score'] for m in wall_metrics),
            'total_walls': len(wall_metrics),
            'total_wall_points': sum(m['num_points'] for m in wall_metrics),
            'total_wall_length': sum(m['length'] for m in wall_metrics)
        }
    else:
        global_metrics = {
            'mean_continuity_score': 0,
            'min_continuity_score': 0,
            'max_continuity_score': 0,
            'total_walls': 0,
            'total_wall_points': 0,
            'total_wall_length': 0
        }
        
    return {
        'wall_segments_metrics': wall_metrics,
        'global_metrics': global_metrics
    }

def detect_wall_landmarks(landmarks, points_x, points_y, max_gap, min_points):
    """
    Detecta quais landmarks fazem parte de paredes contínuas

    Args:
        landmarks: array de landmarks
        points_x, points_y: arrays de coordenadas dos pontos originais
        max_gap: distância máxima permitida entre landmarks de parede
        min_points: número mínimo de landmarks para formar uma parede

    Returns:
        dict com informações sobre landmarks de parede
    """
    if len(landmarks) < 2:
        return {'wall_landmarks': [], 'wall_count': 0}

    # Criar KD-Tree para landmarks
    tree = KDTree(landmarks)

    # Encontrar vizinhos próximos para cada landmark
    distances, indices = tree.query(landmarks, k=3)  # k=3 para pegar 2 vizinhos mais próximos

    # Identificar landmarks que podem ser parte de paredes
    wall_candidates = set()

    for i in range(len(landmarks)):
        # Verificar alinhamento com vizinhos
        neighbors = landmarks[indices[i][1:]]  # Excluir o próprio ponto
        if len(neighbors) >= 2:
            # Calcular ângulo entre pontos consecutivos
            v1 = neighbors[0] - landmarks[i]
            v2 = neighbors[1] - landmarks[i]
            angle = np.abs(np.arctan2(np.cross(v1, v2), np.dot(v1, v2)))

            # Se pontos estão aproximadamente alinhados (ângulo próximo de 0 ou pi)
            if angle < 0.1 or abs(angle - np.pi) < 0.1:
                wall_candidates.add(i)

    # Agrupar landmarks de parede em segmentos contínuos
    wall_segments = []
    current_segment = []

    for i in sorted(wall_candidates):
        if not current_segment:
            current_segment.append(i)
        else:
            prev_landmark = landmarks[current_segment[-1]]
            curr_landmark = landmarks[i]
            if np.linalg.norm(prev_landmark - curr_landmark) < max_gap:
                current_segment.append(i)
            else:
                if len(current_segment) >= min_points:
                    wall_segments.append(current_segment)
                current_segment = [i]

    if len(current_segment) >= min_points:
        wall_segments.append(current_segment)

    # Calcular métricas para cada segmento de parede
    wall_metrics = []
    for segment in wall_segments:
        segment_landmarks = landmarks[segment]
        length = np.linalg.norm(segment_landmarks[-1] - segment_landmarks[0])
        wall_metrics.append({
            'length': length,
            'num_landmarks': len(segment),
            'start_point': segment_landmarks[0],
            'end_point': segment_landmarks[-1]
        })

    return {
        'wall_segments': wall_segments,
        'wall_count': len(wall_segments),
        'wall_metrics': wall_metrics,
        'wall_landmarks_percentage': len(set().union(*wall_segments)) / len(landmarks) if wall_segments else 0
    }
def analyze_tolerance_range(points_x1, points_y1, points_x2, points_y2, sample_size=1000):
    """
    Analisa métricas para diferentes valores de tolerância de 0 a 1
    
    Args:
        points_x1, points_y1: primeira nuvem de pontos
        points_x2, points_y2: segunda nuvem de pontos
        sample_size: tamanho da amostra para análise
    """
    # Amostragem aleatória dos pontos
    indices1 = np.random.choice(len(points_x1), sample_size)
    indices2 = np.random.choice(len(points_x2), sample_size)
    
    sample1 = np.column_stack((points_x1[indices1], points_y1[indices1]))
    sample2 = np.column_stack((points_x2[indices2], points_y2[indices2]))
    
    # Calcular distâncias base
    tree = KDTree(sample2)
    distances, _ = tree.query(sample1)
    
    # Normalizar distâncias para o intervalo [0,1]
    max_dist = np.max(distances)
    normalized_distances = distances / max_dist
    
    # Testar diferentes tolerâncias
    tolerances = np.arange(0, 1.1, 0.1)
    results = []
    
    for tol in tolerances:
        # Filtrar distâncias pela tolerância atual
        valid_distances = normalized_distances[normalized_distances <= tol]
        
        if len(valid_distances) > 0:
            mean_dist = np.mean(valid_distances)
            std_dist = np.std(valid_distances)
            median_dist = np.median(valid_distances)
        else:
            mean_dist = std_dist = median_dist = 0
            
        results.append({
            'tolerance': tol,
            'mean': mean_dist,
            'std': std_dist,
            'median': median_dist,
            'points_included': len(valid_distances)
        })
    
    # Plotar resultados
    plt.figure(figsize=(15, 5))
    
    # Plot médias e desvios
    plt.subplot(131)
    plt.errorbar([r['tolerance'] for r in results], 
                 [r['mean'] for r in results],
                 yerr=[r['std'] for r in results],
                 fmt='o-')
    plt.xlabel('Tolerância')
    plt.ylabel('Média ± Desvio Padrão')
    plt.title('Média e Desvio Padrão')
    plt.grid(True)
    
    # Plot medianas
    plt.subplot(132)
    plt.plot([r['tolerance'] for r in results], 
             [r['median'] for r in results], 'o-')
    plt.xlabel('Tolerância')
    plt.ylabel('Mediana')
    plt.title('Mediana das Distâncias')
    plt.grid(True)
    
    # Plot número de pontos incluídos
    plt.subplot(133)
    plt.plot([r['tolerance'] for r in results], 
             [r['points_included'] for r in results], 'o-')
    plt.xlabel('Tolerância')
    plt.ylabel('Número de Pontos')
    plt.title('Pontos Incluídos')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Imprimir resultados
    print("\n=== Análise por Tolerância ===")
    for r in results:
        print(f"\nTolerância: {r['tolerance']:.1f}")
        print(f"Distância média: {r['mean']:.3f}")
        print(f"Desvio padrão: {r['std']:.3f}")
        print(f"Mediana: {r['median']:.3f}")
        print(f"Pontos incluídos: {r['points_included']}")
    
    return results
    

    
    return suggested_tolerance

def transform_points(points_x, points_y, dx=0, dy=0, theta=0, scale_factor=1.0):
    """
    Aplica transformação (translação e rotação) aos pontos com fator de escala

    Args:
        points_x: array de coordenadas x
        points_y: array de coordenadas y
        dx: deslocamento em x (metros no mundo real)
        dy: deslocamento em y (metros no mundo real)
        theta: ângulo de rotação em radianos
        scale_factor: fator para converter metros reais para unidades da nuvem de pontos
    """
    # Aplica o fator de escala aos deslocamentos
    dx_scaled = dx * scale_factor
    dy_scaled = dy * scale_factor

    # Aplica a transformação
    transformed_x = points_x * np.cos(theta) - points_y * np.sin(theta) + dx_scaled
    transformed_y = points_x * np.sin(theta) + points_y * np.cos(theta) + dy_scaled
    return transformed_x, transformed_y,dx_scaled,dy_scaled

def read_point_cloud(filename):
    """
    Lê arquivo CSV de nuvem de pontos onde cada linha contém: ângulo, distância, x, y

    Args:
        filename: caminho do arquivo
    Returns:
        angles: array de ângulos
        distances: array de distâncias
        points_x: array de coordenadas x
        points_y: array de coordenadas y
    """
    try:
        # Lê o arquivo CSV usando numpy com delimitador vírgula
        data = np.loadtxt(filename, delimiter=',')
        angles = data[:, 0]
        distances = data[:, 1]
        points_x = data[:, 2]
        points_y = data[:, 3]
        return angles, distances, points_x, points_y
    except Exception as e:
        print(f"Erro ao ler arquivo {filename}: {e}")
        return np.array([]), np.array([]), np.array([]), np.array([])

def save_point_cloud(filename, combined_x, combined_y):
    """
    Salva a nuvem de pontos no formato: ângulo, distância, x, y
    
    Args:
        filename: nome do arquivo de saída
        combined_x: array de coordenadas x
        combined_y: array de coordenadas y
    """
    try:
        # Calcula ângulos e distâncias a partir das coordenadas x,y
        angles = np.arctan2(combined_y, combined_x)
        distances = np.sqrt(combined_x**2 + combined_y**2)
        
        # Combina todos os dados
        combined_data = np.column_stack((angles, distances, combined_x, combined_y))
        
        # Salva no arquivo CSV
        np.savetxt(filename, combined_data, delimiter=',', fmt='%.6f')
        
        print(f"Nuvem de pontos salva em: {filename}")
        print(f"Total de pontos salvos: {len(combined_x)}")
        
    except Exception as e:
        print(f"Erro ao salvar arquivo {filename}: {e}")

def extract_landmarks_improved(points_x, points_y, min_samples=5):
    """
    Extrai landmarks usando DBSCAN com parâmetros adaptivos

    Args:
        points_x: array de coordenadas x
        points_y: array de coordenadas y
        min_samples: número mínimo de pontos para formar um cluster
    """
    if len(points_x) == 0 or len(points_y) == 0:
        return np.array([])

    points = np.column_stack((points_x, points_y))

    # Calcula eps adaptativo baseado na densidade dos pontos
    tree = KDTree(points)
    distances, _ = tree.query(points, k=2)  # k=2 para pegar o vizinho mais próximo
    eps = np.mean(distances[:, 1]) * 3  # Aumentado para 3x a média

    print(f"Debug - Epsilon calculado: {eps}")
    print(f"Debug - Número total de pontos: {len(points)}")

    # Aplica DBSCAN
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)

    # Debug - Análise dos clusters
    unique_labels = set(clustering.labels_)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    n_noise = list(clustering.labels_).count(-1)

    print(f"Debug - Número de clusters encontrados: {n_clusters}")
    print(f"Debug - Número de pontos classificados como ruído: {n_noise}")

    landmarks = []
    for label in unique_labels:
        if label != -1:  # Ignora ruído
            cluster_points = points[clustering.labels_ == label]
            # Usa mediana em vez de média para maior robustez
            center = np.median(cluster_points, axis=0)
            landmarks.append(center)

    return np.array(landmarks)

def find_landmark_matches(landmarks1, landmarks2, max_distance):
    """
    Encontra correspondências entre landmarks usando KDTree

    Args:
        landmarks1: array de landmarks da primeira leitura
        landmarks2: array de landmarks da segunda leitura
        max_distance: distância máxima para considerar um match
    """
    if len(landmarks1) == 0 or len(landmarks2) == 0:
        return []

    # Usa KDTree para matching eficiente
    tree = KDTree(landmarks2)
    distances, indices = tree.query(landmarks1, k=1)

    # Filtra matches pela distância máxima
    matches = [(i, int(idx)) for i, (d, idx) in enumerate(zip(distances, indices))
               if d < max_distance]

    return matches

def visualize_landmarks_and_matches(points_x1, points_y1, landmarks1,
                                  points_x2, points_y2, landmarks2,
                                  matches, sensor_pos1=(0,0), sensor_pos2=(0,0)):
    """
    Visualiza as leituras, landmarks e matches com limites consistentes.
    Landmarks1 são representadas como círculos e landmarks2 como triângulos
    para melhor distinção em impressões sem cores.
    """
    # Calcula os limites globais dos eixos baseado em todos os dados
    x_min = min(np.min(points_x1), np.min(points_x2), sensor_pos1[0], sensor_pos2[0])
    x_max = max(np.max(points_x1), np.max(points_x2), sensor_pos1[0], sensor_pos2[0])
    y_min = min(np.min(points_y1), np.min(points_y2), sensor_pos1[1], sensor_pos2[1])
    y_max = max(np.max(points_y1), np.max(points_y2), sensor_pos1[1], sensor_pos2[1])

    # Adiciona uma margem de 10% para melhor visualização
    margin_x = (x_max - x_min) * 0.1
    margin_y = (y_max - y_min) * 0.1
    
    # Ajusta os limites para serem iguais em todas as visualizações
    x_limits = [x_min - margin_x, x_max + margin_x]
    y_limits = [y_min - margin_y, y_max + margin_y]

    plt.figure(figsize=(15, 5))

    # Primeira leitura
    plt.subplot(131)
    plt.scatter(points_x1, points_y1, c='blue', s=1, label='Pontos Leitura 1')
    if len(landmarks1) > 0:
        # Usar círculos (o) para landmarks1
        plt.scatter(landmarks1[:,0], landmarks1[:,1], c='red', s=100, marker='o', edgecolors='black', label='Landmarks 1')
    plt.scatter(sensor_pos1[0], sensor_pos1[1], c='green', s=100, marker='o', label='Posição Sensor 1')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.xlim(x_limits)
    plt.ylim(y_limits)
    plt.title('Primeira Leitura')

    # Segunda leitura
    plt.subplot(132)
    plt.scatter(points_x2, points_y2, c='blue', s=1, label='Pontos Leitura 2')
    if len(landmarks2) > 0:
        # Usar triângulos (^) para landmarks2
        plt.scatter(landmarks2[:,0], landmarks2[:,1], c='red', s=100, marker='^', edgecolors='black', label='Landmarks 2')
    plt.scatter(sensor_pos2[0], sensor_pos2[1], c='green', s=100, marker='^', label='Posição Sensor 2')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.xlim(x_limits)
    plt.ylim(y_limits)
    plt.title('Segunda Leitura')

    # Visualização dos matches
    plt.subplot(133)
    plt.scatter(points_x1, points_y1, c='lightgray', s=1, label='Pontos Leitura 1')
    plt.scatter(points_x2, points_y2, c='lightgray', s=1, label='Pontos Leitura 2')

    # Plota landmarks e linhas conectando os matches
    for i, (idx1, idx2) in enumerate(matches):
        l1 = landmarks1[idx1]
        l2 = landmarks2[idx2]
        # Mantém os formatos diferentes para cada conjunto de landmarks
        plt.scatter(l1[0], l1[1], c=f'C{i}', s=100, marker='o', edgecolors='black')
        plt.scatter(l2[0], l2[1], c=f'C{i}', s=100, marker='^', edgecolors='black')
        plt.plot([l1[0], l2[0]], [l1[1], l2[1]], c=f'C{i}', linestyle='--', alpha=0.5)

    plt.scatter(sensor_pos1[0], sensor_pos1[1], c='blue', s=100, marker='o', label='Posição Sensor 1')
    plt.scatter(sensor_pos2[0], sensor_pos2[1], c='green', s=100, marker='^', label='Posição Sensor 2')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.xlim(x_limits)
    plt.ylim(y_limits)
    plt.title('Matches entre Landmarks')

    plt.tight_layout()
    plt.show()
def combine_point_clouds(points_x1, points_y1, points_x2, points_y2, landmarks1, landmarks2, matches, threshold=0.3):
    """
    Combina duas nuvens de pontos usando KDTree e média de pontos próximos

    Args:
        points_x1, points_y1: arrays de coordenadas da primeira nuvem
        points_x2, points_y2: arrays de coordenadas da segunda nuvem (já transformada)
        landmarks1, landmarks2: arrays de landmarks das duas nuvens
        matches: lista de tuplas (índice_landmark1, índice_landmark2)
        threshold: distância máxima para considerar pontos como próximos

    Returns:
        combined_x, combined_y: arrays de coordenadas da nuvem combinada
    """
    # Converter pontos para formato numpy adequado
    points1 = np.column_stack((points_x1, points_y1))
    points2 = np.column_stack((points_x2, points_y2))

    # Lista para armazenar pontos finais
    pontos_finais = []

    # Criar KD-Tree para ambas as nuvens
    tree1 = KDTree(points1)
    tree2 = KDTree(points2)

    # Para cada ponto da primeira nuvem, verificar se tem correspondentes na segunda
    indices_proximos_1 = tree2.query_radius(points1, r=threshold)

    # Processar pontos da primeira nuvem
    for i, indices in enumerate(indices_proximos_1):
        if len(indices) == 0:
            # Se não tem correspondentes na segunda nuvem, adicionar o ponto
            pontos_finais.append(points1[i])
        else:
            # Se tem correspondentes, calcular a média
            pontos_proximos = points2[indices]
            todos_pontos = np.vstack((points1[i], pontos_proximos))
            ponto_medio = np.median(todos_pontos, axis=0)  # Usando mediana ao invés de média
            pontos_finais.append(ponto_medio)

    # Para cada ponto da segunda nuvem, verificar se já foi processado
    indices_proximos_2 = tree1.query_radius(points2, r=threshold)
    for i, indices in enumerate(indices_proximos_2):
        if len(indices) == 0:
            # Se não tem correspondentes na primeira nuvem, adicionar o ponto
            pontos_finais.append(points2[i])

    # Converter lista final para array numpy
    combined_points = np.array(pontos_finais)

    # Separar em x e y novamente
    combined_x = combined_points[:, 0]
    combined_y = combined_points[:, 1]

    print(f"Número de pontos na primeira nuvem: {len(points1)}")
    print(f"Número de pontos na segunda nuvem: {len(points2)}")
    print(f"Número de pontos na nuvem combinada: {len(combined_points)}")

    return combined_x, combined_y

def visualize_combined_point_cloud(points_x1, points_y1, points_x2, points_y2,
                                 combined_x, combined_y, landmarks1, landmarks2,
                                 matches, sensor_pos1=(0,0), sensor_pos2=(0,0)):
    """
    Visualiza a nuvem de pontos combinada junto com as originais.
    - Pontos da primeira nuvem são representados como pontos (.)
    - Pontos da segunda nuvem são representados como cruzes (x)
    - Posição do sensor 1 como círculo e sensor 2 como triângulo
    """
    # Calcula os limites dos eixos
    x_min = min(np.min(points_x1), np.min(points_x2), np.min(combined_x))
    x_max = max(np.max(points_x1), np.max(points_x2), np.max(combined_x))
    y_min = min(np.min(points_y1), np.min(points_y2), np.min(combined_y))
    y_max = max(np.max(points_y1), np.max(points_y2), np.max(combined_y))

    margin_x = (x_max - x_min) * 0.1
    margin_y = (y_max - y_min) * 0.1

    plt.figure(figsize=(20, 5))

    # Primeira e segunda leituras com formatos diferentes para cada nuvem
    plt.subplot(131)
    # Usar marker='.' (ponto) para a primeira nuvem
    plt.scatter(points_x1, points_y1, c='blue', s=2, marker='.', label='Leitura 1')
    # Usar marker='x' (x) para a segunda nuvem
    plt.scatter(points_x2, points_y2, c='red', s=10, marker='x', label='Leitura 2')
    plt.scatter(sensor_pos1[0], sensor_pos1[1], c='green', s=100, marker='o', label='Sensor 1')
    plt.scatter(sensor_pos2[0], sensor_pos2[1], c='yellow', s=100, marker='^', label='Sensor 2')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.xlim(x_min - margin_x, x_max + margin_x)
    plt.ylim(y_min - margin_y, y_max + margin_y)
    plt.title('Leituras Originais')

    # Nuvem combinada (sem diferenciar por formato, apenas por cor)
    plt.subplot(132)
    plt.scatter(combined_x, combined_y, c='purple', s=2, marker='.', label='Nuvem Combinada')
    plt.scatter(sensor_pos1[0], sensor_pos1[1], c='green', s=100, marker='o', label='Sensor 1')
    plt.scatter(sensor_pos2[0], sensor_pos2[1], c='yellow', s=100, marker='^', label='Sensor 2')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.xlim(x_min - margin_x, x_max + margin_x)
    plt.ylim(y_min - margin_y, y_max + margin_y)
    plt.title('Nuvem Combinada')

    # Comparação com formatos diferentes para cada nuvem
    plt.subplot(133)
    # Usar marker='.' (ponto) para a primeira nuvem com opacidade
    plt.scatter(points_x1, points_y1, c='blue', s=2, marker='.', alpha=0.3, label='Leitura 1')
    # Usar marker='x' (x) para a segunda nuvem com opacidade
    plt.scatter(points_x2, points_y2, c='red', s=10, marker='x', alpha=0.3, label='Leitura 2')
    # Manter o formato de ponto para a nuvem combinada
    plt.scatter(combined_x, combined_y, c='purple', s=2, marker='.', label='Combinada')
    plt.scatter(sensor_pos1[0], sensor_pos1[1], c='green', s=100, marker='o', label='Sensor 1')
    plt.scatter(sensor_pos2[0], sensor_pos2[1], c='yellow', s=100, marker='^', label='Sensor 2')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.xlim(x_min - margin_x, x_max + margin_x)
    plt.ylim(y_min - margin_y, y_max + margin_y)
    plt.title('Comparação')

    plt.tight_layout()
    plt.show()
# Exemplo de uso com arquivos
if __name__ == "__main__":
    print("\n=== Processando primeira leitura ===")
    # Arquivo CSV da primeira leitura
    angles1, distances1, points_x1, points_y1 = read_point_cloud(r'C:\Users\eduar\Desktop\Server - TCC\code\Dados\porta fechada\02\porta-fechada-inicio-intermediaria-landmarks-corrigida.csv')
    print("\nExtraindo landmarks da primeira leitura:")
    landmarks1 = extract_landmarks_improved(points_x1, points_y1)
    print(f"Número de landmarks encontrados na primeira leitura: {len(landmarks1)}")

    print("\n=== Processando segunda leitura ===")
    # Substitua 'leitura2.txt' pelo nome do seu segundo arquivo
    angles2, distances2, points_x2, points_y2 = read_point_cloud(r'C:\Users\eduar\Desktop\Server - TCC\code\Dados\porta fechada\02\porta-fechada-p3.csv')

    # Aplica transformação aos pontos da segunda leitura para compensar o movimento do sensor
    real_dx = 3.05  # deslocamento real de 1 metro em x - 2.8 - 1.55 -    3.05
    real_dy = 0.8   # sem movimento em y - 0.2 - 0.1   - 0.8
    theta = 0.07     # sem rotação - 0.07 -  -0.03
    scale_factor = 1

    points_x2, points_y2, dx, dy = transform_points(points_x2, points_y2,
                                          dx=real_dx,
                                          dy=real_dy,
                                          theta=theta,
                                          scale_factor=scale_factor)
    print("\nExtraindo landmarks da segunda leitura:")
    landmarks2 = extract_landmarks_improved(points_x2, points_y2)
    print(f"Número de landmarks encontrados na segunda leitura: {len(landmarks2)}")

    # Encontra e valida matches entre landmarks
    matches = find_landmark_matches(landmarks1, landmarks2, max_distance=0.25)  # Aumentar max_distance
    print(f"\nNúmero de matches encontrados: {len(matches)}")

    # Visualiza resultados dos matches
    #visualize_landmarks_and_matches(
     # points_x1, points_y1, landmarks1,
     # points_x2, points_y2, landmarks2,
     # matches,
     # sensor_pos1=(0,0),
     # sensor_pos2=(dx,dy)
    #)

    # Combinar as nuvens de pontos
    combined_x, combined_y = combine_point_clouds(
      points_x1, points_y1,
      points_x2, points_y2,
      landmarks1, landmarks2,
      matches,
      threshold=0.05  # Ajuste este valor conforme necessário
    )

    print(f"\nNúmero de pontos na nuvem combinada: {len(combined_x)}")

    # Visualizar resultado da combinação
    visualize_combined_point_cloud(
        points_x1, points_y1,
        points_x2, points_y2,
        combined_x, combined_y,
        landmarks1, landmarks2,
        matches,
        sensor_pos1=(0,0),
        sensor_pos2=(dx,dy)
    )

    # Após combinar as nuvens de pontos...
    save_point_cloud(r'C:\Users\eduar\Desktop\Server - TCC\code\Dados\porta fechada\02\porta-fechada-inicio-intermediaria-landmarks-corrigida-final.csv', combined_x, combined_y)
    
    # Após processar as leituras e encontrar matches
    persistence_metrics = analyze_landmark_persistence(landmarks1, landmarks2, matches)
    print("\n=== Métricas de Persistência ===")
    print(f"Taxa de persistência: {persistence_metrics['persistence_rate']:.2f}")
    print(f"Taxa de estabilidade: {persistence_metrics['stability_rate']:.2f}")
    print(f"Movimento médio: {persistence_metrics['average_movement']:.2f}")

    # Após detectar as paredes
    wall_metrics1 = detect_wall_landmarks(landmarks1, points_x1, points_y1, max_gap=1.0, min_points=3)
    continuity_metrics1 = evaluate_wall_continuity(landmarks1, wall_metrics1['wall_segments'])

    print("\n=== Métricas de Continuidade das Paredes ===")
    print(f"Score médio de continuidade: {continuity_metrics1['global_metrics']['mean_continuity_score']:.3f}")
    print(f"Comprimento total de paredes: {continuity_metrics1['global_metrics']['total_wall_length']:.2f}")
