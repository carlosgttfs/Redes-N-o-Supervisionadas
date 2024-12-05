import os
import pandas as pd
from core.net_interface import Net
from core.net_standard_corrector import Standard
from core.net_potential_corrector import Potential
from core.net_distance_functions import distance_functions

# Carregar o dataset
dataset_path = './dataset/Mall_Customers.csv'
data = pd.read_csv(dataset_path)

# Pré-processamento do dataset
# Supondo que as colunas relevantes sejam 'Age', 'Annual Income (k$)', 'Spending Score (1-100)'
dataset = data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].values

# Definir classes fictícias para demonstração (ajuste conforme necessário)
classes = [j for j in range(5) for i in range(len(dataset) // 5)]

# Configurar a rede de Kohonen
potential = Potential(nu=1, tau=3000, p_min=0.75)
net = Net('Mall_Customers', distance_functions['manhattan'], potential)
net.initialize(input=3, m=10, n=15, factor=1, negative=False)

# Treinar a rede
net.train(5, dataset, stop_error=10 ** -15, stop_delta=10 ** -15)

# Definir o caminho para salvar os resultados
path = './data'
os.makedirs(path, exist_ok=True)

# Visualizar os resultados
try:
    net.visualize_maps(dataset, 'avg', path)
    net.visualize_u_matrix(path)
    net.visualize_clusterization(dataset, cluster_number=5, classes=classes, path=path)
except IndexError as e:
    print(f"An error occurred during visualization: {e}")

# Verificar o estado da rede
state = net.__dict__.get('_Net__state', None)
if state is not None:
    print(f"State type: {type(state)}")
    print(f"State content: {state}")
else:
    print("State attribute not found in Net object.")