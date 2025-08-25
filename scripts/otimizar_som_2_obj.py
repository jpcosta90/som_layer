import os
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
import optuna
import gc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# --- 1. CLASSES E FUNÇÕES AUXILIARES ---
# (As mesmas classes e funções que usamos nos scripts de otimização)

class TokenDataset(Dataset):
    def __init__(self, file_paths, embeddings_dir): self.file_paths = file_paths; self.embeddings_dir = embeddings_dir
    def __len__(self): return len(self.file_paths)
    def __getitem__(self, idx):
        full_path = os.path.join(self.embeddings_dir, self.file_paths[idx]); return torch.load(full_path, weights_only=True)

def collate_tokens(batch): return torch.cat(batch, dim=0)

class SOM_PyTorch:
    # (Cole a classe SOM_PyTorch completa aqui)
    def __init__(self, x, y, input_dim, sigma=1.0, learning_rate=0.5, device='cuda'):
        self.x, self.y, self.device, self.sigma, self.learning_rate = x, y, device, sigma, learning_rate
        tensor_dtype = torch.bfloat16
        self.weights = torch.randn(x * y, input_dim, device=device, dtype=tensor_dtype)
        self.weights /= torch.norm(self.weights, p=2, dim=1, keepdim=True)
        self.locations = torch.cartesian_prod(torch.arange(x, device=device), torch.arange(y, device=device)).to(dtype=tensor_dtype)
    @torch.no_grad()
    def train_batch_from_loader(self, dataloader, val_dataloader, num_epochs=10):
        history = {'qe': [], 'te': []}
        initial_radius, initial_lr, total_iterations = self.sigma, self.learning_rate, num_epochs * len(dataloader)
        print(f"Treinando modelo final por {num_epochs} épocas com {len(dataloader)} lotes por época.")
        for epoch in range(num_epochs):
            self.train()
            for i, batch in enumerate(tqdm(dataloader, desc=f"Época {epoch+1}/{num_epochs} [Treino]", leave=False)):
                batch = batch.to(self.device); current_iteration = epoch * len(dataloader) + i
                dists = torch.cdist(batch, self.weights); winner_indices = torch.argmin(dists, dim=1); winner_locations = self.locations[winner_indices]
                time_constant = total_iterations / np.log(initial_radius if initial_radius > 1 else 1.0001); sigma = initial_radius * torch.exp(torch.tensor(-current_iteration / time_constant)); lr = initial_lr * torch.exp(torch.tensor(-current_iteration / total_iterations))
                dist_from_winner = torch.cdist(self.locations, winner_locations); influence = torch.exp(-dist_from_winner**2 / (2 * sigma**2)) * lr
                numerator = torch.matmul(influence, batch); denominator = influence.sum(axis=1, keepdim=True)
                new_weights = numerator / (denominator + 1e-9); self.weights.copy_(new_weights)
            
            self.eval()
            print(f"\n--- Fim da Época {epoch+1}/{num_epochs}: Avaliando... ---")
            qe = self.quantization_error(val_dataloader)
            te = self.topographic_error(val_dataloader)
            history['qe'].append(qe); history['te'].append(te)
            print(f"Resultados da Época {epoch+1}: QE = {qe:.4f}, TE = {te:.4f}")
        return history
    @torch.no_grad()
    def quantization_error(self, dataloader):
        total_error, total_samples = 0., 0.
        for batch in dataloader:
            batch = batch.to(self.device); dists = torch.cdist(batch, self.weights); winner_dists = torch.min(dists, dim=1)[0]
            total_error += torch.sum(winner_dists).item(); total_samples += len(batch)
        return total_error / total_samples
    @torch.no_grad()
    def topographic_error(self, dataloader):
        if self.x < 2 or self.y < 2: return 0.0
        total_error, total_samples = 0., 0.
        for batch in dataloader:
            batch = batch.to(self.device); dists = torch.cdist(batch, self.weights); _, best_two = torch.topk(dists, 2, dim=1, largest=False)
            bmu1_coords = self.locations[best_two[:, 0]]; bmu2_coords = self.locations[best_two[:, 1]]
            is_error = torch.sum(torch.abs(bmu1_coords - bmu2_coords), dim=1) > 1; total_error += torch.sum(is_error).item(); total_samples += len(batch)
        return total_error / total_samples
    def train(self): pass
    def eval(self): pass

# --- 2. CONFIGURAÇÃO ---
EMBEDDINGS_DIR = '/mnt/data/embeddings'
MANIFEST_CSV = '../data/sampled_dataset_png/manifest_linux.csv'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
STUDY_NAME = 'som_qe_te_v1' # Nome do estudo que você quer carregar
STORAGE_DB = 'sqlite:///../data/som_optimization.db'
FINAL_MODEL_PATH = '../models/final_som_model_2_obj.pth'
NUM_EPOCHS_FINAL = 20

# --- 3. EXECUÇÃO PRINCIPAL ---
if __name__ == '__main__':
    # --- Passo 1: Carregar o estudo e filtrar os trials ---
    print(f"Carregando estudo '{STUDY_NAME}' do banco de dados...")
    study = optuna.load_study(study_name=STUDY_NAME, storage=STORAGE_DB)
    df_all_trials = study.trials_dataframe()
    
    # --- MUDANÇA: dropna agora verifica apenas 'values_0' e 'values_1' ---
    df_successful = df_all_trials.dropna(subset=['values_0', 'values_1'])
    
    VALID_SOM_X_MIN = 10; VALID_SOM_X_MAX = 50 # Ajuste o range conforme o seu estudo
    df_consistent = df_successful[(df_successful['params_som_x'] >= VALID_SOM_X_MIN) & (df_successful['params_som_x'] <= VALID_SOM_X_MAX)]
    if df_consistent.empty:
        print("❌ Nenhum trial válido encontrado. Abortando."); exit()
    print(f"Encontrados {len(df_consistent)} trials consistentes para análise.")

    # --- Passo 2: Seleção Multivariada (AGORA COM 2 OBJETIVOS) ---
    print("\nIniciando seleção multivariada do melhor trial...")
    
    # a) Isolar as colunas de objetivos QE e TE
    objectives = df_consistent[['values_0', 'values_1']].values
    
    # b) Normalizar os valores para a escala [0, 1]
    scaler = MinMaxScaler()
    normalized_objectives = scaler.fit_transform(objectives)
    
    # c) Calcular a distância de cada trial ao ponto ideal (0, 0)
    distances_to_ideal = np.linalg.norm(normalized_objectives, axis=1)
    
    # d) Encontrar o trial com a menor distância
    best_trial_index = np.argmin(distances_to_ideal)
    best_trial_row = df_consistent.iloc[best_trial_index]
    
    print("\n--- Melhor Configuração Selecionada (Equilíbrio QE/TE) ---")
    print(f"Número do Trial: {int(best_trial_row['number'])}")
    print(f"Valores Originais (QE, TE): [{best_trial_row['values_0']:.4f}, {best_trial_row['values_1']:.4f}]")
    best_params = {
        'som_x': int(best_trial_row['params_som_x']),
        'learning_rate': best_trial_row['params_learning_rate'],
        'sigma': best_trial_row['params_sigma']
    }
    print(f"Hiperparâmetros: {best_params}")

    # --- O restante do script continua como antes ---
    
    # Passo 3: Preparar os DataLoaders
    print("\nPreparando DataLoaders de treino e validação...")
    df = pd.read_csv(MANIFEST_CSV)
    embedding_files = df['filepath'].str.replace('.png', '.pt').tolist()
    existing_files = [p for p in embedding_files if os.path.exists(os.path.join(EMBEDDINGS_DIR, p))]
    train_files, val_files = train_test_split(existing_files, test_size=0.2, random_state=42)
    train_dataset = TokenDataset(train_files, EMBEDDINGS_DIR)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_tokens, num_workers=0)
    val_dataset = TokenDataset(val_files, EMBEDDINGS_DIR)
    val_dataloader = DataLoader(val_dataset, batch_size=128, shuffle=False, collate_fn=collate_tokens, num_workers=0)
    print(f"{len(train_files)} arquivos para treino, {len(val_files)} para validação.")

    # Passo 4: Construir e treinar o modelo final
    print("\n--- Treinando o Modelo Final ---")
    final_som = SOM_PyTorch(
        x=best_params['som_x'], y=best_params['som_x'],
        input_dim=1536,
        sigma=best_params['sigma'], learning_rate=best_params['learning_rate'],
        device=DEVICE
    )
    history = final_som.train_batch_from_loader(train_dataloader, val_dataloader, num_epochs=NUM_EPOCHS_FINAL)

    # Passo 5: Salvar o modelo treinado
    FINAL_MODEL_PATH = f"final_som_model_{best_params['som_x']}x{best_params['som_x']}_2obj.pth"
    print(f"\nSalvando o modelo final em: {FINAL_MODEL_PATH}")
    torch.save({
        'som_x': final_som.x, 'som_y': final_som.y,
        'weights': final_som.weights.cpu(),
        'best_params': best_params,
        'source_study': STUDY_NAME
    }, FINAL_MODEL_PATH)
    
    # Passo 6: Plotar o histórico de treinamento
    print("\n--- Plotando o histórico de treinamento ---")
    fig, ax1 = plt.subplots(figsize=(12, 6))
    epochs = range(1, NUM_EPOCHS_FINAL + 1)
    color = 'tab:blue'; ax1.set_xlabel('Época'); ax1.set_ylabel('Erro de Quantização (QE)', color=color)
    ax1.plot(epochs, history['qe'], color=color, marker='o', label='QE'); ax1.tick_params(axis='y', labelcolor=color)
    ax2 = ax1.twinx(); color = 'tab:red'; ax2.set_ylabel('Erro Topográfico (TE)', color=color)
    ax2.plot(epochs, history['te'], color=color, marker='x', linestyle='--', label='TE'); ax2.tick_params(axis='y', labelcolor=color)
    plt.title('Evolução das Métricas de Validação por Época'); fig.tight_layout(); plt.grid(True)
    output_graph_path = f'training_history_{best_params["som_x"]}x{best_params["som_x"]}_2obj.png'
    plt.savefig(output_graph_path)
    plt.close()
    print(f"Gráfico do histórico de treinamento salvo em: {output_graph_path}")
    
    print("\n✅ Modelo final treinado e salvo com sucesso!")