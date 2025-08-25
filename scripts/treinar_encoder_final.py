import os
import gc
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
import optuna
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time

# --- 1. CLASSES E FUNÇÕES AUXILIARES ---
# (As classes DocumentTokenDataset, AttentionEncoder, Decoder e AttentionAutoencoder permanecem as mesmas)
class DocumentTokenDataset(Dataset):
    def __init__(self, file_paths, embeddings_dir): self.file_paths = file_paths; self.embeddings_dir = embeddings_dir
    def __len__(self): return len(self.file_paths)
    def __getitem__(self, idx):
        full_path = os.path.join(self.embeddings_dir, self.file_paths[idx]); return torch.load(full_path, weights_only=True)

class AttentionEncoder(nn.Module):
    def __init__(self, input_dim, n_layers=2, num_heads=4):
        super().__init__(); encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
    def forward(self, x):
        x = x.unsqueeze(0); refined_x = self.transformer_encoder(x); return refined_x.squeeze(0)

class Decoder(nn.Module):
    def __init__(self, input_dim, n_layers=2):
        super().__init__(); layers = [];
        for _ in range(n_layers): layers.append(nn.Linear(input_dim, input_dim)); layers.append(nn.ReLU())
        self.mlp = nn.Sequential(*layers)
    def forward(self, x): return self.mlp(x)

class AttentionAutoencoder(nn.Module):
    def __init__(self, input_dim=1536, encoder_layers=2, num_heads=4, decoder_layers=2):
        super().__init__(); self.encoder = AttentionEncoder(input_dim, encoder_layers, num_heads); self.decoder = Decoder(input_dim, decoder_layers)
    def forward(self, x):
        refined_sequence = self.encoder(x); reconstructed_x = self.decoder(refined_sequence); return reconstructed_x

# --- 2. CONFIGURAÇÃO ---
EMBEDDINGS_DIR = '/mnt/data/embeddings'
MANIFEST_CSV = '../data/sampled_dataset_png/manifest_linux.csv'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
STUDY_NAME = 'attention_autoencoder_v1' # Nome do estudo que você quer carregar
STORAGE_DB = 'sqlite:///../data/autoencoder_optimization.db'
FINAL_ENCODER_PATH = '../models/final_attention_encoder.pth' # Salvaremos apenas o encoder
NUM_EPOCHS_FINAL = 15 # Treinar o modelo final por mais tempo

# --- Configuração do Currículo de Treinamento ---
PHASE_1_EPOCHS = 12 # Épocas para a base geral (RVL-CDIP)
PHASE_2_EPOCHS = 8  # Épocas para os dados de especialização (Total = 20 épocas)

# --- 3. EXECUÇÃO PRINCIPAL ---
if __name__ == '__main__':
    # --- Passo 1: Carregar o estudo e selecionar o melhor trial ---
    print(f"Carregando estudo '{STUDY_NAME}' do banco de dados...")
    study = optuna.load_study(study_name=STUDY_NAME, storage=STORAGE_DB)
    best_trial = study.best_trial
    print("\n--- Melhor Configuração Encontrada ---")
    print(f"  - Hiperparâmetros: {best_trial.params}")
    best_params = best_trial.params

    # --- Passo 2: Preparar os DataLoaders para o Currículo ---
    print("\nPreparando DataLoaders para o Aprendizado por Currículo...")
    df = pd.read_csv(MANIFEST_CSV)
    
    # Filtrar arquivos que realmente existem
    existing_files_df = []
    for _, row in df.iterrows():
        path = os.path.join(EMBEDDINGS_DIR, row['filepath'].replace('.png', '.pt'))
        if os.path.exists(path): existing_files_df.append(row)
    df = pd.DataFrame(existing_files_df)

    # Dividir o DataFrame por fonte para o currículo
    df_general = df[df['source'] == 'rvl-cdip']
    df_specialist = df[df['source'] != 'rvl-cdip']
    
    # Dividir cada conjunto em treino e validação
    train_files_general, val_files_general = train_test_split(df_general['filepath'].str.replace('.png', '.pt').tolist(), test_size=0.2, random_state=42)
    train_files_specialist, val_files_specialist = train_test_split(df_specialist['filepath'].str.replace('.png', '.pt').tolist(), test_size=0.2, random_state=42)
    
    # DataLoaders para a Fase 1
    train_ds_general = DocumentTokenDataset(train_files_general, EMBEDDINGS_DIR)
    train_dl_general = DataLoader(train_ds_general, batch_size=1, shuffle=True, num_workers=0)
    val_ds_general = DocumentTokenDataset(val_files_general, EMBEDDINGS_DIR)
    val_dl_general = DataLoader(val_ds_general, batch_size=1, shuffle=False, num_workers=0)
    
    # DataLoaders para a Fase 2
    train_ds_specialist = DocumentTokenDataset(train_files_specialist, EMBEDDINGS_DIR)
    train_dl_specialist = DataLoader(train_ds_specialist, batch_size=1, shuffle=True, num_workers=0)
    val_ds_specialist = DocumentTokenDataset(val_files_specialist, EMBEDDINGS_DIR)
    val_dl_specialist = DataLoader(val_ds_specialist, batch_size=1, shuffle=False, num_workers=0)

    # --- Passo 3: Construir e treinar o modelo final com o currículo ---
    print("\n--- Treinando o Modelo Final com Aprendizado por Currículo ---")
    final_autoencoder = AttentionAutoencoder(
        input_dim=1536, encoder_layers=best_params['encoder_layers'],
        num_heads=best_params['num_heads'], decoder_layers=best_params['decoder_layers']
    ).to(DEVICE, dtype=torch.bfloat16)

    optimizer = torch.optim.Adam(final_autoencoder.parameters(), lr=best_params['learning_rate'])
    criterion = nn.MSELoss()
    history = {'train_loss': [], 'val_loss': [], 'phase': []}

    curriculum = [
        ('Fundação (RVL-CDIP)', train_dl_general, val_dl_general, PHASE_1_EPOCHS),
        ('Especialização', train_dl_specialist, val_dl_specialist, PHASE_2_EPOCHS)
    ]
    
    for phase_name, train_loader, val_loader, num_epochs in curriculum:
        print(f"\n--- Iniciando Fase de Treinamento: {phase_name} ---")
        for epoch in range(num_epochs):
            # Treinamento
            final_autoencoder.train()
            total_train_loss = 0
            for batch in tqdm(train_loader, desc=f"Fase '{phase_name}' - Época {epoch+1}/{num_epochs} [Treino]"):
                tokens = batch[0].to(DEVICE, dtype=torch.bfloat16)
                optimizer.zero_grad()
                reconstructed_tokens = final_autoencoder(tokens)
                loss = criterion(reconstructed_tokens, tokens)
                loss.backward(); optimizer.step()
                total_train_loss += loss.item()
            avg_train_loss = total_train_loss / len(train_loader)
            history['train_loss'].append(avg_train_loss)

            # Validação
            final_autoencoder.eval()
            total_val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    tokens = batch[0].to(DEVICE, dtype=torch.bfloat16)
                    reconstructed_tokens = final_autoencoder(tokens)
                    loss = criterion(reconstructed_tokens, tokens)
                    total_val_loss += loss.item()
            avg_val_loss = total_val_loss / len(val_loader)
            history['val_loss'].append(avg_val_loss)
            history['phase'].append(phase_name)

            print(f"Fim da Época: Perda de Treino = {avg_train_loss:.6f}, Perda de Validação = {avg_val_loss:.6f}")

    # --- Passo 4: Salvar APENAS o encoder treinado ---
    print(f"\nSalvando o ENCODER final em: {FINAL_ENCODER_PATH}")
    encoder_state_dict = final_autoencoder.encoder.state_dict()
    torch.save({'best_params': best_params, 'encoder_state_dict': encoder_state_dict}, FINAL_ENCODER_PATH)
    
    # --- Passo 5: Plotar o histórico de treinamento ---
    print("\n--- Plotando o histórico de treinamento ---")
    plt.figure(figsize=(12, 7))
    plt.plot(history['train_loss'], label='Perda de Treino', color='blue', alpha=0.6)
    plt.plot(history['val_loss'], label='Perda de Validação', color='orange')
    # Adicionar uma linha vertical para marcar a mudança de fase
    plt.axvline(x=PHASE_1_EPOCHS - 1, color='red', linestyle='--', label='Início da Fase de Especialização')
    plt.title('Evolução da Perda do Autoencoder por Época (Currículo)')
    plt.xlabel('Época')
    plt.ylabel('Erro de Reconstrução (MSE)')
    plt.legend(); plt.grid(True)
    output_graph_path = 'training_history_autoencoder_curriculum.png'
    plt.savefig(output_graph_path)
    plt.close()
    print(f"Gráfico do histórico de treinamento salvo em: {output_graph_path}")
    
    print("\n✅ Encoder final treinado e salvo com sucesso!")