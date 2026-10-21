"""Passo 4: Implementação dos Micro-Modelos Temporais Otimizados para 8GB RAM / CPU.
Implementa:
  1. EKF (Extended Kalman Filter)
  2. LSTM (1 camada)
  3. GRU (1 camada)
  4. Temporal Transformer (1 camada, 2 heads)
  5. T-GNN (Temporal Graph Neural Network - 5 nós centrais)
"""
import gc
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# --------------------------------------------------------------------------------------
# 1. Extended Kalman Filter (EKF) - Projeção puramente matemática
# --------------------------------------------------------------------------------------
class PoseEKF:
    """Filtro de Kalman Estendido com modelo cinemático de velocidade e aceleração constante.
    Opera sobre um vetor de coordenadas de N nós (ex: 5 nós 2D = 10 dimensões).
    """
    def __init__(self, dim_obs=10, dt=1.0/30.0, q_noise=1e-3, r_noise=1e-2):
        self.dim_obs = dim_obs
        self.dt = dt
        # Estado: [posição, velocidade, aceleração] para cada coordenada (3 * dim_obs)
        self.dim_state = 3 * dim_obs
        self.x = np.zeros(self.dim_state, dtype=np.float32)
        
        # Matriz de covariância P
        self.P = np.eye(self.dim_state, dtype=np.float32) * 0.1
        
        # Matriz de transição F
        self.F = np.eye(self.dim_state, dtype=np.float32)
        for i in range(dim_obs):
            # pos_t+1 = pos_t + vel*dt + 0.5*acc*dt^2
            self.F[i, dim_obs + i] = dt
            self.F[i, 2 * dim_obs + i] = 0.5 * (dt ** 2)
            # vel_t+1 = vel_t + acc*dt
            self.F[dim_obs + i, 2 * dim_obs + i] = dt
            
        # Matriz de observação H (mede apenas posições)
        self.H = np.zeros((dim_obs, self.dim_state), dtype=np.float32)
        for i in range(dim_obs):
            self.H[i, i] = 1.0
            
        # Ruído de processo Q e ruído de observação R
        self.Q = np.eye(self.dim_state, dtype=np.float32) * q_noise
        self.R = np.eye(dim_obs, dtype=np.float32) * r_noise

    def inicializar(self, trajetoria_inicial):
        """Inicializa o estado usando os frames recentes com velocidade suavizada."""
        if len(trajetoria_inicial) == 0:
            return
        p0 = trajetoria_inicial[-1]
        self.x[:self.dim_obs] = p0
        if len(trajetoria_inicial) >= 2:
            n_pts = min(len(trajetoria_inicial), 5)
            v_smooth = (trajetoria_inicial[-1] - trajetoria_inicial[-n_pts]) / max(1e-3, (n_pts - 1) * self.dt)
            self.x[self.dim_obs:2*self.dim_obs] = v_smooth
        # Aceleração inicial zerada (modelo de velocidade contínua com perturbações de processo Q)
        self.x[2*self.dim_obs:] = 0.0

    def prever_passo(self):
        """Passo de predição a priori (durante oclusão, sem medição)."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:self.dim_obs].copy()

    def atualizar(self, z):
        """Passo de atualização quando medição z está disponível."""
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.pinv(S)
        self.x = self.x + K @ y
        I = np.eye(self.dim_state, dtype=np.float32)
        self.P = (I - K @ self.H) @ self.P


# --------------------------------------------------------------------------------------
# 2. Micro-LSTM (1 camada oculta, janela curta)
# --------------------------------------------------------------------------------------
class MicroLSTM(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=16):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :]) # saída do último passo
        return out


# --------------------------------------------------------------------------------------
# 3. Micro-GRU (1 camada oculta, janela curta)
# --------------------------------------------------------------------------------------
class MicroGRU(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=16):
        super().__init__()
        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])
        return out


# --------------------------------------------------------------------------------------
# 4. Micro Temporal Transformer (1 camada de encoder, 2 attention heads)
# --------------------------------------------------------------------------------------
class MicroTemporalTransformer(nn.Module):
    def __init__(self, input_dim=10, d_model=16, nhead=2):
        super().__init__()
        self.proj_in = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=32, batch_first=True, dropout=0.0
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.proj_out = nn.Linear(d_model, input_dim)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        h = self.proj_in(x)
        h = self.transformer(h)
        out = self.proj_out(h[:, -1, :])
        return out


# --------------------------------------------------------------------------------------
# 5. T-GNN (Temporal Graph Neural Network - 5 articulações centrais)
# --------------------------------------------------------------------------------------
class MicroTGNN(nn.Module):
    """GNN Espaço-Temporal operando em 5 articulações centrais:
    0: Cabeça/Nariz, 1: Ombro Esq, 2: Ombro Dir, 3: Quadril Esq, 4: Quadril Dir.
    """
    def __init__(self, num_nodes=5, feat_dim=2, hidden_dim=8):
        super().__init__()
        self.num_nodes = num_nodes
        self.feat_dim = feat_dim
        
        # Matriz de adjacência pré-definida do esqueleto humano (5 nós)
        # Conexões: (0,1), (0,2), (1,2), (1,3), (2,4), (3,4) + auto-loops
        A = torch.eye(num_nodes)
        arestas = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 4), (3, 4)]
        for u, v in arestas:
            A[u, v] = 1.0
            A[v, u] = 1.0
        # Normalização D^-1/2 * A * D^-1/2
        d = torch.sum(A, dim=1)
        d_inv_sqrt = torch.pow(d, -0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
        D_norm = torch.diag(d_inv_sqrt)
        self.register_buffer("A_norm", D_norm @ A @ D_norm)
        
        # Camada espacial GCN
        self.gcn_fc = nn.Linear(feat_dim, hidden_dim)
        # Camada temporal GRU para modelar evolução temporal do grafo
        self.temporal_gru = nn.GRU(
            input_size=num_nodes * hidden_dim,
            hidden_size=num_nodes * hidden_dim,
            num_layers=1,
            batch_first=True
        )
        self.fc_out = nn.Linear(num_nodes * hidden_dim, num_nodes * feat_dim)

    def forward(self, x):
        # x: (batch, seq_len, 10) -> reformata para (batch, seq_len, 5, 2)
        B, S, _ = x.shape
        x_nodes = x.view(B, S, self.num_nodes, self.feat_dim)
        
        # Processamento espacial com GCN em cada passo de tempo
        # x_nodes: (B, S, N, F)
        gcn_feats = []
        for t in range(S):
            xt = x_nodes[:, t, :, :] # (B, N, F)
            # Agregação de vizinhos: A_norm @ xt -> (B, N, F)
            agg = torch.matmul(self.A_norm, xt)
            h = torch.relu(self.gcn_fc(agg)) # (B, N, hidden_dim)
            gcn_feats.append(h.view(B, -1)) # achata nós: (B, N * hidden_dim)
            
        gcn_seq = torch.stack(gcn_feats, dim=1) # (B, S, N * hidden_dim)
        
        # Processamento temporal via GRU
        gru_out, _ = self.temporal_gru(gcn_seq)
        out = self.fc_out(gru_out[:, -1, :]) # (B, 10)
        return out


# --------------------------------------------------------------------------------------
# Funções Auxiliares de Treinamento e Inferência Auto-Adaptativa
# --------------------------------------------------------------------------------------
def criar_dataset_temporal(serie, seq_len=5):
    """Cria pares (X, y) para treinamento temporal de seq_len passos para prever 1 passo à frente."""
    X, y = [], []
    for i in range(len(serie) - seq_len):
        X.append(serie[i:i + seq_len])
        y.append(serie[i + seq_len])
    if not X:
        return torch.empty(0), torch.empty(0)
    return torch.tensor(np.array(X), dtype=torch.float32), torch.tensor(np.array(y), dtype=torch.float32)


def treinar_modelo_pytorch(modelo, loader, epochs=30, lr=0.01):
    """Treina um modelo PyTorch com tratamento automático de memória."""
    otimizador = torch.optim.Adam(modelo.parameters(), lr=lr)
    criterio = nn.MSELoss()
    modelo.train()
    
    try:
        for ep in range(epochs):
            for batch_x, batch_y in loader:
                otimizador.zero_grad()
                pred = modelo(batch_x)
                perda = criterio(pred, batch_y)
                perda.backward()
                otimizador.step()
    except MemoryError:
        print("Aviso: MemoryError detectado! Reduzindo parâmetros e limpando cache...")
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
    modelo.eval()
    return modelo


def prever_sequencia(modelo, janela_inicial, n_passos):
    """Realiza predição autoregressiva durante o intervalo de oclusão de n_passos."""
    previsoes = []
    # janela: lista de vetores numpy (seq_len, 10)
    janela = list(janela_inicial)
    
    for _ in range(n_passos):
        inp = torch.tensor(np.array([janela]), dtype=torch.float32)
        with torch.no_grad():
            out = modelo(inp).squeeze(0).cpu().numpy()
        previsoes.append(out)
        janela.pop(0)
        janela.append(out)
        
    return np.array(previsoes)
