# Inteligência Artificial 2 — Visão Computacional, Rastreamento sob Oclusão e Modelagem Temporal Avançada

Este repositório contém um ecossistema completo de Visão Computacional e Aprendizado de Máquina dividido em dois grandes eixos:

1. **Pipeline Clássico de Imagens (Etapas 1 e 2):** Extração de representações visuais (*embeddings*) de imagens do CIFAR-10 utilizando uma **CNN pré-treinada (ResNet18)** e posterior classificação supervisionada com **Random Forest**.
2. **Pipeline de Vídeo, Rastreamento e Predição Espaço-Temporal (Etapas 3 a 6):** Rastreamento de pedestres em vídeos, identificação e tratamento dinâmico de **oclusões** severas, extração de esqueletos corporais via **MediaPipe** e reconstituição/extrapolação de trajetórias comparando **Random Forest**, **Filtro de Kalman Estendido (EKF)**, **Micro-LSTM**, **Micro-GRU**, **Temporal Transformer** e **Graph Neural Networks Espaço-Temporais (T-GNN)**.

---

## Fluxo Arquitetural do Sistema

```
[Eixo 1 - Imagens]
Imagens (CIFAR-10) ──> ResNet18 (Penúltima Camada) ──> Vetor 512D ──> Random Forest ──> Classe

[Eixo 2 - Vídeos e Séries Temporais sob Oclusão]
Vídeos (Sintético / Reais)
       │
       ▼
Detecção & Percepção (Faster R-CNN / Subtração de Fundo)
       │
       ▼
Rastreamento & Detecção de Oclusão (Estados: Rastreando, Oclusão, Aguardando)
       │
       ├────────────────────────────────────────┐
       ▼                                        ▼
Poses Articulares (MediaPipe)            Baseline Cinético & RF
(33 marcos anatômicos ──> 5 centrais)   (Velocidades passadas ──> RF Residual)
       │                                        │
       ▼                                        ▼
Modelos Temporais Otimizados              Predição de Posição Futura
(EKF, LSTM, GRU, Transformer, T-GNN)            │
       │                                        │
       └──────────────────┬─────────────────────┘
                          ▼
            Avaliação Comparativa (RMSE px)
                          │
                          ▼
       Dashboards Analíticos & Vídeos Anotados
```

---

## Estrutura do Projeto

```
├── extrair_features.py              # Etapa 1: Baixa CIFAR-10 e extrai features com ResNet18
├── treinar_rf.py                    # Etapa 2: Treina e avalia o Random Forest em CIFAR-10
├── processar_video.py               # Etapa 3: Rastreamento, detecção de oclusões e baseline RF
├── gerar_video_sintetico.py         # Etapa 4: Gera vídeo sintético calibrado com Ground Truth exato
├── extrair_poses.py                 # Etapa 5: Extração de esqueletos (MediaPipe) e mascaramento em oclusão
├── modelos_temporais.py             # Implementação dos modelos: EKF, LSTM, GRU, Transformer e T-GNN
├── avaliar_comparativo.py           # Benchmark comparativo de erro (RMSE) entre todas as arquiteturas
├── gerar_visualizacoes_completas.py # Geração da suíte de gráficos analíticos e dashboard geral
├── relatorio_comparativo.md         # Relatório detalhado com discussões teóricas e empíricas
├── requirements.txt                 # Dependências do projeto
│
├── modelos/                         # Modelos pré-treinados do MediaPipe Pose Landmarker
│   ├── pose_landmarker_lite.task
│   └── pose_landmarker_full.task
│
├── videos/                          # Banco de vídeos para testes e benchmarks
│   ├── sintetico.mp4                # Vídeo paramétrico com ground truth
│   ├── real.mp4 (meu_video.mp4)     # Cena real no bosque com oclusão por árvore
│   ├── rua_.mp4                     # Transeunte em rua histórica (orientação vertical)
│   ├── cidade.mp4                   # Ambiente urbano com múltiplos pedestres e oclusão mútua
│   └── mato_video.mp4               # Trilha florestal longa com folhagem densa
│
└── resultados/                      # Artefatos gerados (vídeos com previsão, CSVs, gráficos e dashboards)
```

---

## Instalação e Preparação do Ambiente

O projeto é otimizado para rodar localmente tanto em ambientes com **GPU** quanto puramente em **CPU** (com restrição de 8 GB de RAM).

### 1. Criar e Ativar o Ambiente Virtual

* **Linux / macOS:**
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```

* **Windows (PowerShell):**
  ```powershell
  python -m venv .venv
  .venv\Scripts\Activate.ps1
  ```
  *(Se o PowerShell bloquear scripts: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`)*

### 2. Instalar Dependências

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Etapas 1 e 2 — Classificação de Imagens (CIFAR-10)

Um pipeline clássico demonstrando extração de características profundas combinada com algoritmos de *Ensemble Tree*:

```bash
# 1. Baixar CIFAR-10 e extrair features 512D usando ResNet18
python extrair_features.py

# 2. Treinar e avaliar o Random Forest (compara com baseline dummy)
python treinar_rf.py
```

### Opções Úteis
* `--n-treino` e `--n-teste`: Definir subconjuntos para execução rápida (ex: `--n-treino 1000 --n-teste 200`).
* `--tamanho`: Redimensionamento da entrada (ex: `--tamanho 112`).
* `--n-arvores` e `--max-profundidade`: Hiperparâmetros do Random Forest.

### Saídas Geradas em `resultados/`:
* `relatorio.txt`: Acurácia global, macro-F1 e métricas por classe.
* `matriz_confusao.png`: Matriz de confusão das 10 classes.
* `importancia_features.png`: Importância relativa das features da ResNet18.
* `random_forest.joblib`: Modelo serializado.

---

## Etapa 3 — Rastreamento em Vídeo, Oclusões e Baseline (`processar_video.py`)

Acompanha um pedestre em vídeo contínuo, identifica instantaneamente oclusões parciais ou totais e prevê sua trajetória utilizando um **Random Forest Residual** ajustado *walk-forward*:

```bash
python processar_video.py --video videos/real.mp4 --saida resultados/real_previsao.mp4
```

### Mecânica de Funcionamento:
1. **Percepção Visual:** Utiliza detector pré-treinado **Faster R-CNN** (`mobilenet` ou `resnet50`) ou o **`DetectorFundo`** (subtração de fundo adaptativa, ideal quando não se quer baixar pesos).
2. **Detecção de Oclusão:**
   * **Total:** O alvo desaparece abruptamente da cena.
   * **Parcial:** A caixa delimitadora sofre encolhimento vertical/horizontal súbito (`--encolhimento-min`).
3. **Previsão Cinética + Random Forest Residual:** Combina extrapolação cinemática de velocidade recente com correções aprendidas por um Random Forest por horizonte temporal.
4. **Trecho Extra no Vídeo:** Congela o último quadro e desenha a trajetória prevista no futuro além do término do vídeo (`--extra-segundos 4`).

---

## Etapa 4 — Vídeo Sintético com Ground Truth (`gerar_video_sintetico.py`)

Para viabilizar uma validação matemática incontestável do rastreador sob oclusão, foi implementado um gerador procedural:

```bash
python gerar_video_sintetico.py
```

* **Destaques Técnicos:**
  * Renderiza uma figura humana caminhando da esquerda para a direita com oscilação orgânica de braços e pernas.
  * Insere uma coluna de oclusão opaca calibrada ($X \in [537, 742]$).
  * Inclui uma **espera inicial de calibração limpa de 1,5 segundos (45 quadros)** sem a figura humana, garantindo que algoritmos como `DetectorFundo` aprendam o plano de fundo perfeitamente sem criar detecções fantasmas.
  * Exporta as coordenadas reais quadro a quadro em `resultados/sintetico_ground_truth.csv`.

---

## Etapa 5 — Extração de Poses Corporais com MediaPipe (`extrair_poses.py`)

Em vez de utilizar apenas o ponto dos pés (base da caixa delimitadora), este módulo extrai a anatomia esquelética completa:

```bash
python extrair_poses.py
```

* **Integração com Baseline:** Carrega os estados de cada quadro (`RASTREANDO`, `OCLUSAO`, etc.). Durante intervalos de oclusão, as coordenadas são **mascaradas com `NaN`**, simulando perda total de linha de visada dos sensores.
* **Marcos Articulares:** Utiliza os modelos MediaPipe em `modelos/pose_landmarker_full.task` para extrair 33 articulações 2D, com ênfase nas **5 articulações centrais** estruturais do corpo:
  * Nariz / Cabeça (Índice 0)
  * Ombro Esquerdo (11) e Ombro Direito (12)
  * Quadril Esquerdo (23) e Quadril Direito (24)
* Gera arquivos `poses_<video>.csv` para cada vídeo processado.

---

## Etapa 6 — Micro-Modelos Temporais Avançados (`modelos_temporais.py`)

Para superar as limitações do Random Forest sob oclusões complexas, foram desenvolvidas 5 arquiteturas otimizadas para processamento leve em CPU e baixo uso de memória:

1. **Extended Kalman Filter (PoseEKF):**
   * Filtro de Kalman cinemático estendido que opera no espaço 10D (5 nós $\times$ $(x, y)$).
   * Vetor de estado $30\text{D}$ contemplando posição, velocidade e aceleração com atualização de covariância de ruído e suavização de velocidade.
2. **Micro-LSTM:**
   * Arquitetura recorrente compacta (1 camada oculta, janela temporal deslizante).
   * Predição autorregressiva quadro a quadro durante a lacuna de oclusão.
3. **Micro-GRU:**
   * Célula recorrente simplificada, computacionalmente mais rápida e estável para trajetórias curtas.
4. **Micro-Temporal Transformer:**
   * Camada de Encoder Transformer com mecanismo de atenção multi-cabeça (*self-attention*, 2 heads) para capturar dependências e desacelerações temporais.
5. **Micro-T-GNN (Temporal Graph Neural Network):**
   * Rede neural espaço-temporal que opera sobre o grafo do esqueleto humano.
   * Modela as conexões anatômicas reais (cabeça-ombros, ombros-quadris) via Convolução em Grafo (GCN) seguida de uma camada recorrente GRU temporal, preservando a coerência da postura humana mesmo às cegas.

---

## Etapa 7 — Avaliação Comparativa Multimodelo (`avaliar_comparativo.py`)

Executa o treinamento e inferência de todos os modelos temporais e calcula o erro médio quadrático de predição (**RMSE em pixels**) durante o período exato de oclusão:

```bash
python avaliar_comparativo.py
```

### Resultados Quantitativos (RMSE em Pixels sob Oclusão):

| Arquitetura | Sintético (px) | Real / Bosque (px) | Rua (px) | Cidade (px) | Mato (px) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest (Baseline)** | 172.50 | **14.52** | 164.13 | **88.45** | **0.00** |
| **EKF (Kalman Estendido)** | 280.15 | 43.63 | 267.83 | 153.93 | **0.00** |
| **Micro-LSTM** | 282.49 | 51.68 | 130.59 | 209.71 | 3.70 |
| **Micro-GRU** | 282.27 | 173.31 | 125.90 | 218.17 | 44.35 |
| **Temporal Transformer** | **279.57** | 77.09 | **119.39** | 127.75 | 18.99 |
| **T-GNN (Skeletal Graph)** | 282.71 | 43.52 | 126.51 | 195.05 | 121.62 |

> *Para uma discussão teórica aprofundada dos resultados, consulte o arquivo [`relatorio_comparativo.md`](relatorio_comparativo.md).*

---

## Etapa 8 — Visualizações e Dashboards Analíticos (`gerar_visualizacoes_completas.py`)

Gera visualizações ricas de alta resolução reunindo todas as métricas em gráficos publicáveis:

```bash
python gerar_visualizacoes_completas.py
```

### Gráficos Gerados em `resultados/`:
* `dashboard_analise_completa.png`: Painel geral integrando métricas por horizonte, dinâmica de velocidade, distribuição de estados e trajetórias.
* `grafico_comparativo_modelos.png`: Comparação direta do RMSE entre todas as 6 arquiteturas nos 5 vídeos.
* `grafico_estados_temporal.png`: Linha do tempo estilo Gantt demonstrando as transições de estado (`AGUARDANDO`, `RASTREANDO`, `OCLUSAO`, `PREVISAO`).
* `grafico_trajetorias_2d_oclusao.png`: Projeção cartesiana $X \times Y$ da trajetória reconstruída por cada modelo durante a oclusão.
* `grafico_erro_horizontes_rf_vs_constante.png`: Curva de erro em função do horizonte futuro (+1s, +2s, +3s, +4s).
* `grafico_velocidade_dinamica.png`: Perfil de aceleração e velocidade dos pedestres ao longo do tempo.

---

## Como Reproduzir o Pipeline Completo (Ponta a Ponta)

Para executar todo o ciclo de experimentos a partir do zero:

```bash
# 1. Ativar o ambiente virtual e garantir dependências
source .venv/bin/activate  # ou .venv\Scripts\Activate.ps1 no Windows
pip install -r requirements.txt

# 2. Executar Etapas 1 e 2 (Classificação CIFAR-10)
python extrair_features.py
python treinar_rf.py

# 3. Gerar vídeo sintético com Ground Truth calibrado
python gerar_video_sintetico.py

# 4. Processar vídeos com o rastreador e detector
python processar_video.py --video videos/sintetico.mp4 --saida resultados/sintetico_previsao.mp4 --deteccoes resultados/sintetico_deteccoes.csv --reusar-deteccoes
python processar_video.py --video videos/real.mp4 --saida resultados/real_previsao.mp4 --largura 1280
python processar_video.py --video videos/rua_.mp4 --saida resultados/rua__previsao.mp4 --largura 720
python processar_video.py --video videos/cidade.mp4 --saida resultados/cidade_previsao.mp4 --largura 720
python processar_video.py --video videos/mato_video.mp4 --saida resultados/mato_video_previsao.mp4 --largura 720 --passo 4

# 5. Extrair poses MediaPipe com mascaramento de oclusão
python extrair_poses.py

# 6. Treinar e avaliar os modelos temporais (EKF, LSTM, GRU, Transformer, T-GNN)
python avaliar_comparativo.py

# 7. Gerar todos os gráficos analíticos e o dashboard unificado
python gerar_visualizacoes_completas.py
```

---

## Reprodutibilidade e Hardware

* **Seed Global:** Todos os sorteios aleatórios (PyTorch, NumPy, Scikit-learn) utilizam semente fixa (`42`).
* **Gerenciamento de Memória:** Tratamento automático de `MemoryError` com coletas forçadas de lixo (`gc.collect()`), viabilizando a execução completa mesmo em computadores sem GPU dedicada ou com limitações de memória RAM (8 GB).
