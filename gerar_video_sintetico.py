"""Gera um vídeo sintético de teste com figura humana articulada caminhando e uma coluna de oclusão.
Garante Ground Truth exato para avaliar rastreamento e predição temporal sob oclusão.
"""
import math
import numpy as np
import cv2
from pathlib import Path

def gerar_sintetico(caminho_saida="videos/sintetico.mp4", duracao_s=10.0, fps=30, largura=1280, altura=720):
    Path(caminho_saida).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(caminho_saida), fourcc, fps, (largura, altura))
    
    total_quadros = int(duracao_s * fps)
    
    # Parâmetros da pessoa sintética
    h_pessoa = 360
    w_pessoa = 140
    y_chao = int(altura * 0.85)
    
    # Movimento horizontal: da esquerda (x=100) para a direita (x=1180)
    x_inicio = 120
    x_fim = largura - 120
    
    # Obstáculo de oclusão no meio (ex: x entre 540 e 740)
    coluna_x1 = int(largura * 0.42)
    coluna_x2 = int(largura * 0.58)
    coluna_largura = coluna_x2 - coluna_x1
    
    print(f"Gerando vídeo sintético: {total_quadros} quadros em {largura}x{altura} @ {fps}fps...")
    print(f"Oclusão posicionada em X = [{coluna_x1}, {coluna_x2}]")
    
    gt_posicoes = [] # frame, cx, cy, visivel
    
    for f in range(total_quadros):
        # Fundo texturizado suave
        frame = np.full((altura, largura, 3), 235, dtype=np.uint8)
        # Chão
        cv2.rectangle(frame, (0, y_chao), (largura, altura), (180, 180, 180), -1)
        cv2.line(frame, (0, y_chao), (largura, y_chao), (120, 120, 120), 2)
        
        # Posição atual da pessoa (movimento horizontal suave)
        progresso = f / (total_quadros - 1)
        cx = int(x_inicio + progresso * (x_fim - x_inicio))
        cy = y_chao # pés no chão
        
        # Ciclo de caminhada (pernas e braços oscilando)
        fase = 2 * math.pi * (f * 1.5 / fps)
        balanco_perna = math.sin(fase) * 35
        balanco_braco = -math.sin(fase) * 25
        
        # Desenhar figura humana (corpo realista para detector e mediapipe)
        # Cabeça / Nariz
        y_cabeca = cy - h_pessoa + 40
        # Tronco
        y_ombro = y_cabeca + 45
        y_quadril = cy - int(h_pessoa * 0.45)
        
        # 1. Pernas (atrás)
        # Perna esquerda
        cv2.line(frame, (cx - 15, y_quadril), (int(cx - 20 - balanco_perna), cy), (40, 40, 120), 14)
        # Perna direita
        cv2.line(frame, (cx + 15, y_quadril), (int(cx + 20 + balanco_perna), cy), (50, 50, 150), 14)
        
        # 2. Tronco / Corpo (jaqueta/camisa)
        cv2.rectangle(frame, (cx - 35, y_ombro), (cx + 35, y_quadril), (60, 100, 180), -1)
        
        # 3. Cabeça
        cv2.circle(frame, (cx, y_cabeca), 30, (180, 200, 230), -1)
        cv2.circle(frame, (cx, y_cabeca), 30, (50, 50, 50), 2)
        
        # 4. Braços
        cv2.line(frame, (cx - 35, y_ombro + 10), (int(cx - 45 - balanco_braco), y_quadril - 10), (50, 80, 160), 10)
        cv2.line(frame, (cx + 35, y_ombro + 10), (int(cx + 45 + balanco_braco), y_quadril - 10), (50, 80, 160), 10)
        
        # Verificar oclusão: se a pessoa estiver na região da coluna
        # Se a pessoa está atrás da coluna, desenhamos a coluna por cima!
        cv2.rectangle(frame, (coluna_x1, 0), (coluna_x2, y_chao + 30), (70, 70, 90), -1)
        cv2.rectangle(frame, (coluna_x1, 0), (coluna_x2, y_chao + 30), (30, 30, 40), 3)
        
        # Textura na coluna
        for y_lin in range(40, y_chao, 60):
            cv2.line(frame, (coluna_x1 + 10, y_lin), (coluna_x2 - 10, y_lin), (90, 90, 110), 2)
            
        cv2.putText(frame, "OBSTACULO / OCLUSAO", (coluna_x1 + 15, int(altura * 0.4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        
        # Se a pessoa está amplamente atrás da coluna
        visivel = not (coluna_x1 - 30 <= cx <= coluna_x2 + 30)
        gt_posicoes.append((f, f / fps, 1 if visivel else 0, cx, cy, h_pessoa))
        
        out.write(frame)
        
    out.release()
    print(f"Vídeo sintético salvo com sucesso em: {caminho_saida}")
    
    # Salva ground truth sintético
    gt_path = Path("resultados/sintetico_ground_truth.csv")
    gt_path.parent.mkdir(parents=True, exist_ok=True)
    with open(gt_path, "w") as f:
        f.write("quadro,tempo_s,visivel,x_centro,y_pes,altura\n")
        for linha in gt_posicoes:
            f.write(f"{linha[0]},{linha[1]:.2f},{linha[2]},{linha[3]},{linha[4]},{linha[5]}\n")
    print(f"Ground truth salvo em: {gt_path}")

if __name__ == "__main__":
    gerar_sintetico()
