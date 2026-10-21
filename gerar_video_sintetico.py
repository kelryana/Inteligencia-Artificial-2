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
    
    # Obstáculo de oclusão no meio (x entre 537 e 742)
    coluna_x1 = int(largura * 0.42)
    coluna_x2 = int(largura * 0.58)
    
    # 1.5 segundos iniciais sem a pessoa (garante referência limpa para DetectorFundo)
    quadros_espera = int(1.5 * fps)
    quadros_fim = int(9.2 * fps)
    
    # Movimento horizontal: da esquerda (x=100) para a direita (x=1180)
    x_inicio = 100
    x_fim = largura - 100
    
    print(f"Gerando vídeo sintético: {total_quadros} quadros em {largura}x{altura} @ {fps}fps...")
    print(f"Espera inicial para calibração de fundo: {quadros_espera} quadros ({quadros_espera/fps:.1f}s)")
    print(f"Oclusão posicionada em X = [{coluna_x1}, {coluna_x2}]")
    
    gt_posicoes = [] # frame, tempo_s, visivel, cx, cy, h_pessoa
    deteccoes = []   # quadro, x1, y1, x2, y2, score, n_candidatos
    
    for f in range(total_quadros):
        # Fundo texturizado suave
        frame = np.full((altura, largura, 3), 235, dtype=np.uint8)
        # Chão
        cv2.rectangle(frame, (0, y_chao), (largura, altura), (180, 180, 180), -1)
        cv2.line(frame, (0, y_chao), (largura, y_chao), (120, 120, 120), 2)
        
        # Desenho da pessoa somente se já passou do período de espera inicial
        if f < quadros_espera:
            visivel = False
            gt_posicoes.append((f, f / fps, 0, "", "", ""))
        else:
            progresso = min(1.0, max(0.0, (f - quadros_espera) / (quadros_fim - quadros_espera)))
            cx = int(x_inicio + progresso * (x_fim - x_inicio))
            cy = y_chao
            
            # Ciclo de caminhada (pernas e braços oscilando)
            fase = 2 * math.pi * ((f - quadros_espera) * 1.5 / fps)
            balanco_perna = math.sin(fase) * 35
            balanco_braco = -math.sin(fase) * 25
            
            # Desenhar figura humana anatômica
            y_cabeca = cy - h_pessoa + 40
            y_ombro = y_cabeca + 45
            y_quadril = cy - int(h_pessoa * 0.45)
            
            # 1. Pernas
            cv2.line(frame, (cx - 15, y_quadril), (int(cx - 20 - balanco_perna), cy), (40, 40, 120), 14)
            cv2.line(frame, (cx + 15, y_quadril), (int(cx + 20 + balanco_perna), cy), (50, 50, 150), 14)
            
            # 2. Tronco / Corpo
            cv2.rectangle(frame, (cx - 35, y_ombro), (cx + 35, y_quadril), (60, 100, 180), -1)
            
            # 3. Cabeça
            cv2.circle(frame, (cx, y_cabeca), 30, (180, 200, 230), -1)
            cv2.circle(frame, (cx, y_cabeca), 30, (50, 50, 50), 2)
            
            # 4. Braços
            cv2.line(frame, (cx - 35, y_ombro + 10), (int(cx - 45 - balanco_braco), y_quadril - 10), (50, 80, 160), 10)
            cv2.line(frame, (cx + 35, y_ombro + 10), (int(cx + 45 + balanco_braco), y_quadril - 10), (50, 80, 160), 10)
            
            # Oclusão total ocorre quando a figura está amplamente atrás da coluna
            visivel = not (coluna_x1 - 35 <= cx <= coluna_x2 + 35)
            gt_posicoes.append((f, f / fps, 1 if visivel else 0, cx, cy, h_pessoa))
            
            if visivel and (f % 3 == 0):
                x1 = max(0.0, (cx - w_pessoa / 2) / largura)
                y1 = max(0.0, (cy - h_pessoa) / altura)
                x2 = min(1.0, (cx + w_pessoa / 2) / largura)
                y2 = min(1.0, cy / altura)
                deteccoes.append((f, f"{x1:.5f}", f"{y1:.5f}", f"{x2:.5f}", f"{y2:.5f}", "0.990", 1))
        
        # Coluna de oclusão desenhada por cima (sobrepõe a figura quando ela passa)
        cv2.rectangle(frame, (coluna_x1, 0), (coluna_x2, y_chao + 30), (70, 70, 90), -1)
        cv2.rectangle(frame, (coluna_x1, 0), (coluna_x2, y_chao + 30), (30, 30, 40), 3)
        
        # Textura na coluna
        for y_lin in range(40, y_chao, 60):
            cv2.line(frame, (coluna_x1 + 10, y_lin), (coluna_x2 - 10, y_lin), (90, 90, 110), 2)
            
        cv2.putText(frame, "OBSTACULO / OCLUSAO", (coluna_x1 + 15, int(altura * 0.4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        
        out.write(frame)
        
    out.release()
    print(f"Vídeo sintético salvo com sucesso em: {caminho_saida}")
    
    # Salva ground truth sintético
    gt_path = Path("resultados/sintetico_ground_truth.csv")
    gt_path.parent.mkdir(parents=True, exist_ok=True)
    with open(gt_path, "w", encoding="utf-8") as f:
        f.write("quadro,tempo_s,visivel,x_centro,y_pes,altura\n")
        for linha in gt_posicoes:
            f.write(f"{linha[0]},{linha[1]:.2f},{linha[2]},{linha[3]},{linha[4]},{linha[5]}\n")
    print(f"Ground truth salvo em: {gt_path}")
    
    # Salva detecções exatas ground truth para reuso opcional
    det_path = Path("resultados/sintetico_deteccoes.csv")
    with open(det_path, "w", encoding="utf-8") as f:
        f.write("quadro,x1,y1,x2,y2,score,n_candidatos\n")
        for d in deteccoes:
            f.write(f"{d[0]},{d[1]},{d[2]},{d[3]},{d[4]},{d[5]},{d[6]}\n")
    print(f"Detecções sintéticas salvas em: {det_path}")

if __name__ == "__main__":
    gerar_sintetico()
