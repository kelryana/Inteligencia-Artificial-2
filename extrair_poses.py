"""Passo 3: Extração de poses CPU-Friendly com MediaPipe.
Extrai coordenadas 2D das articulações quadro a quadro e mascara com NaN
nos frames identificados como OCLUSAO pelo modelo base (CNN + RF).
"""
import argparse
import csv
import gc
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Articulações de interesse:
# 0: nariz
# 11: ombro_esq, 12: ombro_dir
# 23: quadril_esq, 24: quadril_dir
# 13: cotovelo_esq, 14: cotovelo_dir
# 25: joelho_esq, 26: joelho_dir
# 27: tornozelo_esq, 28: tornozelo_dir
PONTOS_CENTRAIS = [0, 11, 12, 23, 24] # 5 articulações centrais para T-GNN
NOMES_CENTRAIS = ["nariz", "ombro_esq", "ombro_dir", "quadril_esq", "quadril_dir"]


def carregar_estados_baseline(caminho_csv):
    """Carrega o estado (RASTREANDO, OCLUSAO, AGUARDANDO) de cada quadro a partir do log do baseline."""
    estados = {}
    if not Path(caminho_csv).exists():
        print(f"Aviso: {caminho_csv} não encontrado.")
        return estados
    with open(caminho_csv, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            estados[int(row["quadro"])] = row["estado"]
    return estados


def processar_video_pose(video_path, baseline_csv, saida_csv, model_path="modelos/pose_landmarker_full.task", largura_max=1280):
    video_path = Path(video_path)
    saida_csv = Path(saida_csv)
    saida_csv.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\n--- Extraindo poses de {video_path} ---")
    estados_base = carregar_estados_baseline(baseline_csv)
    
    # Inicializa detector do MediaPipe Pose
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=False
    )
    detector = vision.PoseLandmarker.create_from_options(options)
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Erro ao abrir vídeo: {video_path}")
        
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h0 = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Redimensiona para resolução padrão
    if largura_max and w0 > largura_max:
        w = largura_max
        h = int(h0 * largura_max / w0)
    else:
        w, h = w0, h0
        
    cabecalho = ["quadro", "tempo_s", "estado", "mascarado_oclusao"]
    # Coordenadas das 5 centrais
    for nome in NOMES_CENTRAIS:
        cabecalho.extend([f"{nome}_x", f"{nome}_y", f"{nome}_vis"])
    # Coordenadas de todos os 33 landmarks para compatibilidade geral
    for i in range(33):
        cabecalho.extend([f"lm_{i}_x", f"lm_{i}_y"])
        
    linhas_saida = []
    
    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
            
        tempo_s = i / fps
        estado = estados_base.get(i, "DESCONHECIDO")
        # Se for oclusão pelo baseline, marcar como oclusão
        eh_oclusao = (estado == "OCLUSAO")
        
        # Mascaramento: se estiver em oclusão, não passamos detecção (mascaramento imposto)
        if eh_oclusao:
            linha = [i, f"{tempo_s:.3f}", estado, 1]
            # Preenche 5 centrais com NaN
            for _ in NOMES_CENTRAIS:
                linha.extend(["nan", "nan", "nan"])
            # Preenche 33 landmarks com NaN
            for _ in range(33):
                linha.extend(["nan", "nan"])
            linhas_saida.append(linha)
        else:
            # Redimensiona frame se necessário
            if (w, h) != (frame.shape[1], frame.shape[0]):
                frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
            
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            resultado = detector.detect(mp_image)
            
            linha = [i, f"{tempo_s:.3f}", estado, 0]
            if resultado.pose_landmarks and len(resultado.pose_landmarks) > 0:
                lms = resultado.pose_landmarks[0]
                # 5 Centrais
                for idx in PONTOS_CENTRAIS:
                    lm = lms[idx]
                    linha.extend([f"{lm.x * w:.2f}", f"{lm.y * h:.2f}", f"{lm.visibility:.3f}"])
                # 33 landmarks
                for idx in range(33):
                    lm = lms[idx]
                    linha.extend([f"{lm.x * w:.2f}", f"{lm.y * h:.2f}"])
            else:
                # Não detectou mesmo sem oclusão marcada
                for _ in NOMES_CENTRAIS:
                    linha.extend(["nan", "nan", "nan"])
                for _ in range(33):
                    linha.extend(["nan", "nan"])
            linhas_saida.append(linha)
            
        # Limpeza periódica de memória RAM (CPU-friendly)
        if i > 0 and i % 50 == 0:
            gc.collect()
            print(f"  Quadro {i}/{total_frames} ({(i/total_frames)*100:.1f}%) processado...")
            
    cap.release()
    detector.close()
    gc.collect()
    
    with open(saida_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(cabecalho)
        writer.writerows(linhas_saida)
        
    print(f"Poses salvas em: {saida_csv} ({len(linhas_saida)} quadros)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sintetico", default="videos/sintetico.mp4")
    parser.add_argument("--sintetico-base", default="resultados/sintetico_previsao.csv")
    parser.add_argument("--sintetico-saida", default="resultados/poses_sintetico.csv")
    parser.add_argument("--real", default="videos/real.mp4")
    parser.add_argument("--real-base", default="resultados/real_previsao.csv")
    parser.add_argument("--real-saida", default="resultados/poses_real.csv")
    args = parser.parse_args()
    
    # 1. Processar vídeo sintético
    processar_video_pose(args.sintetico, args.sintetico_base, args.sintetico_saida)
    
    # 2. Processar vídeo real
    processar_video_pose(args.real, args.real_base, args.real_saida)
    print("\nExtração de poses concluída com sucesso para ambos os vídeos!")


if __name__ == "__main__":
    main()
