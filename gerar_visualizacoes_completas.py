"""Gera visualizações completas e ricas a partir de todos os CSVs, TXTs e PKLs de resultados.
Produz gráficos de alta resolução para análise de tracking, oclusões, previsões temporais e comparação de modelos.
"""
import csv
import glob
import os
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


def carregar_metricas_txt():
    """Lê todos os arquivos *_previsao_metricas.txt e extrai erros de RF vs Movimento Constante."""
    dados_horizontes = {}
    for caminho in sorted(glob.glob("resultados/*_previsao_metricas.txt")):
        nome = Path(caminho).stem.replace("_previsao_metricas", "").capitalize()
        if nome == "Video":
            continue # ignora duplicata de video_previsao (legado)
        dados_horizontes[nome] = {"horizontes": [], "rf": [], "constante": []}
        
        with open(caminho, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("|")
                if len(parts) >= 3 and "+ " in parts[0] or "+1" in parts[0] or "+2" in parts[0] or "+3" in parts[0] or "+4" in parts[0]:
                    try:
                        h = parts[0].strip()
                        rf_err = float(parts[1].strip())
                        const_err = float(parts[2].strip())
                        dados_horizontes[nome]["horizontes"].append(h)
                        dados_horizontes[nome]["rf"].append(rf_err)
                        dados_horizontes[nome]["constante"].append(const_err)
                    except ValueError:
                        continue
    return dados_horizontes


def plotar_estados_temporal():
    """Gera gráfico de barras horizontais tipo Gantt mostrando os estados de cada vídeo ao longo do tempo."""
    csvs = [
        ("Sintético", "resultados/sintetico_previsao.csv"),
        ("Real", "resultados/real_previsao.csv"),
        ("Rua", "resultados/rua__previsao.csv"),
        ("Cidade", "resultados/cidade_previsao.csv"),
        ("Mato", "resultados/mato_video_previsao.csv"),
    ]
    
    cores_estado = {
        "AGUARDANDO": "#7f7f7f",
        "RASTREANDO": "#2ca02c",
        "OCLUSAO": "#d62728",
        "PREVISAO": "#ff7f0e"
    }
    
    fig, ax = plt.subplots(figsize=(12, 5.5), dpi=150)
    
    y_pos = 0
    yticks = []
    yticklabels = []
    
    for nome, caminho in csvs:
        if not Path(caminho).exists():
            continue
        with open(caminho, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
            
        if not rows:
            continue
            
        fps = 30.0
        # Agrupa intervalos contínuos de estados
        intervalos = []
        cur_estado = rows[0]["estado"]
        inicio_t = float(rows[0]["tempo_s"])
        
        for r in rows[1:]:
            t = float(r["tempo_s"])
            st = r["estado"]
            if st != cur_estado:
                intervalos.append((cur_estado, inicio_t, t - inicio_t))
                cur_estado = st
                inicio_t = t
        fim_t = float(rows[-1]["tempo_s"])
        intervalos.append((cur_estado, inicio_t, fim_t - inicio_t))
        
        for st, t_ini, dur in intervalos:
            cor = cores_estado.get(st, "#333333")
            ax.barh(y_pos, dur, left=t_ini, height=0.55, color=cor, edgecolor="black", linewidth=0.4)
            
        yticks.append(y_pos)
        yticklabels.append(f"{nome} ({fim_t:.1f}s)")
        y_pos += 1
        
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels, fontsize=11, fontweight="bold")
    ax.set_xlabel("Linha do Tempo (segundos)", fontsize=11, fontweight="bold")
    ax.set_title("Evolução Temporal dos Estados de Rastreamento sob Oclusão e Previsão", fontsize=12, pad=12, fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    
    # Legenda customizada
    legend_elements = [
        patches.Patch(facecolor=cores_estado["AGUARDANDO"], edgecolor="black", label="Aguardando Aparição"),
        patches.Patch(facecolor=cores_estado["RASTREANDO"], edgecolor="black", label="Rastreando Ativo"),
        patches.Patch(facecolor=cores_estado["OCLUSAO"], edgecolor="black", label="Oclusão Detectada"),
        patches.Patch(facecolor=cores_estado["PREVISAO"], edgecolor="black", label="Previsão Extrapolada"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=9.5, framealpha=0.9)
    
    fig.tight_layout()
    caminho_saida = "resultados/grafico_estados_temporal.png"
    plt.savefig(caminho_saida)
    plt.close()
    print(f"Salvo: {caminho_saida}")


def plotar_erro_horizontes():
    """Gera gráfico de comparação do erro do Random Forest vs Movimento Constante por horizonte."""
    dados = carregar_metricas_txt()
    if not dados:
        return
        
    valid_videos = [v for v, d in dados.items() if len(d["horizontes"]) > 0]
    n_v = len(valid_videos)
    if n_v == 0:
        return
        
    fig, axes = plt.subplots(1, n_v, figsize=(4 * n_v, 4.5), dpi=150, sharey=False)
    if n_v == 1:
        axes = [axes]
        
    for ax, nome in zip(axes, valid_videos):
        d = dados[nome]
        h = d["horizontes"]
        rf = d["rf"]
        const = d["constante"]
        
        x = np.arange(len(h))
        larg = 0.35
        
        ax.bar(x - larg/2, rf, larg, label="Random Forest", color="#2ca02c", edgecolor="black", linewidth=0.6)
        ax.bar(x + larg/2, const, larg, label="Mov. Constante", color="#1f77b4", edgecolor="black", linewidth=0.6)
        
        ax.set_xticks(x)
        ax.set_xticklabels(h, fontsize=10)
        ax.set_title(f"Vídeo {nome}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Horizonte Futuro", fontsize=10)
        ax.set_ylabel("Erro Médio de Posição (px)", fontsize=10)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        ax.legend(fontsize=8.5)
        
    fig.suptitle("Desempenho da Predição de Trajetória: Random Forest vs Movimento Constante", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    caminho_saida = "resultados/grafico_erro_horizontes_rf_vs_constante.png"
    plt.savefig(caminho_saida)
    plt.close()
    print(f"Salvo: {caminho_saida}")


def plotar_trajetorias_oclusao():
    """Visualiza as trajetórias projetadas por cada modelo no espaço 2D durante a oclusão."""
    caminho_pkl = Path("resultados/trajetorias_modelos.pkl")
    if not caminho_pkl.exists():
        return
        
    with open(caminho_pkl, "rb") as f:
        trajs = pickle.load(f)
        
    # Plota para os vídeos que tiverem trajetórias registradas
    videos = [v for v in trajs.keys() if "gt" in trajs[v] and len(trajs[v]["gt"]) > 0]
    if not videos:
        return
        
    fig, axes = plt.subplots(1, len(videos), figsize=(5.5 * len(videos), 5), dpi=150)
    if len(videos) == 1:
        axes = [axes]
        
    cores = {
        "gt": ("#000000", "Ground Truth / Ref", 3.0, "-"),
        "rf": ("#ff7f0e", "Random Forest", 2.0, "--"),
        "ekf": ("#2ca02c", "EKF", 2.0, "-."),
        "lstm": ("#1f77b4", "Micro-LSTM", 1.8, ":"),
        "gru": ("#9467bd", "Micro-GRU", 1.8, ":"),
        "trans": ("#8c564b", "Temporal Transformer", 1.8, "--"),
        "tgnn": ("#e377c2", "T-GNN", 2.2, "-")
    }
    
    for ax, v_nome in zip(axes, videos):
        v_data = trajs[v_nome]
        for mod, (cor, label, lw, ls) in cores.items():
            if mod in v_data:
                pts = v_data[mod]
                if len(pts) > 0:
                    ax.plot(pts[:, 0], pts[:, 1], color=cor, label=label, linewidth=lw, linestyle=ls)
                    # Marca ponto de início e fim
                    ax.scatter(pts[0, 0], pts[0, 1], color=cor, s=25, zorder=5)
                    ax.scatter(pts[-1, 0], pts[-1, 1], color=cor, marker="x", s=30, zorder=5)
                    
        ax.set_title(f"Trajetórias sob Oclusão: {v_nome}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Coordenada X (pixels)", fontsize=10)
        ax.set_ylabel("Coordenada Y (pixels)", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(fontsize=8, loc="best")
        
    fig.suptitle("Comparativo de Projeção Espacial 2D durante Janela de Oclusão", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    caminho_saida = "resultados/grafico_trajetorias_2d_oclusao.png"
    plt.savefig(caminho_saida)
    plt.close()
    print(f"Salvo: {caminho_saida}")


def plotar_dinamica_velocidade():
    """Gera gráficos de perfil de velocidade e aceleração instantânea a partir das detecções."""
    csvs = [
        ("Sintético", "resultados/sintetico_previsao.csv"),
        ("Real", "resultados/real_previsao.csv"),
        ("Rua", "resultados/rua__previsao.csv"),
        ("Cidade", "resultados/cidade_previsao.csv"),
    ]
    
    fig, ax = plt.subplots(figsize=(11, 5), dpi=150)
    
    cores = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    
    for (nome, caminho), cor in zip(csvs, cores):
        if not Path(caminho).exists():
            continue
        with open(caminho, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
            
        t_vals = []
        v_vals = []
        
        pts_validos = []
        for r in rows:
            if r["x_centro"] != "" and r["y_pes"] != "" and r["estado"] == "RASTREANDO":
                pts_validos.append((float(r["tempo_s"]), float(r["x_centro"]), float(r["y_pes"])))
                
        if len(pts_validos) < 5:
            continue
            
        for i in range(1, len(pts_validos)):
            dt = pts_validos[i][0] - pts_validos[i-1][0]
            if 0 < dt < 0.2:
                dx = pts_validos[i][1] - pts_validos[i-1][1]
                dy = pts_validos[i][2] - pts_validos[i-1][2]
                vel = np.hypot(dx, dy) / dt
                t_vals.append(pts_validos[i][0])
                v_vals.append(vel)
                
        if len(v_vals) > 5:
            # Suavização por média móvel
            v_smooth = np.convolve(v_vals, np.ones(5)/5, mode="valid")
            t_smooth = t_vals[2:-2]
            ax.plot(t_smooth, v_smooth, label=nome, color=cor, linewidth=1.8)
            
    ax.set_xlabel("Tempo (segundos)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Velocidade Observada (pixels/s)", fontsize=11, fontweight="bold")
    ax.set_title("Dinâmica de Velocidade Observada das Pessoas ao Longo do Tempo", fontsize=12, pad=12, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=10)
    
    fig.tight_layout()
    caminho_saida = "resultados/grafico_velocidade_dinamica.png"
    plt.savefig(caminho_saida)
    plt.close()
    print(f"Salvo: {caminho_saida}")


def plotar_dashboard_geral():
    """Gera um Dashboard Executivo integrado com 4 quadrantes resumindo todos os experimentos."""
    fig = plt.figure(figsize=(15, 10), dpi=160)
    gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.25)
    
    # 1. Gráfico Multivídeo de RMSE
    ax1 = fig.add_subplot(gs[0, 0])
    # Lê os dados do relatório comparativo markdown ou recalcula
    modelos = ["Random Forest", "EKF", "LSTM", "GRU", "Transformer", "T-GNN"]
    # Coleta RMSE médio ou por vídeo principal
    caminho_pkl = Path("resultados/trajetorias_modelos.pkl")
    if caminho_pkl.exists():
        with open(caminho_pkl, "rb") as f:
            trajs = pickle.load(f)
        # Calcula RMSE dos vídeos
        mod_keys = ["rf", "ekf", "lstm", "gru", "trans", "tgnn"]
        for v_nome in ["Sintético", "Real", "Cidade"]:
            if v_nome in trajs and "gt" in trajs[v_nome]:
                gt = trajs[v_nome]["gt"]
                rmses = []
                for mk in mod_keys:
                    p = trajs[v_nome].get(mk, gt)
                    diff = np.nan_to_num(p) - np.nan_to_num(gt)
                    rmse = np.sqrt(np.mean(np.sum(diff**2, axis=1)))
                    rmses.append(rmse)
                ax1.plot(modelos, rmses, marker="o", label=v_nome, linewidth=2)
    ax1.set_title("(A) Erro de Posição sob Oclusão (RMSE em px)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("RMSE (pixels)", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.set_xticklabels(modelos, rotation=20, ha="right", fontsize=9)
    ax1.legend(fontsize=9)
    
    # 2. Distribuição Percentual de Estados (Pizza agregada)
    ax2 = fig.add_subplot(gs[0, 1])
    contagem_estados = {"RASTREANDO": 0, "OCLUSAO": 0, "AGUARDANDO": 0, "PREVISAO": 0}
    for c in glob.glob("resultados/*_previsao.csv"):
        if "video_previsao" in c: continue
        with open(c, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                st = r.get("estado", "")
                if st in contagem_estados:
                    contagem_estados[st] += 1
    
    labels = ["Rastreando", "Oclusão", "Aguardando", "Previsão"]
    valores = [contagem_estados["RASTREANDO"], contagem_estados["OCLUSAO"], contagem_estados["AGUARDANDO"], contagem_estados["PREVISAO"]]
    cores_pie = ["#2ca02c", "#d62728", "#7f7f7f", "#ff7f0e"]
    wedges, texts, autotexts = ax2.pie(valores, labels=labels, autopct="%1.1f%%", colors=cores_pie,
                                       startangle=140, textprops=dict(fontweight="semibold"))
    ax2.set_title("(B) Distribuição Global de Quadros por Estado", fontsize=11, fontweight="bold")
    
    # 3. Trajetória 2D Sintética sob Oclusão
    ax3 = fig.add_subplot(gs[1, 0])
    if caminho_pkl.exists() and "Sintético" in trajs:
        s_data = trajs["Sintético"]
        gt = s_data["gt"]
        ax3.plot(gt[:, 0], gt[:, 1], color="black", label="Ground Truth", linewidth=3)
        if "rf" in s_data: ax3.plot(s_data["rf"][:, 0], s_data["rf"][:, 1], label="Random Forest", linestyle="--", linewidth=2)
        if "ekf" in s_data: ax3.plot(s_data["ekf"][:, 0], s_data["ekf"][:, 1], label="EKF", linestyle="-.", linewidth=2)
        if "trans" in s_data: ax3.plot(s_data["trans"][:, 0], s_data["trans"][:, 1], label="Transformer", linestyle=":", linewidth=2)
        if "tgnn" in s_data: ax3.plot(s_data["tgnn"][:, 0], s_data["tgnn"][:, 1], label="T-GNN", linestyle="-", linewidth=2)
        # Desenha coluna de oclusão
        ax3.axvspan(537, 742, color="gray", alpha=0.25, label="Coluna de Oclusão")
    ax3.set_title("(C) Projeção Espacial 2D: Vídeo Sintético sob Oclusão", fontsize=11, fontweight="bold")
    ax3.set_xlabel("X (px)", fontsize=10)
    ax3.set_ylabel("Y (px)", fontsize=10)
    ax3.grid(True, linestyle="--", alpha=0.5)
    ax3.legend(fontsize=8, loc="lower left")
    
    # 4. Comparativo de Horizontes RF vs Constante
    ax4 = fig.add_subplot(gs[1, 1])
    dados_h = carregar_metricas_txt()
    if "Sintetico" in dados_h and len(dados_h["Sintetico"]["horizontes"]) > 0:
        dh = dados_h["Sintetico"]
        ax4.plot(dh["horizontes"], dh["rf"], marker="o", color="#2ca02c", label="Random Forest (Sintético)", linewidth=2)
        ax4.plot(dh["horizontes"], dh["constante"], marker="s", color="#1f77b4", label="Mov. Constante (Sintético)", linewidth=2, linestyle="--")
    if "Real" in dados_h and len(dados_h["Real"]["horizontes"]) > 0:
        dh = dados_h["Real"]
        ax4.plot(dh["horizontes"], dh["rf"], marker="^", color="#d62728", label="Random Forest (Real)", linewidth=2)
        ax4.plot(dh["horizontes"], dh["constante"], marker="d", color="#ff7f0e", label="Mov. Constante (Real)", linewidth=2, linestyle="--")
    ax4.set_title("(D) Erro de Previsão por Horizonte Futuro", fontsize=11, fontweight="bold")
    ax4.set_xlabel("Horizonte de Previsão", fontsize=10)
    ax4.set_ylabel("Erro Médio (pixels)", fontsize=10)
    ax4.grid(True, linestyle="--", alpha=0.5)
    ax4.legend(fontsize=8.5)
    
    fig.suptitle("DASHBOARD INTEGRADO DE ANÁLISE DE RASTREAMENTO E OCLUSÃO", fontsize=14, fontweight="bold", y=0.98)
    caminho_saida = "resultados/dashboard_analise_completa.png"
    plt.savefig(caminho_saida)
    plt.close()
    print(f"Salvo: {caminho_saida}")


def main():
    print("Gerando conjunto completo de gráficos visuais...")
    plotar_estados_temporal()
    plotar_erro_horizontes()
    plotar_trajetorias_oclusao()
    plotar_dinamica_velocidade()
    plotar_dashboard_geral()
    print("\nTodos os gráficos foram gerados com sucesso na pasta resultados/!")


if __name__ == "__main__":
    main()
