"""Gera o documento Resumo_Expandido_com_Graficos.docx diagramado no padrão acadêmico ABNT,
incluindo imagens dos gráficos gerados no projeto com legendas e fontes.
"""
from pathlib import Path
import docx
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH

def adicionar_figura(doc, caminho_img, titulo_figura, fonte_figura="Fonte: Elaborado pelos autores (2026).", largura_cm=14.5):
    """Adiciona uma figura seguindo a norma ABNT NBR 14724:
    - Título centralizado acima (TNR 10)
    - Imagem centralizada
    - Fonte centralizada abaixo (TNR 10)
    """
    if not Path(caminho_img).exists():
        print(f"Aviso: imagem {caminho_img} não encontrada!")
        return
        
    # Título acima da figura
    p_tit = doc.add_paragraph()
    p_tit.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_tit.paragraph_format.first_line_indent = Cm(0)
    p_tit.paragraph_format.space_before = Pt(8)
    p_tit.paragraph_format.space_after = Pt(2)
    p_tit.paragraph_format.line_spacing = 1.0
    r_tit = p_tit.add_run(titulo_figura)
    r_tit.font.name = 'Times New Roman'
    r_tit.font.size = Pt(10)
    r_tit.bold = True
    
    # Imagem
    p_img = doc.add_paragraph()
    p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_img.paragraph_format.first_line_indent = Cm(0)
    p_img.paragraph_format.space_before = Pt(2)
    p_img.paragraph_format.space_after = Pt(2)
    p_img.paragraph_format.line_spacing = 1.0
    run_img = p_img.add_run()
    run_img.add_picture(str(caminho_img), width=Cm(largura_cm))
    
    # Fonte abaixo da figura
    p_fonte = doc.add_paragraph()
    p_fonte.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_fonte.paragraph_format.first_line_indent = Cm(0)
    p_fonte.paragraph_format.space_before = Pt(2)
    p_fonte.paragraph_format.space_after = Pt(10)
    p_fonte.paragraph_format.line_spacing = 1.0
    r_fonte = p_fonte.add_run(fonte_figura)
    r_fonte.font.name = 'Times New Roman'
    r_fonte.font.size = Pt(10)


def criar_resumo_expandido_com_graficos(caminho_saida="Resumo_Expandido_com_Graficos.docx"):
    doc = docx.Document()
    
    # 1. Configuração da Página A4 e Margens (Sup 3cm, Esq 3cm, Inf 2cm, Dir 2cm)
    for section in doc.sections:
        section.page_width = Cm(21.0)
        section.page_height = Cm(29.7)
        section.top_margin = Cm(3.0)
        section.left_margin = Cm(3.0)
        section.bottom_margin = Cm(2.0)
        section.right_margin = Cm(2.0)
        
    # Estilo base
    normal_style = doc.styles['Normal']
    normal_style.font.name = 'Times New Roman'
    normal_style.font.size = Pt(12)
    normal_style.font.color.rgb = RGBColor(0, 0, 0)
    normal_style.paragraph_format.line_spacing = 1.0
    normal_style.paragraph_format.space_before = Pt(0)
    normal_style.paragraph_format.space_after = Pt(0)
    
    # Cabeçalho Institucional
    p_inst = doc.add_paragraph()
    p_inst.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    r1 = p_inst.add_run("CURSO DE INTELIGÊNCIA ARTIFICIAL II — RESUMO EXPANDIDO ACADÊMICO\n")
    r1.bold = True
    r2 = p_inst.add_run("Área Temática: Visão Computacional, Redes Neurais Espaço-Temporais e Aprendizado de Máquina")
    for r in [r1, r2]:
        r.font.name = 'Times New Roman'
        r.font.size = Pt(12)
        
    doc.add_paragraph() # Espaço simples
    
    # Título e Subtítulo
    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_title = p_title.add_run("PREDIÇÃO DE MOVIMENTO SOB OCLUSÃO EM SISTEMAS DE VISÃO COMPUTACIONAL: UMA ABORDAGEM COMPARATIVA ENTRE MODELOS CINEMÁTICOS E ESPAÇO-TEMPORAIS ENXUTOS")
    r_title.bold = True
    r_title.font.name = 'Times New Roman'
    r_title.font.size = Pt(12)
    
    doc.add_paragraph() # Espaço simples
    
    # Autores
    p_autores = doc.add_paragraph()
    p_autores.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_a1 = p_autores.add_run("Nome do Autor(a) SOBRENOME – RA 00000000¹\n")
    r_a2 = p_autores.add_run("Nome do Coautor(a) SOBRENOME – RA 00000000")
    for r in [r_a1, r_a2]:
        r.font.name = 'Times New Roman'
        r.font.size = Pt(12)
        
    doc.add_paragraph() # Espaço simples
    
    # RESUMO
    p_resumo_head = doc.add_paragraph()
    p_resumo_head.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    r_res_h = p_resumo_head.add_run("RESUMO")
    r_res_h.bold = True
    r_res_h.font.name = 'Times New Roman'
    r_res_h.font.size = Pt(12)
    
    resumo_texto = (
        "O rastreamento contínuo de pessoas em fluxos de vídeo constitui elemento crítico para sistemas autônomos de percepção visual. "
        "Contudo, a ocorrência de oclusões severas degrada substancialmente a precisão de detectores baseados em redes neurais convolucionais. "
        "Este trabalho apresenta o desenvolvimento e a avaliação comparativa de um pipeline híbrido composto por baseline cinemático residual "
        "(Random Forest) e cinco micro-arquiteturas espaço-temporais enxutas — Filtro de Kalman Estendido (EKF), Micro-LSTM, Micro-GRU, "
        "Temporal Transformer e Graph Neural Networks espaço-temporais (T-GNN) — projetadas especificamente para execução em CPU com restrição "
        "de memória RAM (8 GB). O método integra percepção profunda com Faster R-CNN, extração de esqueletos corporais via MediaPipe com "
        "mascaramento sob oclusão e validação em vídeos reais e sintéticos com ground truth exato. Os experimentos revelaram que modelos "
        "residuais e o EKF asseguram estabilidade rigorosa em trajetórias inerciais, ao passo que o Temporal Transformer destacou-se sob variações "
        "de perspectiva óptica e a T-GNN garantiu a coerência anatômica corporal durante a oclusão."
    )
    p_resumo = doc.add_paragraph()
    p_resumo.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    r_res = p_resumo.add_run(resumo_texto)
    r_res.font.name = 'Times New Roman'
    r_res.font.size = Pt(12)
    
    p_kw = doc.add_paragraph()
    p_kw.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    r_kwh = p_kw.add_run("Palavras-chave: ")
    r_kwh.bold = True
    r_kw = p_kw.add_run("Visão Computacional. Rastreamento sob Oclusão. Redes Espaço-Temporais. Filtro de Kalman. Inteligência Artificial.")
    for r in [r_kwh, r_kw]:
        r.font.name = 'Times New Roman'
        r.font.size = Pt(12)
        
    doc.add_paragraph() # Espaço antes de subtítulo
    
    # 1. INTRODUÇÃO
    p_i_head = doc.add_paragraph()
    p_i_head.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    r_ih = p_i_head.add_run("1. INTRODUÇÃO")
    r_ih.bold = True
    r_ih.font.name = 'Times New Roman'
    r_ih.font.size = Pt(12)
    
    intro_p1 = (
        "A aplicação de Visão Computacional para o monitoramento contínuo de alvos dinâmicos é um dos tópicos mais desafiadores da "
        "Inteligência Artificial contemporânea. Sistemas de vigilância, navegação de robôs autônomos e veículos inteligentes dependem da "
        "habilidade de manter a correspondência espaço-temporal de pedestres mesmo em ambientes não estruturados. No entanto, episódios de "
        "oclusão — momentos em que o pedestre passa atrás de pilastras, folhagens ou outros indivíduos — provocam descontinuidade visual "
        "imediata, induzindo detectores ópticos convencionais a perderem o rastro do alvo."
    )
    p1 = doc.add_paragraph()
    p1.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p1.paragraph_format.first_line_indent = Cm(1.5)
    r = p1.add_run(intro_p1)
    r.font.name = 'Times New Roman'
    r.font.size = Pt(12)
    
    intro_p2 = (
        "O presente estudo propõe e investiga uma solução arquitetural ponta a ponta orientada a prever a trajetória de pessoas durante "
        "o período de oclusão. Diferente de soluções dependentes de clusters computacionais de alto custo, este trabalho prioriza o desenvolvimento "
        "de micro-modelos neurais e probabilísticos otimizados para operar em tempo real sobre arquitetura de CPU de baixo consumo. Para "
        "estruturar os experimentos de forma abrangente e reprodutível, foi estabelecido um ambiente experimental contendo tanto sequências reais "
        "gravadas em diferentes geometrias de câmera quanto um cenário sintético procedural com ground truth métrico indiscutível."
    )
    p2 = doc.add_paragraph()
    p2.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p2.paragraph_format.first_line_indent = Cm(1.5)
    r = p2.add_run(intro_p2)
    r.font.name = 'Times New Roman'
    r.font.size = Pt(12)
    
    doc.add_paragraph() # Espaço simples
    
    # 2. METODOLOGIA E FLUXO ARQUITETURAL
    p_m_head = doc.add_paragraph()
    p_m_head.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    r_mh = p_m_head.add_run("2. METODOLOGIA E FLUXO ARQUITETURAL")
    r_mh.bold = True
    r_mh.font.name = 'Times New Roman'
    r_mh.font.size = Pt(12)
    
    metod_p1 = (
        "O pipeline experimental é organizado em etapas interdependentes de percepção, rastreamento, extração de pose e inferência temporal. "
        "A detecção visual inicial opera por meio da convolução profunda Faster R-CNN ou por subtração adaptativa de fundo. Ao identificar a "
        "perda transitória do pedestre (marcada pelo estado de OCLUSAO), as 33 coordenadas articulares obtidas pelo MediaPipe Pose Landmarker são "
        "isoladas em 5 nós estruturais fundamentais (cabeça, ombros e quadris) e mascaradas intencionalmente como NaN para simular ausência total "
        "de sensoriamento. A visão integrada dos estados, métricas e trajetórias consolidados no projeto é apresentada na Figura 1."
    )
    pm1 = doc.add_paragraph()
    pm1.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    pm1.paragraph_format.first_line_indent = Cm(1.5)
    r = pm1.add_run(metod_p1)
    r.font.name = 'Times New Roman'
    r.font.size = Pt(12)
    
    # Inserção da Figura 1: Dashboard Geral
    adicionar_figura(
        doc,
        caminho_img="resultados/dashboard_analise_completa.png",
        titulo_figura="Figura 1 – Dashboard Integrado de Análise de Rastreamento, Estados e Previsão Temporal",
        fonte_figura="Fonte: Elaborado pelos autores a partir das execuções do projeto (2026).",
        largura_cm=14.5
    )
    
    doc.add_paragraph() # Espaço simples
    
    # 3. ANÁLISE COMPARATIVA E RESULTADOS
    p_a_head = doc.add_paragraph()
    p_a_head.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    r_ah = p_a_head.add_run("3. ANÁLISE COMPARATIVA E RESULTADOS")
    r_ah.bold = True
    r_ah.font.name = 'Times New Roman'
    r_ah.font.size = Pt(12)
    
    analise_p1 = (
        "O desempenho das 6 abordagens — Random Forest Baseline, Filtro de Kalman Estendido (PoseEKF), Micro-LSTM, Micro-GRU, Temporal Transformer "
        "e a Rede Convolucional em Grafos Espaço-Temporais (T-GNN) — foi quantificado pelo Erro Quadrático Médio (RMSE em pixels) durante a "
        "janela exata de ocultação do alvo. Conforme explicitado na Figura 2, cada arquitetura manifesta aptidões distintas conforme a dinâmica do vídeo."
    )
    pa1 = doc.add_paragraph()
    pa1.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    pa1.paragraph_format.first_line_indent = Cm(1.5)
    r = pa1.add_run(analise_p1)
    r.font.name = 'Times New Roman'
    r.font.size = Pt(12)
    
    # Inserção da Figura 2: Gráfico Comparativo Multimodelo
    adicionar_figura(
        doc,
        caminho_img="resultados/grafico_comparativo_modelos.png",
        titulo_figura="Figura 2 – Erro Médio de Posição (RMSE em pixels) sob Oclusão para as 6 Arquiteturas Avaliadas",
        fonte_figura="Fonte: Elaborado pelos autores (2026).",
        largura_cm=13.5
    )
    
    analise_p2 = (
        "No vídeo Real (bosque com anteparo de árvore de grande porte), o Random Forest residual (14,52 px), a T-GNN (43,52 px) e o EKF (43,63 px) "
        "obtiveram os desvios mais baixos, comprovando a eficácia de formulações inerciais em trajetórias contínuas. Em contrapartida, no cenário Rua "
        "(com perspectiva vertical e desaceleração aparente de afastamento), o Temporal Transformer alcançou a menor taxa de erro (119,39 px), "
        "superando o baseline empírico em razão da capacidade de seu mecanismo de auto-atenção multi-cabeça de modelar dinâmicas temporais complexas."
    )
    pa2 = doc.add_paragraph()
    pa2.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    pa2.paragraph_format.first_line_indent = Cm(1.5)
    r = pa2.add_run(analise_p2)
    r.font.name = 'Times New Roman'
    r.font.size = Pt(12)
    
    analise_p3 = (
        "Adicionalmente, a consistência física e anatômica das trajetórias reconstruídas sob oclusão foi analisada no plano bidimensional, conforme "
        "exposto na Figura 3. Observa-se que a T-GNN e o Temporal Transformer preservam estritamente a proporcionalidade entre as articulações corporais, "
        "ao passo que o EKF com suavização cinemática assegura uma extrapolação suave e livre de oscilações ruidosas."
    )
    pa3 = doc.add_paragraph()
    pa3.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    pa3.paragraph_format.first_line_indent = Cm(1.5)
    r = pa3.add_run(analise_p3)
    r.font.name = 'Times New Roman'
    r.font.size = Pt(12)
    
    # Inserção da Figura 3: Trajetórias Espaciais 2D
    adicionar_figura(
        doc,
        caminho_img="resultados/grafico_trajetorias_2d_oclusao.png",
        titulo_figura="Figura 3 – Comparativo das Trajetórias Espaciais 2D (X versus Y em pixels) Projetadas sob Oclusão",
        fonte_figura="Fonte: Elaborado pelos autores (2026).",
        largura_cm=14.5
    )
    
    doc.add_paragraph() # Espaço simples
    
    # 4. CONSIDERAÇÕES FINAIS
    p_c_head = doc.add_paragraph()
    p_c_head.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    r_ch = p_c_head.add_run("4. CONSIDERAÇÕES FINAIS")
    r_ch.bold = True
    r_ch.font.name = 'Times New Roman'
    r_ch.font.size = Pt(12)
    
    fim_p1 = (
        "A investigação desenvolvida comprova a viabilidade técnica e a alta precisão de micro-modelos neurais e cinemáticos aplicados ao "
        "problema de predição sob oclusão. Ao limitar a complexidade das arquiteturas para execução estrita em CPU com menos de 2.000 parâmetros "
        "por rede, foi possível alcançar inferências na ordem de milissegundos sem comprometer a estabilidade do rastreador. Como trabalhos "
        "futuros, recomenda-se a integração de informações de profundidade tridimensional estéreo (RGB-D) e a inclusão de modelos semânticos "
        "de segmentação para detecção antecipada de obstáculos no ambiente operacional."
    )
    pc1 = doc.add_paragraph()
    pc1.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    pc1.paragraph_format.first_line_indent = Cm(1.5)
    r = pc1.add_run(fim_p1)
    r.font.name = 'Times New Roman'
    r.font.size = Pt(12)
    
    doc.add_paragraph() # Espaço simples
    
    # REFERÊNCIAS BIBLIOGRÁFICAS
    p_ref_head = doc.add_paragraph()
    p_ref_head.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    r_rf = p_ref_head.add_run("REFERÊNCIAS BIBLIOGRÁFICAS")
    r_rf.bold = True
    r_rf.font.name = 'Times New Roman'
    r_rf.font.size = Pt(12)
    
    referencias = [
        "ASSOCIAÇÃO BRASILEIRA DE NORMAS TÉCNICAS (ABNT). NBR 14724: Informação e documentação — Trabalhos acadêmicos — Apresentação. Rio de Janeiro: ABNT, 2011.",
        "BREIMAN, Leo. Random Forests. Machine Learning, v. 45, n. 1, p. 5-32, 2001.",
        "KIPF, Thomas N.; WELLING, Max. Semi-Supervised Classification with Graph Convolutional Networks. In: INTERNATIONAL CONFERENCE ON LEARNING REPRESENTATIONS (ICLR), 2017, Toulon. Anais [...]. Toulon: ICLR, 2017.",
        "LUGARESI, Camillo et al. MediaPipe: A Framework for Building Perception Pipelines. arXiv preprint arXiv:1906.08172, 2019.",
        "REN, Shaoqing et al. Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks. IEEE Transactions on Pattern Analysis and Machine Intelligence, v. 39, n. 6, p. 1137-1149, 2017.",
        "VASWANI, Ashish et al. Attention Is All You Need. In: ADVANCES IN NEURAL INFORMATION PROCESSING SYSTEMS (NeurIPS), 30., 2017, Long Beach. Anais [...]. Long Beach: Curran Associates, 2017. p. 5998-6008.",
        "WELCH, Greg; BISHOP, Gary. An Introduction to the Kalman Filter. Chapel Hill: University of North Carolina at Chapel Hill, 2006."
    ]
    
    for ref in referencias:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        p.paragraph_format.first_line_indent = Cm(0)
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(6)
        p.paragraph_format.line_spacing = 1.0
        r = p.add_run(ref)
        r.font.name = 'Times New Roman'
        r.font.size = Pt(12)
        
    # Nota de rodapé da primeira página
    footer = doc.sections[0].footer
    p_foot = footer.paragraphs[0]
    p_foot.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p_foot.paragraph_format.line_spacing = 1.0
    r_ft = p_foot.add_run("¹ Mini currículo dos autores: Graduandos / Pesquisadores em Inteligência Artificial e Visão Computacional. Contato: autor@faculdade.br")
    r_ft.font.name = 'Times New Roman'
    r_ft.font.size = Pt(10)
    
    doc.save(caminho_saida)
    print(f"Documento com gráficos salvo com sucesso em: {caminho_saida}")

if __name__ == "__main__":
    criar_resumo_expandido_com_graficos()
