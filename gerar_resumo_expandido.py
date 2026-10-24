"""Gera o documento Resumo_Expandido_Modelo_Rio_Branco.docx estritamente formatado
de acordo com o template da XVIII Semana de Relações Internacionais das Faculdades Integradas Rio Branco.
"""
import docx
from docx.shared import Inches, Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.style import WD_STYLE_TYPE
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

def criar_resumo_expandido(caminho_saida="Resumo_Expandido_Modelo_Rio_Branco.docx"):
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
    r1 = p_inst.add_run("Faculdades Integradas Rio Branco\n")
    r1.bold = True
    r2 = p_inst.add_run("XVIII Semana de Relações Internacionais\n")
    r3 = p_inst.add_run("Área Temática: Segurança Internacional, Tecnologias Críticas e Defesa")
    for r in [r1, r2, r3]:
        r.font.name = 'Times New Roman'
        r.font.size = Pt(12)
        
    doc.add_paragraph() # Um espaço simples antes do título
    
    # Título e Subtítulo
    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_title = p_title.add_run("PREDIÇÃO DE MOVIMENTO SOB OCLUSÃO EM SISTEMAS DE VISÃO COMPUTACIONAL: UMA ABORDAGEM COMPARATIVA ENTRE MODELOS CINEMÁTICOS E ESPAÇO-TEMPORAIS ENXUTOS")
    r_title.bold = True
    r_title.font.name = 'Times New Roman'
    r_title.font.size = Pt(12)
    
    doc.add_paragraph() # Um espaço simples
    
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
    
    # Texto do Resumo (máx 200 palavras)
    resumo_texto = (
        "O rastreamento contínuo de pessoas em fluxos de vídeo constitui elemento crítico para sistemas de monitoramento "
        "inteligente e segurança. Contudo, a ocorrência de oclusões parciais ou totais frequentemente degrada a precisão de sensores "
        "ópticos convencionais. Este trabalho tem como objetivo comparar métodos cinemáticos clássicos e cinco micro-arquiteturas "
        "espaço-temporais enxutas — Filtro de Kalman Estendido (EKF), Micro-LSTM, Micro-GRU, Temporal Transformer e Graph Neural "
        "Networks espaço-temporais (T-GNN) — projetadas especificamente para execução em unidades centrais de processamento (CPU). "
        "A metodologia empregou percepção via convolução profunda (Faster R-CNN), extração de coordenadas esqueléticas centrais "
        "com MediaPipe, mascaramento deliberado de dados sob oclusão e modelagem de movimento walk-forward. Foram avaliados vídeos com "
        "trajetórias lineares sob controle paramétrico com ground truth exato e gravações reais em ambientes complexos. Os resultados "
        "demonstraram que o modelo Random Forest residual e o EKF garantem estabilidade estrita em movimentos inerciais, ao passo que "
        "o Temporal Transformer superou os demais sob variações de perspectiva visual. Conclui-se que arquiteturas espaço-temporais "
        "leves viabilizam predição acurada em hardware de baixo custo, mitigando falhas de observabilidade em vigilância autônoma."
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
        
    doc.add_paragraph() # Espaço antes do subtítulo
    
    # Seções
    def adicionar_secao(titulo, paragrafos):
        p_tit = doc.add_paragraph()
        p_tit.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        r_t = p_tit.add_run(titulo)
        r_t.bold = True
        r_t.font.name = 'Times New Roman'
        r_t.font.size = Pt(12)
        
        for par in paragrafos:
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            p.paragraph_format.first_line_indent = Cm(1.5)
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.line_spacing = 1.0
            r = p.add_run(par)
            r.font.name = 'Times New Roman'
            r.font.size = Pt(12)
            
        doc.add_paragraph() # Espaço antes do próximo subtítulo

    # 1. INTRODUÇÃO
    intro_p1 = (
        "A evolução acelerada da inteligência artificial aplicada ao processamento de sinais visuais redefiniu as capacidades "
        "operacionais de monitoramento e vigilância de infraestruturas estratégicas no cenário internacional. No entanto, a eficácia de "
        "sensores computacionais enfrenta desafios severos diante de perturbações ambientais, dentre as quais se destacam as oclusões "
        "ópticas, episódios em que alvos de interesse deixam temporariamente de ser observados devido à interferência de obstáculos físicos, "
        "aglomerações ou vegetação densa. Na ausência de mecanismos preditivos de extrapolação temporal, sistemas autônomos tendem a "
        "descartar a identidade do pedestre ou gerar alarmes inconclusivos."
    )
    intro_p2 = (
        "Diante desse panorama, o presente estudo investiga e desenvolve um arcabouço metodológico capaz de reconstituir a trajetória "
        "humana durante intervalos cegos, articulando modelos cinemáticos de engenharia e redes neurais profundas de alta eficiência. O "
        "objetivo central é avaliar o compromisso entre acurácia de posicionamento e viabilidade computacional sob restrições estritas de "
        "memória e processamento (CPU). Para tanto, fundamentou-se a pesquisa na extração de representações visuais com convoluções residuais, "
        "mapeamento esquelético das articulações corporais e predição autorregressiva por meio de arquiteturas recorrentes, atencionais "
        "e estruturadas em grafos anatômicos."
    )
    adicionar_secao("1. INTRODUÇÃO", [intro_p1, intro_p2])
    
    # 2. ANÁLISE E COMENTÁRIO DO CONTEÚDO
    analise_p1 = (
        "O pipeline experimental foi estruturado a partir da integração entre percepção visual, detecção heurística de perda de sinal "
        "e modelagem preditiva. Inicialmente, um módulo de extração baseado em Faster R-CNN e subtração adaptativa de fundo identifica as "
        "coordenadas do indivíduo. Em seguida, o framework MediaPipe isola cinco articulações anatômicas essenciais — cabeça, ombros e quadris —, "
        "cujas coordenadas são mascaradas intencionalmente durante os quadros rotulados como oclusão. Nesse intervalo cego, foram submetidos a "
        "teste cinco micro-modelos: o Filtro de Kalman Estendido (EKF), redes Micro-LSTM e Micro-GRU, um Micro-Temporal Transformer com atenção "
        "multi-cabeça e uma Rede Neural Convolucional em Grafos Espaço-Temporais (T-GNN)."
    )
    analise_p2 = (
        "A mensuração do erro quadrático médio de translação (RMSE em pixels) evidenciou comportamentos complementares entre as técnicas. "
        "Em cenários de dinâmica linear e deslocamento inercial constante, como em trechos de bosque com anteparo arbóreo, o modelo baseline de "
        "Random Forest residual alcançou erro mínimo de 14,52 pixels, seguido pelo EKF com 43,63 pixels. A formulação estatística do EKF assegurou "
        "continuidade de movimento sem sofrer do sobreajuste decorrente do curto histórico pré-oclusão. Por outro lado, em ambiente urbano com "
        "forte distorção de perspectiva óptica e afastamento em profundidade, o Temporal Transformer destacou-se com o melhor desempenho (119,39 "
        "pixels), superando o baseline empírico em virtude de sua capacidade de ponderar atenções em desacelerações aparentes. Paralelamente, a "
        "T-GNN comprovou sua robustez ao preservar a coerência das proporções físicas do corpo humano, impedindo dispersões anatômicas aleatórias."
    )
    adicionar_secao("2. ANÁLISE E COMENTÁRIO DO CONTEÚDO", [analise_p1, analise_p2])
    
    # 3. CONSIDERAÇÕES FINAIS
    fim_p1 = (
        "Os resultados obtidos demonstram que a incorporação de modelagem espaço-temporal enxuta constitui uma resposta eficaz às "
        "vulnerabilidades de sistemas de monitoramento visual sujeitos a interrupções de visibilidade. A substituição de extrapolações cinemáticas "
        "rígidas por mecanismos inteligentes de predição viabiliza a manutenção da continuidade de rastreio com baixo custo computacional. "
        "Como desdobramentos e recomendações para trabalhos futuros, sugere-se a incorporação de mapas tridimensionais de profundidade e a "
        "fusão de sensores multimodais, ampliando a resiliência de sistemas autônomos em cenários de alta complexidade operacional."
    )
    adicionar_secao("3. CONSIDERAÇÕES FINAIS", [fim_p1])
    
    # REFERÊNCIAS BIBLIOGRÁFICAS
    p_ref_head = doc.add_paragraph()
    p_ref_head.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    r_rf = p_ref_head.add_run("REFERÊNCIAS BIBLIOGRÁFICAS")
    r_rf.bold = True
    r_rf.font.name = 'Times New Roman'
    r_rf.font.size = Pt(12)
    
    referencias = [
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
        
    # Adicionar Nota de Rodapé no rodapé da primeira página ou seção
    # No Word, notas de rodapé de autor podem ser adicionadas no rodapé da página inicial
    footer = doc.sections[0].footer
    p_foot = footer.paragraphs[0]
    p_foot.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p_foot.paragraph_format.line_spacing = 1.0
    r_ft = p_foot.add_run("¹ Mini currículo dos autores: Graduandos / Pesquisadores em Inteligência Artificial e Tecnologia. Contato: autor@faculdade.br")
    r_ft.font.name = 'Times New Roman'
    r_ft.font.size = Pt(10)
    
    doc.save(caminho_saida)
    print(f"Documento salvo com sucesso em: {caminho_saida}")

if __name__ == "__main__":
    criar_resumo_expandido()
