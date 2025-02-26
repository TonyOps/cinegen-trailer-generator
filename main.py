import os
import sys
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Concatenate, Dense
from collections import defaultdict
import traceback
from scipy.stats import gaussian_kde

# =============================================
# CONFIGURAÇÕES INICIAIS E VARIÁVEIS GLOBAIS
# =============================================
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

dados = None
modelo = None
vocab = None
idx_to_element = None
transicoes_contexto = defaultdict(lambda: defaultdict(int))
transicoes_elemento = defaultdict(lambda: defaultdict(int))
duracao_distribuicao = defaultdict(list)
contexto_inicial = defaultdict(float)
elemento_inicial = None
nv_por_elemento = defaultdict(lambda: defaultdict(int))
probabilidade_x2 = defaultdict(float)

# =============================================
# FUNÇÕES AUXILIARES
# =============================================
def verificar_dependencias():
    try:
        import scipy
        print(f"✓ SciPy {scipy.__version__} instalado")
    except ImportError:
        print("✗ SciPy não instalado! Execute:")
        print("pip install scipy")
        sys.exit(1)

def carregar_configuracoes():
    config = {
        'python_version': sys.version,
        'tensorflow_version': tf.__version__,
        'numpy_version': np.__version__,
        'caminho_script': os.path.abspath(__file__)
    }
    print("\n--- DIAGNÓSTICO INICIAL ---")
    for key, value in config.items():
        print(f"{key.replace('_', ' ').title()}: {value}")
    print("---------------------------\n")

# =============================================
# FUNÇÕES DE APOIO
# =============================================
def gerar_timestamp(duracao, ultimo_timestamp):
    horas, minutos, segundos, quadros = map(int, ultimo_timestamp.split(':'))
    total_segundos = horas * 3600 + minutos * 60 + segundos + quadros / 24.0
    total_segundos += duracao
    novas_horas = int(total_segundos // 3600)
    resto = total_segundos % 3600
    novos_minutos = int(resto // 60)
    novos_segundos = int(resto % 60)
    novos_quadros = int(round((resto - int(resto)) * 24))
    return f"{novas_horas:02d}:{novos_minutos:02d}:{novos_segundos:02d}:{novos_quadros:02d}"

def obter_duracao(elemento):
    duracoes = duracao_distribuicao.get(elemento, [3.0])
    if len(duracoes) < 5:
        media = np.mean(duracoes)
        desvio = max(np.std(duracoes), 0.1)
        return max(1.0, np.random.normal(media, desvio))
    else:
        kde = gaussian_kde(duracoes)
        return max(1.0, kde.resample(1)[0][0])

def obter_nv(elemento):
    nv_opcoes = nv_por_elemento.get(elemento, {'NV1': 1})
    nvs, contagens = zip(*nv_opcoes.items())
    total = sum(contagens)
    return np.random.choice(nvs, p=[c/total for c in contagens])

# =============================================
# FUNÇÕES PRINCIPAIS DO SISTEMA (MODIFICADAS)
# =============================================
def carregar_dados(caminho_arquivo):
    global dados, vocab, idx_to_element, transicoes_contexto, transicoes_elemento
    global duracao_distribuicao, contexto_inicial, elemento_inicial, nv_por_elemento, probabilidade_x2

    with open(caminho_arquivo, 'r', encoding='utf-8') as f:
        dados = json.load(f)
    
    if 'metadata' in dados:
        print("\nRegras carregadas:")
        print(json.dumps(dados['metadata']['regras'], indent=2, ensure_ascii=False))

    elementos = []
    contextos = []
    duracoes = []
    primeiros_elementos = []
    elementos_x2 = defaultdict(list)

    for trailer in dados['trailers']:
        cenas = trailer['cenas']
        if not cenas:
            continue

        # Processar primeira cena com elemento base
        primeira_cena = cenas[0]
        elemento_base = primeira_cena['elemento'].replace('X2', '')
        primeiros_elementos.append(elemento_base)
        contexto_inicial[primeira_cena['contexto']] += 1
        
        elemento_anterior_base = None
        for i, cena in enumerate(cenas):
            elemento_completo = cena['elemento']
            elemento = elemento_completo.replace('X2', '')
            contexto = cena['contexto']
            duracao = cena['duracao_segundos']
            nv = cena['NV']
            
            # Validar repetições inválidas
            if elemento_completo.endswith('X2') and (elemento_anterior_base != elemento):
                print(f"⚠️ Aviso: Repetição inválida na cena {i+1} - {elemento_completo} após {elemento_anterior_base or 'Nada'}")
            
            elementos.append(elemento)
            contextos.append(contexto)
            duracoes.append(duracao)
            duracao_distribuicao[elemento].append(duracao)
            nv_por_elemento[elemento][nv] += 1

            if i > 0:
                transicoes_contexto[cenas[i-1]['contexto']][contexto] += 1
            
            if elemento_anterior_base is not None:
                transicoes_elemento[elemento_anterior_base][elemento] += 1
                
                # Registrar apenas repetições válidas
                if elemento_completo.endswith('X2') and (elemento_anterior_base == elemento):
                    elementos_x2[elemento].append(1)
                else:
                    elementos_x2[elemento_anterior_base].append(0)
            
            elemento_anterior_base = elemento

    # Calcular probabilidades de repetição
    for base, ocorrencias in elementos_x2.items():
        if ocorrencias:
            probabilidade_x2[base] = sum(ocorrencias) / len(ocorrencias)

    # Normalizar contextos iniciais
    total = sum(contexto_inicial.values())
    if total > 0:
        for ctx in contexto_inicial:
            contexto_inicial[ctx] /= total

    # Criar vocabulário
    elementos_unicos = list(set(elementos))
    vocab = {elem: idx for idx, elem in enumerate(elementos_unicos)}
    idx_to_element = {idx: elem for elem, idx in vocab.items()}

    # Definir elemento inicial
    contagem_inicial = defaultdict(int)
    for elem in primeiros_elementos:
        contagem_inicial[elem] += 1
    
    if contagem_inicial:
        total = sum(contagem_inicial.values())
        elemento_inicial = np.random.choice(
            list(contagem_inicial.keys()),
            p=[v/total for v in contagem_inicial.values()]
        )
    else:
        elemento_inicial = None

    print("\nDados carregados com sucesso.")

def carregar_dados_automaticamente():
    diretorio_atual = os.path.dirname(os.path.abspath(__file__))
    arquivo_padrao = os.path.join(diretorio_atual, "banco.json")
    if os.path.exists(arquivo_padrao):
        carregar_dados(arquivo_padrao)
        print(f"Dados carregados automaticamente de: {arquivo_padrao}")
    else:
        print("Arquivo 'banco.json' não encontrado.")

def treinar_modelo():
    global modelo
    X_elemento, X_contexto, X_duracao, y = [], [], [], []
    
    for trailer in dados['trailers']:
        cenas = trailer['cenas']
        for i in range(1, len(cenas)):
            # Usar elementos base para treinamento
            elemento_anterior = cenas[i-1]['elemento'].replace('X2', '')
            elemento_atual = cenas[i]['elemento'].replace('X2', '')
            
            contexto_anterior = cenas[i-1]['contexto']
            duracao_anterior = cenas[i-1]['duracao_segundos']
            
            X_elemento.append(vocab[elemento_anterior])
            X_contexto.append(['L', 'M', 'H'].index(contexto_anterior))
            X_duracao.append(duracao_anterior)
            y.append(vocab[elemento_atual])
    
    # Converter para arrays numpy
    X_elemento = np.array(X_elemento)
    X_contexto = tf.one_hot(np.array(X_contexto), depth=3).numpy()
    X_duracao = np.array(X_duracao).reshape(-1, 1)
    y = np.array(y)
    
    # Construir modelo
    input_elemento = Input(shape=(1,))
    input_contexto = Input(shape=(3,))
    input_duracao = Input(shape=(1,))
    
    embed = Embedding(input_dim=len(vocab), output_dim=64)(input_elemento)
    flat_embed = Flatten()(embed)
    
    concat = Concatenate()([flat_embed, input_contexto, input_duracao])
    hidden = Dense(128, activation='relu')(concat)
    output = Dense(len(vocab), activation='softmax')(hidden)
    
    modelo = Model(inputs=[input_elemento, input_contexto, input_duracao], outputs=output)
    modelo.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
    
    print("\nIniciando treinamento...")
    modelo.fit([X_elemento, X_contexto, X_duracao], y, epochs=50, batch_size=32, verbose=1)
    print("Treinamento concluído com sucesso.")

def gerar_trailer(num_cenas):
    trailer = []
    if not elemento_inicial:
        print("Erro: Nenhum elemento inicial definido!")
        return trailer

    try:
        # Contexto inicial probabilistico
        contextos, probs = zip(*contexto_inicial.items())
        contexto_atual = np.random.choice(contextos, p=probs)
    except ValueError:
        print("Erro: Contextos iniciais não definidos!")
        return trailer

    elemento_atual = elemento_inicial
    duracao = obter_duracao(elemento_atual)
    timestamp = "00:00:00:00"
    
    # Primeira cena
    trailer.append({
        "timestamp": timestamp,
        "elemento": elemento_atual,
        "tipo": elemento_atual[:2],
        "contexto": contexto_atual,
        "duracao_segundos": round(duracao, 2),
        "repetido": False,
        "elemento_base": None,
        "ordem_aparicao": 1,
        "NV": obter_nv(elemento_atual)
    })

    for _ in range(num_cenas - 1):
        ultimo_elemento_base = trailer[-1]['elemento'].replace('X2', '')
        
        # Verificar probabilidade de repetição
        prob = probabilidade_x2.get(ultimo_elemento_base, 0.0)
        if np.random.rand() < prob:
            novo_elemento = ultimo_elemento_base + 'X2'
            duracao = obter_duracao(novo_elemento)
            
            trailer.append({
                "timestamp": gerar_timestamp(duracao, timestamp),
                "elemento": novo_elemento,
                "tipo": novo_elemento[:2],
                "contexto": trailer[-1]['contexto'],  # Mantém contexto
                "duracao_segundos": round(duracao, 2),
                "repetido": True,
                "elemento_base": ultimo_elemento_base,
                "ordem_aparicao": len(trailer) + 1,
                "NV": obter_nv(novo_elemento)
            })
            timestamp = trailer[-1]['timestamp']
            continue

        # Predição normal
        try:
            entrada_elemento = np.array([vocab[ultimo_elemento_base]])
            idx_ctx = ['L', 'M', 'H'].index(trailer[-1]['contexto'])
            entrada_contexto = tf.one_hot([idx_ctx], depth=3).numpy()
            entrada_duracao = np.array([trailer[-1]['duracao_segundos']]).reshape(1, 1)
            
            predicao = modelo.predict([entrada_elemento, entrada_contexto, entrada_duracao], verbose=0)
            elemento_idx = np.random.choice(len(predicao[0]), p=predicao[0])
            elemento_atual = idx_to_element[elemento_idx]
        except KeyError as e:
            print(f"Erro: Elemento não encontrado no vocabulário - {str(e)}")
            break

        # Atualizar contexto
        transicoes = transicoes_contexto[trailer[-1]['contexto']]
        total = sum(transicoes.values())
        if total > 0:
            prob_ctx = [transicoes[ctx]/total for ctx in ['L', 'M', 'H']]
            contexto_atual = np.random.choice(['L', 'M', 'H'], p=prob_ctx)
        else:
            contexto_atual = trailer[-1]['contexto']

        duracao = obter_duracao(elemento_atual)
        timestamp = gerar_timestamp(duracao, timestamp)
        
        trailer.append({
            "timestamp": timestamp,
            "elemento": elemento_atual,
            "tipo": elemento_atual[:2],
            "contexto": contexto_atual,
            "duracao_segundos": round(duracao, 2),
            "repetido": False,
            "elemento_base": None,
            "ordem_aparicao": len(trailer) + 1,
            "NV": obter_nv(elemento_atual)
        })
    
    return trailer

def salvar_resultados(trailer, nome_arquivo):
    with open(nome_arquivo, 'w', encoding='utf-8') as f:
        json.dump({"trailer_gerado": trailer}, f, indent=2, ensure_ascii=False)
    print(f"Trailer salvo em: {nome_arquivo}")

# =============================================
# MENU PRINCIPAL (ATUALIZADO)
# =============================================
def menu_principal():
    global modelo, dados
    while True:
        print("\n--- MENU PRINCIPAL ---")
        print("1. Carregar dados automaticamente")
        print("2. Carregar dados manualmente")
        print("3. Treinar modelo")
        print("4. Gerar novo trailer")
        print("5. Sair")
        
        opcao = input("Escolha: ").strip()
        
        if opcao == "1":
            carregar_dados_automaticamente()
        elif opcao == "2":
            caminho = input("Caminho completo do JSON: ").strip()
            try:
                carregar_dados(caminho)
            except Exception as e:
                print(f"Erro: {str(e)}")
        elif opcao == "3":
            if dados is None:
                print("Erro: Carregue os dados primeiro!")
            else:
                treinar_modelo()
        elif opcao == "4":
            if modelo is None:
                print("Erro: Treine o modelo primeiro!")
                continue
            try:
                num = int(input("Número de cenas (5-100): "))
                if not 5 <= num <= 100:
                    raise ValueError
                trailer = gerar_trailer(num)
                print("\n=== TRAILER GERADO ===")
                for cena in trailer:
                    dur = cena['duracao_segundos']
                    seg = int(dur)
                    ms = int((dur - seg) * 1000)
                    print(f"{cena['timestamp']} - {cena['elemento']} ({cena['contexto']} - {dur:.2f}s [{seg}s {ms:03d}ms] | Ordem: {cena['ordem_aparicao']} | NV: {cena['NV']}")
                if input("\nSalvar? (s/n): ").lower() == 's':
                    nome = input("Nome do arquivo: ").strip()
                    salvar_resultados(trailer, nome)
            except Exception as e:
                print(f"Erro na geração: {str(e)}")
        elif opcao == "5":
            print("Encerrando...")
            break
        else:
            print("Opção inválida!")

# =============================================
# INICIALIZAÇÃO
# =============================================
if __name__ == "__main__":
    try:
        verificar_dependencias()
        carregar_configuracoes()
        print("\n=== APLICAÇÃO INICIADA ===")
        menu_principal()
    except Exception as e:
        print("\n!!! ERRO INICIAL !!!")
        traceback.print_exc()
    finally:
        input("\nPressione Enter para sair...")