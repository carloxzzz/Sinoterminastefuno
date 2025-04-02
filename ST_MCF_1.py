import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import kurtosis, skew, shapiro ,norm, t
import altair as alt



st.cache_data.clear()


st.title("Calculo de Value-At-Risk y de Expected Shortfall.")

#######################################---BACKEND---##################################################


st.title("Visualización de Rendimientos de Acciones")
# st.write('hola')
@st.cache_data
def obtener_datos(stocks):
    df = yf.download(stocks, start="2010-01-01")['Close']
    return df

@st.cache_data
def calcular_rendimientos(df):
    return df.pct_change().dropna()

# Lista de acciones de ejemplo
stocks_lista = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN']


with st.spinner("Descargando datos..."):
    df_precios = obtener_datos(stocks_lista)
    df_rendimientos = calcular_rendimientos(df_precios)

    print(df_rendimientos)

#######################################---FRONTEND---##################################################

st.header("Selección de Acción")

st.text("Selecciona una acción de la lista ya que apartir de ella se calculara todo lo que se indica en cada ejercicio")

stock_seleccionado = st.selectbox("Selecciona una acción", stocks_lista)

if stock_seleccionado:
    st.subheader(f"Métricas de Rendimiento: {stock_seleccionado}")
    
    rendimiento_medio = df_rendimientos[stock_seleccionado].mean()
    Kurtosis = kurtosis(df_rendimientos[stock_seleccionado])
    skew = skew(df_rendimientos[stock_seleccionado])
    

    
    col1, col2, col3= st.columns(3)
    col1.metric("Rendimiento Medio Diario", f"{rendimiento_medio:.4%}")
    col2.metric("Kurtosis", f"{Kurtosis:.4}")
    col3.metric("Skew", f"{skew:.2}")

    #Calculo de Value-At-Risk y de Expected Shortfall (historico)

    std_dev = np.std(df_rendimientos[stock_seleccionado])

    # Definir niveles de confianza
    alphas = [0.95, 0.975, 0.99]
    resultados = []
    df_size = df_rendimientos[stock_seleccionado].size
    df_t = df_size - 1  # Grados de libertad para t-Student
    
    for alpha in alphas:
        # VaR y ES históricos
        hVaR = df_rendimientos[stock_seleccionado].quantile(1 - alpha)
        ES_hist = df_rendimientos[stock_seleccionado][df_rendimientos[stock_seleccionado] <= hVaR].mean()
        
        # VaR y ES paramétrico normal
        VaR_norm = norm.ppf(1 - alpha, rendimiento_medio, std_dev)
        ES_norm = df_rendimientos[stock_seleccionado][df_rendimientos[stock_seleccionado] <= VaR_norm].mean()
        
        # VaR y ES paramétrico t-Student
        t_ppf = t.ppf(1 - alpha, df_t)
        VaR_t = rendimiento_medio + std_dev * t_ppf * np.sqrt((df_t - 2) / df_t)
        ES_t = df_rendimientos[stock_seleccionado][df_rendimientos[stock_seleccionado] <= VaR_t].mean()
        
        # VaR y ES Monte Carlo
        simulaciones = np.random.normal(rendimiento_medio, std_dev, 10000)
        VaR_mc = np.percentile(simulaciones, (1 - alpha) * 100)
        ES_mc = df_rendimientos[stock_seleccionado][df_rendimientos[stock_seleccionado] <= VaR_mc].mean()
        
        resultados.append([alpha, hVaR, ES_hist, VaR_norm, ES_norm, VaR_t, ES_t, VaR_mc, ES_mc])
    
    # Crear DataFrame que contiene el VaR y ES para cada nivel de confianza

    df_resultados = pd.DataFrame(resultados, columns=["Alpha", "hVaR", "ES_hist", "VaR_Norm", "ES_Norm", "VaR_t", "ES_t", "VaR_MC", "ES_MC"])

    #Basicamente mostramos en patalla el dataframe antes creado

    st.subheader("Tabla comparativa de VaR y ES")
    st.text("Esta tabla muestra los resultados de los diferentes metodos de calculo de VaR y ES")

    #Mostramos el dataframe en pantalla de manera bonita
    st.dataframe(
    df_resultados.set_index("Alpha").style.format("{:.4%}")
    .applymap(lambda _: "background-color: #FFDDC1; color: black;", subset=["hVaR"])  # Durazno 
    .applymap(lambda _: "background-color: #C1E1FF; color: black;", subset=["ES_hist"])  # Azul 
    .applymap(lambda _: "background-color: #B5EAD7; color: black;", subset=["VaR_Norm"])  # Verde 
    .applymap(lambda _: "background-color: #FFB3BA; color: black;", subset=["ES_Norm"])  # Rosa 
    .applymap(lambda _: "background-color: #FFDAC1; color: black;", subset=["VaR_t"])  # Naranja 
    .applymap(lambda _: "background-color: #E2F0CB; color: black;", subset=["ES_t"])  # Verde 
    .applymap(lambda _: "background-color: #D4A5A5; color: black;", subset=["VaR_MC"])  # Rojo 
    .applymap(lambda _: "background-color: #CBAACB; color: black;", subset=["ES_MC"])  # Lila 
    )
    #Gráfico de barras para comparar los resultados
    st.subheader("Gráfico de comparación de VaR y ES")
    st.text("Este gráfico muestra la comparación de los diferentes métodos de cálculo de VaR y ES")
    st.bar_chart(df_resultados.set_index("Alpha").T)

    
    ##################################################################################################
    
    #Calculo de VaR y ES con Rolling Window

    st.subheader("Cálculo de VaR y ES con Rolling Window")

    window = 252
    alphas2 = [0.95, 0.99]

    # Cálculo de estadísticas necesarias
    rolling_mean = df_rendimientos[stock_seleccionado].rolling(window).mean()
    rolling_std = df_rendimientos[stock_seleccionado].rolling(window).std()

    # DataFrame para almacenar resultados
    var_es_rolling_df = pd.DataFrame(index=df_rendimientos.index)

    for alpha in alphas2:
        col_hVaR = f'{int(alpha * 100)}% VaR Historical'
        col_VaR_norm = f'{int(alpha * 100)}% VaR Normal'
        col_ES_hist = f'{int(alpha * 100)}% ES Historical'
        col_ES_norm = f'{int(alpha * 100)}% ES Normal'
        
        # Cálculo de VaR
        var_es_rolling_df[col_hVaR] = df_rendimientos[stock_seleccionado].rolling(window).quantile(1 - alpha)
        var_es_rolling_df[col_VaR_norm] = rolling_mean + rolling_std * norm.ppf(1 - alpha)

        # Cálculo de ES Histórico
        var_es_rolling_df[col_ES_hist] = df_rendimientos[stock_seleccionado].rolling(window).apply(
            lambda x: x[x <= x.quantile(1 - alpha)].mean(), raw=True
        )

        # Cálculo de ES Normal
        var_es_rolling_df[col_ES_norm] = rolling_mean - rolling_std * (norm.pdf(norm.ppf(1 - alpha)) / (1 - alpha))

    # Eliminamos filas con valores NaN generados por la ventana móvil
    var_es_rolling_df.dropna(inplace=True)

    # Graficamos
    plt.figure(figsize=(14, 7))

    # Gráfica de rendimientos diarios (convertidos a porcentaje)
    plt.plot(df_rendimientos.index, df_rendimientos[stock_seleccionado] * 100, label='Retornos Diarios (%)', color='blue', alpha=0.5)

    # Graficamos el VaR al 95% rolling
    plt.plot(var_es_rolling_df.index, var_es_rolling_df['95% VaR Historico'], label='95% Historico VaR', color='red', linestyle='dashed')
    plt.plot(var_es_rolling_df.index, var_es_rolling_df['95% VaR Normal'], label='95% Normal VaR', color='red')

    # Graficamos el ES al 95% rolling
    plt.plot(var_es_rolling_df.index, var_es_rolling_df['95% ES Historico'], label='95% Historico ES', color='purple', linestyle='dashed')
    plt.plot(var_es_rolling_df.index, var_es_rolling_df['95% ES Normal'], label='95% Normal ES', color='purple')

    # Agregar título y etiquetas
    plt.title('Retornos Diarios, VaR, and ES (95%) Rolling')
    plt.xlabel('Fecha')
    plt.ylabel('Cofianza (%)')

    # Mostrar leyenda
    plt.legend()
    plt.show()
