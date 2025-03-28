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
    df = yf.download(stocks, period="1y")['Close']
    return df

@st.cache_data
def calcular_rendimientos(df):
    return df.pct_change().dropna()

# Lista de acciones de ejemplo
#url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
#df = pd.read_html(url, header=0)[0]  # Extrae la tabla de Wikipedia
#stocks_lista = df['Symbol'].tolist()
#stocks_lista = [ticker.replace('.', '-') for ticker in df['Symbol'].tolist()]

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
    st.subheader("Resultados del Value-at-Risk (VaR) y Expected Shortfall (ES)")
    st.dataframe(df_resultados.style.format("{:.4%}").background_gradient(cmap="coolwarm"))
    st.bar_chart(df_resultados.set_index("Alpha")[["hVaR", "ES_hist", "VaR_Norm", "ES_Norm", "VaR_t", "ES_t", "VaR_MC", "ES_MC"]])

    # Convertir DataFrame a formato largo
    df_melted = df_resultados.melt(id_vars=["Alpha"], var_name="Métrica", value_name="Valor")

    # Crear histograma
    histograma = alt.Chart(df_melted).mark_bar().encode(
        x=alt.X("Valor:Q", bin=alt.Bin(maxbins=20)),  # Se agrupan los valores en bins
        y="count()",
        color="Métrica:N",  # Diferenciamos cada métrica por color
        tooltip=["Métrica", "Valor"]
    ).properties(title="Histograma de VaR y ES")

    st.altair_chart(histograma, use_container_width=True)


