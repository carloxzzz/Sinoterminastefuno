import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import kurtosis, skew, shapiro ,norm


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
url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
df = pd.read_html(url, header=0)[0]  # Extrae la tabla de Wikipedia
stocks_lista = df['Symbol'].tolist()
stocks_lista = [ticker.replace('.', '-') for ticker in df['Symbol'].tolist()]


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

        # Gráfico de rendimientos diarios
    st.subheader(f"Gráfico de Rendimientos: {stock_seleccionado}")
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(df_rendimientos.index, df_rendimientos[stock_seleccionado], label=stock_seleccionado)
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.7)
    ax.legend()
    ax.set_title(f"Rendimientos de {stock_seleccionado}")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Rendimiento Diario")
    st.pyplot(fig)
    