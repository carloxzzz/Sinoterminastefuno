import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import kurtosis, skew, shapiro ,norm

st.title("Calculo de Value-At-Risk y de Expected Shortfall.")

#######################################---BACKEND---##################################################


@st.cache_data
def obtener_datos(stock):
    try:
        df = yf.download(stock, period="1y")['Close']
        if df.empty:
            return None
        return df
    except Exception:
        return None

@st.cache_data
def calcular_rendimientos(df):
    return df.pct_change().dropna()
# Lista de acciones de ejemplo
stocks_lista = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN']
    

#######################################---FRONTEND---##################################################

st.header("Selección de Acción")

st.text("Selecciona una acción de la lista ya que apartir de ella se calculara todo lo que se indica en cada ejercicio")

# Meter el ticker de la acción
ticker_manual = st.text_input("Ticker de la acción:", "").strip().upper()
if ticker_manual:
    with st.spinner(f"Descargando datos de {ticker_manual}..."):
        df_precios = obtener_datos(ticker_manual)

    if df_precios is not None:
        df_rendimientos = calcular_rendimientos(df_precios)
        st.success(f"Datos descargados correctamente para {ticker_manual}.")
    else:
        st.error("Error al obtener datos. Verifica que el ticker sea correcto.")
else:
    st.warning("Por favor, ingresa un ticker para continuar.")

######## 1.-Ejercicio

st.subheader(f"Métricas de Rendimiento: {ticker_manual}")
    
rendimiento_medio = df_rendimientos.mean()
Kurtosis = kurtosis(df_rendimientos)
skew = skew(df_rendimientos)
    
col1, col2, col3= st.columns(3)
col1.metric("Rendimiento Medio Diario", f"{rendimiento_medio:.4%}")
col2.metric("Kurtosis", f"{Kurtosis:.4}")
col3.metric("Skew", f"{skew:.2}")