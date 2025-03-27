import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import kurtosis, skew, shapiro ,norm

st.title("Calculo de Value-At-Risk y de Expected Shortfall.")

######################################---BACKEND---##################################################

@st.cache_data
def obtener_datos(stock):
    df = yf.download(stock, period="1y")['Close']
    return df.to_frame(name=stock)  # Convertimos a DataFrame con nombre de columna explícito

@st.cache_data
def calcular_rendimientos(df):
    return df.pct_change().dropna()

#######################################---FRONTEND---##################################################

st.header("Selección de Acción")

st.text("Ingresa el símbolo de la acción para calcular métricas de rendimiento:")

# Entrada de usuario para cualquier acción
stock_seleccionado = st.text_input("Símbolo de la acción", "AAPL").upper()

if stock_seleccionado:
    try:
        df_precios = obtener_datos(stock_seleccionado)
        df_rendimientos = calcular_rendimientos(df_precios)

        if not df_rendimientos.empty:
            ######## 1.-Ejercicio
            st.subheader(f"Métricas de Rendimiento: {stock_seleccionado}")

            rendimiento_medio = df_rendimientos.mean().iloc[0]  # Extraemos el valor numérico
            Kurtosis = kurtosis(df_rendimientos).iloc[0]
            Skew = skew(df_rendimientos).iloc[0]

            col1, col2, col3 = st.columns(3)
            col1.metric("Rendimiento Medio Diario", f"{rendimiento_medio:.4%}")
            col2.metric("Kurtosis", f"{Kurtosis:.4}")
            col3.metric("Skew", f"{Skew:.2}")

        else:
            st.error("No se pudieron obtener datos de rendimiento para la acción ingresada.")
    except Exception as e:
        st.error(f"Error al obtener los datos: {e}")