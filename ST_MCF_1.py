import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import kurtosis, skew, shapiro ,norm

st.title("Calculo de Value-At-Risk y de Expected Shortfall.")

st.header("1.-Ejercicio")

st.subheader("1.1.-Descarga y carga de datos")