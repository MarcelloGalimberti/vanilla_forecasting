# env neuralprophet
# vanilla forecaster per test deployment in streamlit cloud


import streamlit as st
import pandas as pd
from PIL import Image
import warnings
warnings.filterwarnings('ignore')
import plotly.express as px
import matplotlib.pyplot as plt
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
from prophet.diagnostics import cross_validation

from io import BytesIO
import io
from io import StringIO

#from neuralprophet import NeuralProphet

# layout
st.set_page_config(layout="wide")
st.title('Vanilla forecaster')

# input
uploaded_file = st.file_uploader('Serie temporale sell_in_subset.xlsx')
if not uploaded_file:
    st.stop()
df_dati=pd.read_excel(uploaded_file, engine='openpyxl')


st.dataframe(df_dati.head(), width=1500)

#  streamlit run streamlit_vanilla.py


# preprocessing
st.header('File caricato', divider='red')
df_dati['Totale subset'] = df_dati['Monster'].fillna(0)+df_dati['Multistrada V4']+df_dati['Panigale 7G e Panigale V4']+df_dati['Panigale V2']
st.dataframe(df_dati.head(), width=1500)

df_long = df_dati.melt(id_vars=['Data'],  # Colonne da mantenere intatte
                       value_vars=['Monster', 'Multistrada V4', 'Panigale 7G e Panigale V4',
                                   'Panigale V2', 'Totale subset'],  # Colonne da impilare
                       var_name='item_id',  # Nome della nuova colonna che conterrà i nomi delle colonne originali
                       value_name='target')

df_multistrada = df_dati[['Data','Multistrada V4']]

#grafico
fig = px.line(df_long, x='Data', y='target', color='item_id', template='plotly_dark')
st.plotly_chart(fig)

# preprocessing prophet
df_multistrada_prophet = df_multistrada.rename(columns={'Data': 'ds', 'Multistrada V4': 'y'})
st.dataframe(df_multistrada_prophet.head())

# prophet
m = Prophet(changepoint_prior_scale=0.5)
m.fit(df_multistrada_prophet)
future = m.make_future_dataframe(periods=5, freq='MS')
fcst = m.predict(future)

st.dataframe(fcst)

from prophet.plot import plot_plotly, plot_components_plotly
fig2 = plot_plotly(m, fcst, trend=True, changepoints=True)
st.plotly_chart(fig2, use_container_width=True)


st.stop()


