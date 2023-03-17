# Usage: streamlit run main.py

import streamlit as st
from transformers import pipeline
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

# Load model
model = st.selectbox('Choose model', ('Medium', 'Large'))

if model == 'Medium':
    model_name = 'multi_label_medium'
elif model == 'Large':
    model_name = 'multi_label_large'

text = st.text_area('Enter Press Release Text')

if len(text)>0:

    clf = pipeline('text-classification', model_name, return_all_scores=True)
    pred = clf(text)

    scores_df = pd.DataFrame(pred[0]).sort_values(by='score')
    scores_df['score'] = scores_df['score'].apply(lambda x: round(x, 2))

    fig = px.bar(scores_df, x='score', y='label', orientation='h', title='Classification Probabilities')
    fig.update_layout(height=2000)
    st.plotly_chart(fig)

    scores_df = scores_df.sort_values(by='score', ascending=False)
    scores_df