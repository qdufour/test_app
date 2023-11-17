# -*- coding: utf-8 -*-

import joblib
import pandas as pd
import numpy as np
import shap
import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import requests
st.set_option('deprecation.showPyplotGlobalUse', False)

def fetch_scores(client_data):
    api_url = "http://127.0.0.1:5000/predict"
    # Convertir le DataFrame en JSON orienté 'split'
    client_data_json = client_data.to_json(orient='split')
    response = requests.post(api_url, json={'client_data': client_data_json})
    return response.json()['scores']

def main():
    st.title("Application de Visualisation des Scores Clients")
    model = joblib.load('model.pkl')
    df = pd.read_csv('data_test.csv')
    explainer = shap.Explainer(model, df)
    
    # Initialize session state
    if 'prev_selected_client' not in st.session_state:
        st.session_state.prev_selected_client = None
        st.session_state.prev_scores = None
    
    # Sidebar for user input
    st.sidebar.header("Client ID")
    id_client = st.sidebar.selectbox("Sélection du client", df["SK_ID_CURR"])
    
    if st.session_state.prev_selected_client != id_client:
        scores_clients = fetch_scores(df)
        st.session_state.prev_selected_client = id_client
        st.session_state.prev_scores = scores_clients
    else :
        scores_clients = st.session_state.prev_scores
        st.session_state.prev_selected_client = id_client
        st.session_state.prev_scores = scores_clients
        
    example_idx = np.searchsorted(df["SK_ID_CURR"], id_client)
    instance = df[example_idx:example_idx+1]    
        
    shap_values = explainer.shap_values(instance)
    score = scores_clients[example_idx]
    
    # Titre de l'application
    st.title("Tableau de Bord Client")
    
    # Afficher un tableau avec les informations des clients
    st.write("Informations du client :")
    st.dataframe(instance)
    
    # Gauge Score
    threshold = 0.5
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Score", 'font': {'size': 24}},
        delta = {'reference': threshold, 'increasing': {'color': "RebeccaPurple"}},
        gauge = {
            'axis': {'range': [None, 1], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, threshold], 'color': 'cyan'},
                {'range': [threshold, 1], 'color': 'royalblue'}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': score}}))
    
    fig.update_layout(paper_bgcolor = "lavender", font = {'color': "darkblue", 'family': "Arial"})
    st.write(fig)
    
    # Plot SHAP summary
    st.write("Feature Importance Locale")
    feature_importances = np.abs(shap_values).mean(axis=0)
    st.pyplot(shap.summary_plot(shap_values, df, feature_names=df.columns))
    
    # Feature Importance Globale
    st.title("Feature Importance du Modèle")
    feature_importances = model.feature_importances_
    top_feature_indices = feature_importances.argsort()[-20:][::-1]
    top_feature_names = [df.columns[i] for i in top_feature_indices]
    top_feature_importances = feature_importances[top_feature_indices]
    plt.figure(figsize=(10, 8))
    plt.xticks(rotation=90)
    plt.barh(top_feature_names, top_feature_importances)
    plt.xlabel("Feature Importance")
    plt.ylabel("Features")
    plt.title("Top 20 Feature Importance")
    st.pyplot(plt)
    
    # Distribution d'une Feature selon les classes
    st.title("Analyse Univariée")
    selected_feature = st.selectbox("Sélectionnez la Feature :", df.columns)
    subset_data = df[selected_feature]
    client_feature = df[selected_feature][example_idx]
    
    fig = px.histogram(subset_data, nbins=100, title="Histogramme de la Feature", labels={"value": "Data Value"})
    fig.add_annotation(
        x=client_feature,
        y=10,  # Adjust the vertical position of the annotation
        text=f"Client: {client_feature}",
        showarrow=True,
        arrowhead=4,
        ax=-20,  # Adjust the arrowhead position
        ay=30,  # Adjust the arrowhead position
    )
    st.plotly_chart(fig)
    
    # Analyse Bivariée entre 2 Features
    st.title("Analyse Bivariée")
    
    selected_feature_1 = st.selectbox("Sélectionnez la première Feature :", df.columns)
    selected_feature_2 = st.selectbox("Sélectionnez la deuxième Feature :", df.columns)
    
    fig = px.scatter(df, x=selected_feature_1, y=selected_feature_2, color=scores_clients, title="Analyse Bivariée")
    fig.update_traces(marker=dict(size=10, opacity=0.6), selector=dict(mode='markers'))
    
    client_feature_1 = df[selected_feature_1][example_idx]
    client_feature_2 = df[selected_feature_2][example_idx]
    
    fig.add_annotation(
        x=client_feature_1,
        y=client_feature_2,
        text=f"Client: ({client_feature_1},{client_feature_2})",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor="#636363",
        ax=-20,
        ay=30
    )
    
    st.plotly_chart(fig)

if __name__ == '__main__':
    main()