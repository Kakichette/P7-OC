#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys
sys.path.append(r'/home/p7api/.local/lib/python3.10/site-packages')

from flask import Flask, request
import pandas as pd
import pickle
from zipfile import ZipFile
import os
application = Flask(__name__)

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

# Charger le modèle pickle


filename = '/home/p7api/data/lgbm.pkl'  # Replace with the actual filename or path of your pickle file

with open(filename, 'rb') as fs:
    model = pickle.load(fs)

# Charger l'ensemble de données depuis un fichier CSV
z1 = ZipFile("/home/p7api/data/X_data_test.zip")
data = pd.read_csv(z1.open('X_data_test.csv'), encoding ='utf-8')
#data = data.drop(["TARGET"], axis=1)
###################################################################################################################
@application.route("/")
def home():
    return "Bienvenue sur l'application Flask !"
###################################################################################################################
@application.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        # Récupérer l'identifiant du client depuis le formulaire
        client_id = int(request.form["client_id"])
    elif request.method == "GET":
        # Récupérer l'identifiant du client depuis l'URL
        client_id = int(request.args.get("client_id"))

    # Vérifier si l'identifiant du client existe dans l'ensemble de données
    if client_id not in data["SK_ID_CURR"].values:
        return f"L'identifiant client {client_id} n'existe pas dans l'ensemble de données."

    # Filtrer les données pour le client spécifié
    client_data = data[data["SK_ID_CURR"] == client_id]

    # Extraire les caractéristiques du client
    features = client_data.drop(["SK_ID_CURR"], axis=1)

    # Effectuer la prédiction en utilisant le modèle chargé
    y_pred = model.predict_proba(features)

    # Calculer le score  à partir de y_true et y_pred
    score_metier = custom_score(y_pred[:, 1])  # Appeler votre fonction de calcul du score métier

    # Renvoyer le score métier en tant que réponse
    return score_metier


def custom_score(y_pred):
    # Implémentez votre propre logique pour calculer le score métier
    # Utilisez les valeurs de y_true et y_pred pour effectuer les calculs nécessaires
    # Par exemple :
    if y_pred >= 0.5 and y_pred <0.59:
        score = 3
    elif y_pred >= 0.59:
        score = 1
    elif y_pred < 0.5:
        score = 0
    return str(score)

