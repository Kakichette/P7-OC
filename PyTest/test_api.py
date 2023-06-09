#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pytest
from flask import Flask
from flask.testing import FlaskClient

from flask_app import application as app  # Remplacez `your_app_file` par le nom de votre fichier d'application


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_predict_with_valid_id_0(client: FlaskClient):
    response = client.post("/predict", data={"client_id": 100013})
    assert response.status_code == 200
    assert response.data.decode("utf-8") == "0"  # Remplacez `your_expected_response` par la réponse attendue

def test_predict_with_valid_id_3(client: FlaskClient):
    response = client.post("/predict", data={"client_id": 101428})
    assert response.status_code == 200
    assert response.data.decode("utf-8") == "3"

def test_predict_with_valid_id_3(client: FlaskClient):
    response = client.post("/predict", data={"client_id": 100092})
    assert response.status_code == 200
    assert response.data.decode("utf-8") == "1"

def test_predict_with_invalid_id(client: FlaskClient):
    response = client.post("/predict", data={"client_id": 999})
    assert response.status_code == 200
    assert response.data.decode("utf-8") == "L'identifiant client 999 n'existe pas dans l'ensemble de données."  # Remplacez `your_expected_error_message` par le message d'erreur attendu


if __name__ == "__main__":
    pytest.main()


