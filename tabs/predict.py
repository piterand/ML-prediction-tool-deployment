from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html

from joblib import load
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from pandas.api.types import CategoricalDtype
import json

from app import app


style = {'padding': '1.5em'}

layout = html.Div([
    dcc.Markdown("""
        ### Predict
        Use the controls below to predict outcome

    """),

    html.Div([
        dcc.Markdown('###### Age'),
        dcc.Input(
            placeholder='Enter a value...',
            type='number',
            debounce=False,
            id='Age',
        ),
    ], style=style),

    html.Div([
        dcc.Markdown('###### Pclass'),
        dcc.Input(
            placeholder='Enter a value...',
            type='number',
            debounce=False,
            id='Pclass'
        ),
    ], style=style),

    html.Div([
        dcc.Markdown('###### Fare'),
        dcc.Input(
            placeholder='Enter a value...',
            type='number',
            debounce=False,
            id='Fare'
        ),
    ], style=style),

    html.Div([
        dcc.Markdown('###### Sex'),
        dcc.Dropdown(
            ['male', 'female', 'dont know'],
            'dont know',
            id='Sex'
        ),
    ], style=style),
    html.Br(),
    html.Div(id='prediction-content', style={'fontWeight': 'bold'}),

])

@app.callback(
    Output('prediction-content', 'children'),
    [Input('Age', 'value'),
     Input('Pclass', 'value'),
     Input('Fare', 'value'),
     Input('Sex', 'value')])
def predict(Age, Pclass, Fare, Sex):

    df = pd.DataFrame(
        columns=['Age', 'Pclass', 'Fare', 'Sex'],
        data=[[Age, Pclass, Fare, Sex, ]]
    )
    with open('model/categories.json', 'r') as fp1:
        read_categories = json.load(fp1)

    df['Age'] = df['Age'].astype('float')
    df['Pclass'] = df['Pclass'].astype('float')
    df['Fare'] = df['Fare'].astype('float')
    df['Sex'] = df['Sex'].astype(CategoricalDtype(categories=read_categories['Sex'], ordered=True))
    
    #print(df)

    model_from_save = XGBClassifier()
    model_from_save.load_model("model/model.json")

    pred = model_from_save.predict_proba(df)

    results = pred[0][1]
    format_results = float('{:.2f}'.format(100*results))


    return f'Probability: {format_results}%'
