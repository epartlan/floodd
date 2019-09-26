import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import pandas as pd
import pickle

filename = 'MA1.sav'
loaded_model = pickle.load(open(filename, 'rb'))

coords = pd.read_csv("coords")
obs_count = pd.read_csv('obs_count')

zhat_all = pd.Series([])
for i in range(len(coords.coords)):
    data = obs_count[coords.coords[i]][:-5]
    if data.sum() == 0:
        zhat = pd.DataFrame(pd.Series([0,0,0,0,0,0,0,0]).values.reshape(1,-1))
        zhat_all = pd.concat([zhat_all, zhat])
    else:
        zhat = loaded_model.predict(len(data), len(data)+7)
        zhat = pd.DataFrame(zhat.values.reshape(1,-1))
        zhat_all = pd.concat([zhat_all, zhat]) 



external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(children=[
	html.H1(children='Salty Living: Coastal housing under threat'),

    html.Div(children='An interactive tool for understanding extents of flood damage in Florida and effectiveness of preventative measures.'),

 #    dcc.Input(
	# 	placeholder='',
	# 	id='address',
	# 	type='text',
	# 	value=''
	# ),



	dcc.Graph(
		id='example-graph',
		),
		style={
			'width': '35%',
			'display': 'inline-block',
			'vertical-align': 'middle'
		}
	),

	dcc.Dropdown(
		id='address',
	    options=[
	        {'label': '2019', 'value': 5},
	        {'label': '2020', 'value': 6},
	        {'label': '2021', 'value': 7}
	    ],
	    value=5
	),  

	html.Div(
		children=coords["longitude"].values[0],
		id='model-values',
		style={
			'width': '35%',
			'display': 'inline-block',
			'vertical-align': 'middle'
		}
	),
])

@app.callback(Output('model-values', 'children'),
              [Input('address', 'value')])
def update_metrics(n):
	return zhat[n].mean()


@app.callback(Output('example-graph', 'children'),
              [Input('address', 'value')])
def update_metrics(n):
	return go.Figure(
			data=go.Scatter(x=coords["longitude"].values, y=coords["latitude"].values, mode='markers'),
			layout=go.Layout(dict(
				height=600,
				margin={'l': 0, 'b': 0, 't': 0, 'r': 0}
			))


if __name__ == '__main__':
	app.run_server(debug=True)
