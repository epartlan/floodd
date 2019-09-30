import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import pickle
from patsy import dmatrices
import statsmodels.api as sm
import geopy
from geopy.geocoders import Nominatim

#IMPORT MODELS
filename = 'model/Poisson2reg.sav'
count_model = pickle.load(open(filename, 'rb'))

filename = 'model/rfClaimFrac.sav'
value_model = pickle.load(open(filename, 'rb'))


# IMPORT NON-USER INPUT DATA
coord_lookup = pd.read_csv('data/coord_lookup.csv') #contains coordinate [coords], distance [distance], and future housing density [future_density] info
coords_NN_tract = pd.read_csv('data/coords_NN_tract.csv')



# external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.COSMO])

server = app.server

app.layout = dbc.Container(children=[
	dbc.Jumbotron(
		[
			html.H1("Flood\'d", className="display-1"),
			html.P(
				"Is flood insurance right for you?",
				className="lead",
			),
			html.Hr(className="my-2"),
			html.P(
				"Uses the likelihood of future flood damage to estimate the value of insurance for homes in Florida."
			)
		]
	),

	dbc.FormGroup(
		[
			dbc.Label("What is your address?", html_for="address", className="h2"),
			dbc.Input(type="text", id="address", placeholder="Enter address"),
			# dbc.FormText(
			#	 "Are you on email? You simply have to be these days",
			#	 color="secondary",
			# ),
		]
	),
	dbc.FormGroup(
		[
			dbc.Label("What is the value of your home?", html_for="home_value", className="h2"),
			dbc.Input(type="text", id="home_value", placeholder="Enter home value"),
		]
	),
	dbc.FormGroup(
		[
			dbc.Label("What is your annual flood insurance premium?", html_for="home_value", className="h2"),
			dbc.Input(type="text", id="premium", placeholder="Enter premium"),
		]
	),

	dbc.Container(
		[
			html.Hr(className="my-2"),
			html.H1("RESULTS", className="text-center"),
			html.H2("The expected annual value of your flood insurance is: ", className="text-center"),
			html.H2(id="result", children="", className="text-center"),
			html.Hr(className="my-2"),
		]
	),
	
])


@app.callback(Output('result', 'children'),
			   [Input('address', 'value'),
				Input('home_value', 'value'),
				Input('premium', 'value')])
def update_metrics(address, home_value, premium):
	geolocator = Nominatim(user_agent="surge-protector")
	location = geolocator.geocode(address)
	if (not address) or (location == None) or (not home_value) or (not premium):
		return ''
	lat_ = round(location.latitude,1)
	long_ = round(location.longitude,1)
	coord_input =  str(lat_)+', '+str(long_)

	dt = coord_lookup[coord_lookup.coords == coord_input].distance.values[0] #calculated from input location (NN coords or active calc to polygon)
	ds = coord_lookup[coord_lookup.coords == coord_input].future_density.values[0] # lookup from input location (NN coords)
	pp = coords_NN_tract[coords_NN_tract.coords == coord_input]['policy_percent'].values[0]
	
	# RUN MODELS
	intercept = 1
	count_test = np.array([intercept,dt,ds])
	count_preds = count_model.get_prediction(count_test)
	count_func = count_preds.summary_frame()
	count = count_func['mean']/(pp/100.0) # first column is mean count (2 is std err, 3 is ci lower, 4 is ci upper)
	prob = count/(ds*123876900) # 123876900 square meters in a square with side 0.1 degrees
	prob_30 = 1-((1-prob)**30)

	# return ('ds: '+str(ds)+', pp: '+str(pp)+'count: '+str(count)+', prob: '+str(prob)+', prob_30: '+str(prob_30))

	ct = count # predicted by Poisson2reg
	xtest = np.array([dt,ct,ds])
	frac =np.exp(value_model.predict(xtest.reshape(1, -1)))

	exp_value = prob_30*frac*float(home_value) - (1 - prob_30)*float(premium)

	return '${:0.2f}'.format(round(exp_value.values[0],2))

# @app.callback(
#	 dash.dependencies.Output('slider-output-container', 'children'),
#	 [dash.dependencies.Input('exp_value', 'value')])
# def update_output(exp_value):
#	 return #'You have selected "{}"'.format(exp_value)

if __name__ == '__main__':
	app.run_server(debug=True)
