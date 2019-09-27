import dash
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



external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

server = app.server

app.layout = html.Div(children=[
	html.H1(children='Surge Protector',
		style={
            'textAlign': 'center',
        }),
    html.H2(children='Make smarter home buying decisions',
    	style={
            'textAlign': 'center',
        }),
    html.Div(children="Uses the likelihood of future flood damage to estimate insurance premiums for homes in Florida.",
  		style={
            'textAlign': 'center',
        }),


    html.Div(className="row", children=[
        html.Div(className="six columns", 
            children=['Location: ',
            dcc.Input(
		     	placeholder='Enter address, etc.',
		     	id='address',
		     	type='text',
		     	value=''
	    	),
	    	'Home Value: ',
	    	dcc.Input(
		     	placeholder='Enter home value',
		     	id='home_value',
		     	type='text',
		     	value=''
	    	)],
        ),
        html.Div(className="one column",children='Min. Coverage '),
        html.Div(className="four columns", children=[
        	dcc.Slider(
                id='exp_value',
                min=0,
                max=500,
                step=50,
                marks={
                    0: 'High Risk',
                    500: 'Low Risk',
                },
                value=250,
            ),
            html.Div(id='slider-output-container'),
        ]),
        html.Div(className="one column",children='Excess Coverage'),
                
        html.Div(
			children='',
			id='premium'
			),
		]),
    ])


@app.callback(Output('premium', 'children'),
              [Input('address', 'value'),
               Input('exp_value', 'value'),
               Input('home_value', 'value')])
def update_metrics(address, exp_value, home_value):
	geolocator = Nominatim(user_agent="surge-protector")
	location = geolocator.geocode(address)
	if location == None or home_value == "":
		return ''
	lat_ = round(location.latitude,1)
	long_ = round(location.longitude,1)
	coord_input =  str(lat_)+', '+str(long_)
	dt = coord_lookup[coord_lookup.coords == coord_input].distance.values[0] #calculated from input location (NN coords or active calc to polygon)
	ds = coord_lookup[coord_lookup.coords == coord_input].future_density.values[0] # lookup from input location (NN coords
	# RUN MODELS
	intercept = 1
	count_test = np.array([intercept,dt,ds])
	count_preds = count_model.get_prediction(count_test)
	count_func = count_preds.summary_frame()
	count = count_func['mean'] # first column is mean count (2 is std err, 3 is ci lower, 4 is ci upper)
	prob = count/(dt*123876900) # 123876900 square meters in a square with side 0.1 degrees

	ct = count # predicted by Poisson2reg
	xtest = np.array([dt,ct,ds])
	frac =np.exp(value_model.predict(xtest.reshape(1, -1)))

	premium = (float(exp_value) - prob*frac*float(home_value))/(1 - prob)

	return 'Monthly Premium = ${:0.2f}'.format(round(premium.values[0],2))
	# return "$"+home_value+"$"

@app.callback(
    dash.dependencies.Output('slider-output-container', 'children'),
    [dash.dependencies.Input('exp_value', 'value')])
def update_output(exp_value):
    return #'You have selected "{}"'.format(exp_value)

if __name__ == '__main__':
	app.run_server(debug=True)
