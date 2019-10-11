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
import shapely.geometry as sgeom
from shapely.geometry import Point, Polygon
import fiona

geolocator = Nominatim(user_agent="surge-protector")

#IMPORT MODELS
# filename = 'model/Poisson2reg.sav'
# count_model = pickle.load(open(filename, 'rb'))

filename = 'model/NegBin4reg.sav'
count_model = pickle.load(open(filename, 'rb'))

filename = 'model/rfClaimFrac.sav'
value_model = pickle.load(open(filename, 'rb'))


# IMPORT NON-USER INPUT DATA
coord_lookup = pd.read_csv('data/coord_lookup.csv') #contains coordinate [coords], distance [distance], and future housing density [future_density] info
# coords_NN_tract = pd.read_csv('data/coords_NN_tract_original.csv')
twoftSLR = "data/2ft_SLR.shp"
elevation = pd.read_csv('data/elevation.csv')
bio16 = pd.read_csv('data/bc_2050_16.csv')


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.COSMO])

server = app.server

app.layout = dbc.Container(children=[
	dbc.Jumbotron(
		[
			html.Div([
				html.H1("Flood\'d", className="display-1 text-white text-left font-weight-bold pl-4 bg"),
				html.H2(
					"Is flood insurance right for your Florida home?",
					className="text-white text-right font-weight-bold pb-4 pr-4",
				),
				html.Hr(className="my-2"),
				
			],
			style={ 'backgroundImage': "url(/assets/background-dark.jpg)",
					'backgroundRepeat': 'no-repeat',
					'backgroundPosition': 'center',
					'backgroundSize': 'cover'
				}
			)
		],
		className="p-0"
	),

	dbc.Row(
		[
			dbc.Col([
				html.Div()
			], className="col-1",
			),
			dbc.Col([
				html.H3(
						"Florida homeowners: ",
						className="text-left font-weight-bold"
				),
				html.H3(
						"Calculate the value of your annual flood insurance premium over 30 years, taking into account flood risk under climate change projections.",
						className="text-left pb-4"
				),
				html.H4(
							"A positive value indicates an expected gain.",
							className="text-left font-weight-bold pb-0 mb-0"
					),
				html.H4(
							"A negative value indicates an expected loss.",
							className="text-left font-weight-bold pb-0 mb-0"
					)
			], className='col-10'
			),
			dbc.Col([
				
			], className='col-6'
			),	
			dbc.Col([
				html.Div()
			], className="col-1",
			)
		]
	),


	dbc.Row(
		[
			dbc.Col(
				[
					dbc.FormGroup(
						[
							dbc.Label("What is your address?", html_for="address", className="h2"),
							dbc.Input(type="text", id="address", placeholder="Enter address"),
							html.P(id='coord_check', className="text-danger")
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
					)
				],
				className="col-6",
			),

			dbc.Col(
				[
					dcc.Graph(id='graph')
				],
				className="col-6",
			),
		],
		align="center"
	),
	dbc.Container(
		[
			html.Hr(className="my-2"),
			html.H1("RESULTS", className="text-center"),
			# html.H2(id="result", children="", className="text-center"),
			html.Div(id="result", className='text-center'),
			html.Hr(className="my-5"),
		]
	),
	dbc.Container(
		[
			html.Div([
					html.P("Data from FEMA NFIP, US Census, and IPCC", className="lead text-white text-center font-weight-bold"),
					html.P("Copyright: Erin Partlan, 2019", className="lead text-white text-center font-weight-bold"),
				],
				style={ 'backgroundImage': "url(/assets/background-dark.jpg)",
						'backgroundRepeat': 'no-repeat',
						'backgroundPosition': 'bottom',
						'backgroundSize': 'cover'
					}
				)
		],
		className='app-footer mb-0')
	
])


@app.callback(Output('coord_check', 'children'),
			   [Input('address', 'value')])
def coord_check(address):
	if (not address):
		return
	location = geolocator.geocode(address)
	if (location == None) :
		return 'Address not found, choose a nearby location.'



@app.callback(Output('result', 'children'),
			   [Input('address', 'value'),
				Input('home_value', 'value'),
				Input('premium', 'value')])
def update_metrics(address, home_value, premium):
	# geolocator = Nominatim(user_agent="surge-protector")
	if (not address):
		return '...'
	location = geolocator.geocode(address)
	if (location == None) or (not home_value) or (not premium):
		return '...'
	lat_ = round(location.latitude,1)
	long_ = round(location.longitude,1)
	coord_input =  str(lat_)+', '+str(long_)

	# get coastal proximity with 2ft SLR. better option is to calculate NN with exact coord input
	# dt = coord_lookup[coord_lookup.coords == coord_input].distance.values[0] #calculated from input location (NN coords or active calc to polygon)
	point1 =  Point(long_, lat_)
	coast = fiona.open(twoftSLR)
	distance = []
	for feature in coast:
	    geom = sgeom.shape(feature["geometry"])
	    distance_between_pts = geom.distance(point1)
	    distance.append(distance_between_pts)
	dt = max(distance)
	if dt == 0:
		return (html.Div([html.H4("Location expected to be chronically inundated")]))


	# get future density by looking up from table (future density is calculated as a linear extrapolation from 2013)
	ds = coord_lookup[coord_lookup.coords == coord_input].future_density.values[0] # lookup from input location (NN coords)
	# policy percent not needed because counts in model have been scaled. pp = coord_lookup[coords_NN_tract.coords == coord_input]['policy_percent'].values[0]
	el = elevation[elevation.coords == coord_input]['rvalue_1'].values[0]
	b16 = bio16[bio16.coords == coord_input]['rvalue_1'].values[0]

	# RUN MODELS
	intercept = 1
	count_test = np.array([intercept,dt,ds, el, b16])
	count_preds = count_model.get_prediction(count_test)
	count_func = count_preds.summary_frame()
	count = count_func['mean']#/(pp/100.0) # first column is mean count (2 is std err, 3 is ci lower, 4 is ci upper)
	prob = count/(ds*123876900) # 123876900 square meters in a square with side 0.1 degrees

	if prob.values[0]>=1:
		prob_30=1
	else:
		prob_30 = 1-((1-prob)**30)
	# return ('ds: '+str(ds)+', pp: '+str(pp)+'count: '+str(count)+', prob: '+str(prob)+', prob_30: '+str(prob_30))

	ct = count # predicted by Poisson2reg
	xtest = np.array([dt,ct,ds])
	frac =np.exp(value_model.predict(xtest.reshape(1, -1)))

	exp_value = prob_30*frac*float(home_value) - float(premium)*30

	if exp_value[0]>0:
		# return "Your expected annual value is: "+'${:0.2f}'.format(round(exp_value.values[0],2))+"A potential payout is likely to exceed your premium payments."
		return (html.Div([html.H4("The 30-year expected value of your insurance is: "+'${:0.2f}'.format(round(exp_value[0],2))), 
				html.H4("A potential claims payout is likely to exceed your premium payments.")]))

	if exp_value[0]<=0:
		# return "Your expected annual value is: "+'${:0.2f}'.format(round(exp_value.values[0],2))+"A potential payout is NOT likely to exceed your premium payments."
		return (html.Div([html.H4("The 30-year expected value of your insurance is: "+'${:0.2f}'.format(round(exp_value[0],2))), 
				html.H4("A potential claims payout is NOT likely to exceed your premium payments.")]))

@app.callback(Output('graph', 'figure'),
			   [Input('address', 'value')])
def graph_metrics(address):
	# geolocator = Nominatim(user_agent="surge-protector")
	# if (not address):
	# 	return
	lat_ = 28.5
	long_ = -81.4

	if address:
		location = geolocator.geocode(address)
		if (location != None):
			lat_ = round(location.latitude,1)
			long_ = round(location.longitude,1)

	data = [ 
		dict(
	        type = 'scattermapbox',
	        lon = [long_],
	        lat = [lat_],
	        mode = 'markers',
	        marker = dict(
	            size = 14,
	            color='black',
	        )
	    )
	]

	layout = dict(
			# hovermode='closest',
			width=600,
			height=600,
			# autosize=True,
			mapbox=dict(
				# layers=[],
				bearing=0,
	        	center=dict(
		            lat=lat_,
		            lon=long_
	        	),
	        	pitch=0,
	        	zoom=7,
				style='open-street-map',
				margin=dict(r=0,t=0,l=0,b=0)
				)
	        )

	fig = dict( data=data, layout=layout )  

	return fig

# @app.callback(
#	 dash.dependencies.Output('slider-output-container', 'children'),
#	 [dash.dependencies.Input('exp_value', 'value')])
# def update_output(exp_value):
#	 return #'You have selected "{}"'.format(exp_value)

if __name__ == '__main__':
	app.run_server(debug=True)
