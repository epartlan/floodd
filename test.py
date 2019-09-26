import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd

mapbox_access_token = 'pk.eyJ1IjoiZGhhdmFscGFybWFyIiwiYSI6ImNrMHMyNm1pZTBidWwzY293MG1hbmVweHQifQ.mCPAd6toRMO1YdNVVnz-fA'

unique_coords = pd.read_csv('coords')

app = dash.Dash(__name__)

datamap = [px.scatter_mapbox(unique_coords, lat="latitude", lon="longitude", zoom=5, height=500)]



layoutmap = go.Layout(
	height=500,
	mapbox=dict(
		accesstoken=mapbox_access_token,
        style='white-bg',
        layers=[dict(
        	below='traces',
        	sourcetype="raster",
        	source=[
            	"https://basemap.nationalmap.gov/arcgis/rest/services/USGSImageryOnly/MapServer/tile/{z}/{y}/{x}"
        	]
        )]
	),
)

app.layout = html.Div(
    children=[
        dcc.Graph(
    		figure=dict(data=datamap, layout=layoutmap),
        	id='my-graph'
        ),
        dcc.Input(
		    placeholder='Search a location (address, etc.)',
		    type='text',
		    value='',
		    style={'width': '49%', 'display': 'inline-block'}),  
	])

# @app.callback(Output('my-graph', 'figure'))
# def update_plot():
# 	gapminder = px.data.gapminder().query("continent == 'Oceania'")
# 	fig = px.line(gapminder, x='year', y='lifeExp', color='country')

# 	return fig

if __name__ == '__main__':
    app.run_server(debug=True)
