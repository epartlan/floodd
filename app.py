import dash
import base64
import dash_html_components as html

app = dash.Dash(__name__)


test_png = 'plot.png'
test_base64 = base64.b64encode(open(test_png, 'rb').read()).decode('ascii')

#test_jpg = 'EDA(-80.7, 28.3).jpg'
#jpg_base64 = ase64.b64encode(open(test_jpg, 'rb').read()).decode('ascii')
#

app.layout = html.Div(children=[

    html.H1(children='Salty Living: Coastal housing under threat'),
    
    html.Div(children='''
        An interactive tool for understanding extents of flood damage in Florida and effectiveness of preventative measures.
    '''),

    html.Div([
        html.Img(src='data:image/png;base64,{}'.format(test_base64)),

#    html.Div([
#        html.Img(src='data:image/png;base64,{}'.format(jpg_base64)),
    ])
])

if __name__ == '__main__':
    app.run_server(debug=True)

