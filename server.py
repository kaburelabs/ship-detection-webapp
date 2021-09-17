import dash
import dash_bootstrap_components as dbc


# CSS files and links
external_stylesheets = [dbc.themes.BOOTSTRAP,  # adding the bootstrap inside the application
                        'https://use.fontawesome.com/releases/v5.8.1/css/all.css',
                        ]

app = dash.Dash(__name__, 
                external_stylesheets=external_stylesheets, 
                meta_tags=[
                    {
                        "name": "viewport", 
                        "content":"width=device-width, initial-scale=1, maximum-scale=1"
                    }])

app.config.suppress_callback_exceptions = True
app.css.config.serve_locally=True

app.title="Plotly Ship Detection"