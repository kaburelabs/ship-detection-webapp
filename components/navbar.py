import dash_bootstrap_components as dbc
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State

PLOTLY_LOGO = "https://images.plot.ly/logo/new-branding/plotly-logomark.png"

options_navbar = dbc.Row(
    [
        dbc.Col(
            [
                dcc.Link("Classsifier", href="/classifier", className="right8 white"),
            ],
            width="auto",
        ),
    ],
    no_gutters=True,
    className="ml-auto flex-nowrap mt-3 mt-md-0",
    align="center",
)

navbar = dbc.Navbar(
    [
        # Use row and col to control vertical alignment of logo / brand
        dbc.Row(
            [
                dbc.Col(
                    html.A(
                        html.Img(src=PLOTLY_LOGO, height="30px"),
                        href="https://plotly.com",
                        target="blank",
                    )
                ),
                dbc.Col(dbc.NavbarBrand("Ship detection", className="ml-2")),
            ],
            align="center",
            no_gutters=True,
        ),
        dbc.NavbarToggler(id="navbar-toggler", n_clicks=0),
        dbc.Collapse(options_navbar, id="navbar-collapse", navbar=True, is_open=False),
    ],
    # className="bottom32",
    color="dark",
    dark=True,
)
