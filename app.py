from server import app

import dash
import dash_core_components as dcc
import dash_html_components as html

from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc

import plotly_express as px

from components.navbar import navbar
from components.segmentation_model import loading_image_and_fig, pytorch_segmentation

import pandas as pd
import json
import random
import os
import time

server = app.server

app.layout = html.Div(
    [
        dcc.Store(data={"status": "Starting"}, id="webapp-starter"),
        dcc.Store(id="webapp-started"),
        dcc.Store(id="image-df-data"),
        navbar,
        html.Div(
            [
                html.Div(
                    [
                        html.Div(
                            html.Div(
                                [
                                    html.B("Ship detection"),
                                    " from remote sensing imagery is a crucial application for maritime security which includes among others traffic surveillance, protection against illegal fisheries, oil discharge control and sea pollution monitoring.",
                                ],
                                className="cardStyle",
                            ),
                            className="font-sm bottom64",
                        ),
                        html.Div(
                            [
                                html.Div(
                                    [
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    html.Div(
                                                        [
                                                            html.Div([
                                                                html.H4(
                                                                    ["Select the parameters of the prediction", 
                                                                        html.Span(html.I(className="fas fa-question-circle font-sm", 
                                                                                    id="tooltip-churn-info"), style={"marginLeft":"5px", "cursor":"pointer"})],
                                                                    id="title-box",
                                                                    className="h4", style={"marginBottom":"0px"},
                                                                ), 
                                                                
                                                                dbc.Tooltip(
                                                                    ["Noun: rare, "
                                                                    "the action or habit of estimating something as worthless."],
                                                                    target="tooltip-churn-info",
                                                                ),
                                                            ], className="flexVetCenter bottom16"),
                                                            html.Div(
                                                                [
                                                                    dbc.Label("Images"),
                                                                    dcc.RadioItems(
                                                                        id="pick-image-folder-radio",
                                                                        options=[
                                                                            {
                                                                                "label": "All Images",
                                                                                "value": "all",
                                                                            },
                                                                            {
                                                                                "label": "Only Ship Images",
                                                                                "value": "ship",
                                                                            },
                                                                        ],
                                                                        value="all",
                                                                        labelStyle={
                                                                            "display": "block",
                                                                            "marginBottom": "0px",
                                                                        },
                                                                    ),
                                                                ],
                                                                className="bottom16",
                                                            ),
                                                            html.Div(
                                                                html.Button(
                                                                    "Load Image",
                                                                    id="load-img",
                                                                    className="btn-style",
                                                                ),
                                                                className="bottom16",
                                                            ),
                                                            html.Div(
                                                                id="has-ship-on-image",
                                                                className="font-xs",
                                                            ),
                                                        ],
                                                        className="inputs-card-layout",
                                                    ),
                                                    lg=4,
                                                ),
                                                dbc.Col(
                                                    [
                                                        dbc.Row(
                                                            [
                                                                dbc.Col(
                                                                    dcc.Loading(
                                                                        html.Div(
                                                                            [
                                                                                html.Div(
                                                                                    [
                                                                                        "ImageID loaded: ",
                                                                                        html.Span(
                                                                                            id="loaded-image-name",
                                                                                            style={"fontWeight":"bold"}
                                                                                        ),
                                                                                    ], className="bottom32"
                                                                                ),
                                                                                dcc.Graph(
                                                                                    id="original-image-renderer"
                                                                                ),
                                                                            ],
                                                                            className="card-img-renderer",
                                                                        )
                                                                    ),
                                                                    lg=8,
                                                                ),
                                                                dbc.Col(
                                                                    [
                                                                        html.Div(
                                                                            html.Button(
                                                                                "Predict Ships",
                                                                                id="predict-btn",
                                                                                className="btn-style"
                                                                            )
                                                                        )
                                                                    ],
                                                                    lg=4,
                                                                ),
                                                            ],
                                                            style={
                                                                "alignItems": "center"
                                                            },
                                                            className="bottom16",
                                                        ),
                                                        html.Div([
                                                        html.H3(
                                                            "Validation Images",
                                                            className="text-center h3 bottom16",
                                                        ),
                                                        dbc.Row(
                                                            [
                                                                dbc.Col(
                                                                    [
                                                                        html.Div(
                                                                            "Ground Truth Mask",
                                                                            className="text-center",
                                                                        ),
                                                                        dcc.Loading(
                                                                            html.Div(
                                                                                [
                                                                                    html.Div(
                                                                                        id="ground-truth-image"
                                                                                    )
                                                                                ],
                                                                                className="card-gt-out",
                                                                            )
                                                                        ),
                                                                    ],
                                                                    lg=6,
                                                                ),
                                                                dbc.Col(
                                                                    [
                                                                        html.Div(
                                                                            "Model Output Mask",
                                                                            className="text-center",
                                                                        ),
                                                                        dcc.Loading(
                                                                            html.Div(
                                                                                [
                                                                                    html.Div(
                                                                                        id="output-image"
                                                                                    ),
                                                                                ],
                                                                                className="card-gt-out",
                                                                            )
                                                                        ),
                                                                    ],
                                                                    lg=6,
                                                                ),
                                                            ],
                                                            style={
                                                                "alignItems": "center"
                                                            },
                                                        )], id="output-results"),
                                                    ],
                                                    lg=8,
                                                ),
                                            ]
                                        )
                                    ]
                                )
                            ]
                        ),
                    ],
                    className="containerStyle",
                    style={"min-height": "85vh"},
                ),
                html.Footer(
                    "© 2021 Plotly. All rights reserved.", className="font-xs footer-padding"
                ),
            ]
        ),
    ],
    className="bgStyle",
)


# @app.callback(
#     Output("has-ship-on-image", "children"),
#     Output("original-image-renderer", "figure"),
#     Output("loaded-image-name", "children"),
#     Output("image-df-data", "data"),
#     Input("load-img", "n_clicks"),
#     State("pick-image-folder-radio", "value"),
# )
# def starting_the_app(btn_n, images_to_load):

#     txt, img_fig, validation_df = loading_image_and_fig(images_to_load)

#     return (
#         txt,
#         img_fig,
#         f"{validation_df[0]['ImageId']}",
#         validation_df,
#     )

# @app.callback(
#     Output("ground-truth-image", "children"),
#     Output("output-image", "children"),
#     Input("predict-btn", "n_clicks"),
#     State("image-df-data", "data"),
# )
# def starting_the_app(n_clicks, content):

#     if n_clicks == None:
#         raise PreventUpdate

#     time.sleep(2.5)

#     img, mask, output = pytorch_segmentation(pd.DataFrame.from_dict(content))

#     mask_fig = px.imshow(mask)

#     mask_fig.update_layout(
#         margin=dict(t=0, b=0, r=0, l=0),
#         height=300,
#         width=300,
#         xaxis=dict(ticks=None),
#         plot_bgcolor="rgba(0,0,0,0)",
#         paper_bgcolor="rgba(0,0,0,0)",
#         dragmode="drawrect",
#         newshape=dict(line_color="cyan"),
#     )

#     output_fig = px.imshow(output)
#     output_fig.update_layout(
#         margin=dict(t=0, b=0, r=0, l=0),
#         height=300,
#         width=300,
#         xaxis=dict(ticks=None),
#         plot_bgcolor="rgba(0,0,0,0)",
#         paper_bgcolor="rgba(0,0,0,0)",
#         dragmode="drawrect",
#         newshape=dict(line_color="cyan"),
#     )

#     return (
#         dcc.Graph(
#             figure=mask_fig,
#             config={
#                 "modeBarButtonsToAdd": [
#                     "drawline", "drawopenpath", "drawclosedpath",
#                     "drawcircle", "drawrect", "eraseshape" ]
#             },
#         ),
#         dcc.Graph(
#             figure=output_fig,
#             config={
#                 "modeBarButtonsToAdd": [
#                     "drawline", "drawopenpath", "drawclosedpath",
#                     "drawcircle", "drawrect", "eraseshape"]
#             },
#         ),
#     )


# @app.callback(
#     Output("webapp-started", "data"),
#     Input("webapp-starter", "data"),
#     State("loaded-image-name", "children"),
# )
# def starting_the_app(app_starter, content):

#     if app_starter["status"] == "Starting" and content == None:
#         return {"status": "started"}
#     else:
#         return dash.no_update


# @app.callback(
#     Output("output-results", "style"),
#     Input("predict-btn", "n_clicks")
# )
# def starting_the_app(predict_btn):

#     if predict_btn == None:
#         return {"display":"none"}
#     else:
#         return {"display":"block"}


if __name__ == "__main__":
    app.run_server(debug=True, port="8110")
