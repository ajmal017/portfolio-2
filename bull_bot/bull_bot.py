import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html
from IPython.display import display, HTML
from dash.dependencies import Input, Output
import requests

# TODO: Change to CSS
from BullColors import colors
from BullScreener import BullScreener
from BullGraph import *
import rh_interface

app = dash.Dash(__name__)
bg = BullGraph(start_date=start_date, end_date=end_date)
rh_holdings = rh_interface.get_rh_holdings()

# Creating the Ticker input
app.scripts.config.serve_locally = True
app.css.config.serve_locally = True

ticker_bar = dcc.Input(id="ticker_in", type="text", placeholder="NVDA", debounce=True)
    # Creating dash table of rh holdings
columns = [{'name' : column, 'id' : column} for column in rh_holdings.columns]
holdings_table = dash_table.DataTable(id="holdings_table", columns=columns, data=rh_holdings.to_dict('records'))
graph_style = {'display':'inline-block', 'vertical-align' : 'top', 'margin-left' : '3vw', 'margin-top' : '3vw', 'width':'49%'}
app.layout = html.Div(
    style={
        "background-color" : colors["page_background"]
    },
    children=[
        dcc.Graph(id="main_graph", figure=bg.fig, style=graph_style),
        html.Div(
            children=holdings_table,
            style={'display':'inline-block', 'vertical_align':'right', 'margin-left':'3vw', 'margin-right':'3vw', 'margin-top':'10vw', 'width':'40%'}
        ),
        ticker_bar,
        html.Div(id="ticker_out")
    ]
)

# Rendering the ticker input for the graph
@app.callback(Output("main_graph", "figure"),
[Input("ticker_in", "value")])
def ticker_render(ticker=None):
    if (ticker == None):
        bg.ticker = "NVDA"
        fig = bg.createFig()
        fig, historical = bg.styleFig()
        return fig

    bg.ticker = ticker
    fig = bg.createFig()
    fig, historical = bg.styleFig()
    return fig


if __name__ == "__main__":
    # Initializing stock screener
    bs = BullScreener(ticker_list=['NVDA', 'MSFT', 'AAPL', 'TSLA'])
    #bs.get_trendlines()

    # Run server
    app.run_server(debug=True, use_reloader=False)  # Turn off reloader if inside Jupyter