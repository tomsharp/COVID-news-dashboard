import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output, State

import plotly.express as px
import plotly.graph_objects as go

import pandas as pd

app = dash.Dash(__name__)
app.title = 'COVID-19 News'
app.css.config.serve_locally = True
app.scripts.config.serve_locally = True
server = app.server


vis_df = pd.read_csv('data/lda/vis.csv')
topic_term_df = pd.read_csv('data/lda/topic_term.csv').sort_values('Freq', ascending=True)

app.layout = html.Div([

    # header
    html.Div(className='header', children=[
        html.Div(className='my-auto row', children=[
            html.H2('COVID-19 News'),
        ])
    ]),
    
    # tabs
    # dcc.Tabs(id="tabs", value='topic-modeling', children=[
    #     dcc.Tab(label='Topic Modeling', value='topic-modeling'),
    #     dcc.Tab(label='Sentiment Analysis', value='sentiment-analysis'),
    # ]),

    # content
    html.Div(className='mx-6', children=[
        dcc.Dropdown(
            id='news-dropdown',
            options = [{'label': i, 'value': i} for i in ['business', 'finance', 'general', 'tech']],
            value = 'general'
        ),

        html.Div(className='row', children=[
            html.Div(className='col-lg-5', children=[
                dcc.Graph(id='topic-plot', 
                    config={'displayModeBar': False}
                )
            ]),
            html.Div(className='col-lg-5', children=[
                dcc.Graph(id='terms-plot', 
                    config={'displayModeBar': False}
                )
            ]),
        ]),

    ])

])



@app.callback(Output('topic-plot', 'figure'),
            [Input('news-dropdown', 'value')])
def display_topic_vis(news):
    df = vis_df[vis_df['topic_area']==news]
    if news != None:
        fig = go.Figure(
            go.Scatter(
                x = df['x'],
                y = df['y'],
                mode='markers',
                marker = {
                    'size': 100*df['Freq']/len(df),
                    'color': 'lightblue',
                },
                hovertext = df['topics'],
            
            ),
        )

        fig.update_layout(
            xaxis = {'title':'PC1', 'tickvals':[0]},
            yaxis = {'title':'PC2', 'tickvals':[0]},
            plot_bgcolor='rgba(0,0,0,0)',
            autosize=False,
        )
        fig.update_xaxes(zeroline=True, zerolinewidth=1, zerolinecolor='black')
        fig.update_yaxes(zeroline=True, zerolinewidth=1, zerolinecolor='black')
        
        return fig
    else:
        return go.Figure()

@app.callback(Output('terms-plot', 'figure'),
             [Input('news-dropdown', 'value'),
             Input('topic-plot', 'hoverData')])
def display_terms_vis(news, hover):
    if hover != None:
        topic = hover['points'][0]['hovertext']

        df = topic_term_df[(topic_term_df['topic_area']==news)&(topic_term_df['Category']=='Topic{}'.format(topic))].head(30)
        print(df)
        
        fig = go.Figure(
            data=[
                go.Bar(name='Term Frequency for Selected Topic', y=df['Term'], x=df['Freq'], orientation='h'),
                go.Bar(name='Overall Term Frequency', y=df['Term'], x=df['Total'], orientation='h'),
            ],
            layout={
                'transition': {
                    'duration': 500,
                    'easing': 'cubic-in-out'
                },
            },
        )
        fig.update_layout(barmode='stack', legend=dict(x=0, y=-0.5))
        return fig
    else:
        return go.Figure()

if __name__ == '__main__':
    app.run_server(debug=True)