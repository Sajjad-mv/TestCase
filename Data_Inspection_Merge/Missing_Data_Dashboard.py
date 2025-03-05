import pandas as pd
import numpy as np
import os
import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go


def prepare_heatmap_data(home_df, away_df):
    """Combine Home and Away teams' missing values into one dataframe with player IDs as columns."""
    home_data = home_df.isna().astype(int)
    away_data = away_df.isna().astype(int)
    home_data.columns = [f'Home_{col}' for col in home_data.columns]
    away_data.columns = [f'Away_{col}' for col in away_data.columns]
    combined_data = pd.concat([home_data, away_data], axis=1)
    return combined_data, home_data, away_data


def create_unified_heatmap(home_df, away_df, match_id):
    """Create a single heatmap with distinct colors for Home and Away players and detailed hover information."""
    combined_data, home_data, away_data = prepare_heatmap_data(home_df, away_df)

    home_indices = [i for i, col in enumerate(combined_data.columns) if col.startswith('Home_')]
    away_indices = [i for i, col in enumerate(combined_data.columns) if col.startswith('Away_')]

    fig = go.Figure()

    # Plot Home team with hover info
    fig.add_trace(go.Heatmap(
        z=home_data.values,
        x=home_data.columns,
        y=[f'Time_{i}' for i in range(home_data.shape[0])],
        colorscale=[[0, 'white'], [1, 'blue']],
        colorbar=dict(title='Home Missing', len=0.5, y=0.75),
        hovertemplate='Player: %{x}<br>Time: %{y}<extra></extra>'
    ))

    # Plot Away team with hover info
    fig.add_trace(go.Heatmap(
        z=away_data.values,
        x=away_data.columns,
        y=[f'Time_{i}' for i in range(away_data.shape[0])],
        colorscale=[[0, 'white'], [1, 'orange']],
        colorbar=dict(title='Away Missing', len=0.5, y=0.25),
        hovertemplate='Player: %{x}<br>Time: %{y}<extra></extra>'
    ))

    fig.update_layout(
        title=f'Missing Values Heatmap - {match_id} (Home and Away Combined)',
        xaxis_title='Player IDs (Home and Away)',
        yaxis_title='Time Steps',
        width=2600, height=900,
        showlegend=True
    )

    return fig


def load_data(data_dir, match_ids):
    """Load data for all matches."""
    match_data = {}
    for match_id in match_ids:
        home_df = pd.read_csv(os.path.join(data_dir, match_id, 'Home.csv'))
        away_df = pd.read_csv(os.path.join(data_dir, match_id, 'Away.csv'))
        match_data[match_id] = {'Home': home_df, 'Away': away_df}
    return match_data


def run_dashboard(data_dir, match_ids):
    """Run the Dash dashboard for interactive heatmaps."""
    match_data = load_data(data_dir, match_ids)

    app = dash.Dash(__name__)

    app.layout = html.Div([
        html.H1('Football Tracking Data - Combined Missing Values Heatmap', style={'textAlign': 'center'}),
        html.Div([
            html.Button(f'Match {i + 1}', id=f'btn-{match_id}', n_clicks=0,
                        style={'margin': '10px', 'padding': '10px', 'fontSize': '16px'})
            for i, match_id in enumerate(match_ids)
        ], style={'textAlign': 'center'}),
        dcc.Graph(id='heatmap-graph')
    ])

    @app.callback(
        Output('heatmap-graph', 'figure'),
        [Input(f'btn-{match_id}', 'n_clicks') for match_id in match_ids]
    )
    def update_heatmap(*args):
        ctx = dash.callback_context
        if not ctx.triggered:
            return create_unified_heatmap(match_data[match_ids[0]]['Home'], match_data[match_ids[0]]['Away'], match_ids[0])
        match_id = ctx.triggered[0]['prop_id'].split('.')[0].split('-')[1]
        return create_unified_heatmap(match_data[match_id]['Home'], match_data[match_id]['Away'], match_id)

    app.run_server(debug=True, host='127.0.0.1', port=8050)


if __name__ == '__main__':
    data_dir = ''
    match_ids = ['match_1', 'match_2', 'match_3', 'match_4', 'match_5']
    run_dashboard(data_dir, match_ids)