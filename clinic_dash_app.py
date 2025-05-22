
# -*- coding: utf-8 -*-
"""
Enhanced Clinic P&L Analyzer - Dash App
Features:
- Upload P&L Excel file and run margin analysis
- Display table with all results
- Chart: Number of categories per Priority Score
- Chart: Number of times each Action Item is assigned
- Table: Categories per Action Item
- Table: Categories in Top 3 Priority Scores
- Export table to CSV
"""

import pandas as pd
import numpy as np
import base64
from io import BytesIO
from dash import Dash, dcc, html, dash_table, Input, Output, State
import plotly.express as px

app = Dash(__name__)
server = app.server  # for deployment

app.layout = html.Div([
    html.H2("📊 Enhanced Clinic Margin Analyzer"),
    dcc.Upload(
        id='upload-data',
        children=html.Div(['📁 Drag and drop or click to upload your P&L Excel file']),
        style={
            'width': '100%', 'height': '60px', 'lineHeight': '60px',
            'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '5px',
            'textAlign': 'center', 'margin': '10px'
        },
        multiple=False
    ),
    html.Div(id='full-table-output'),
    dcc.Download(id="download-dataframe-csv"),
    html.Button("Download Table as CSV", id="btn_csv"),
    html.Br(), html.H4("📊 Categories by Priority Score"),
    dcc.Graph(id="priority-score-chart"),
    html.H4("📊 Action Assignment Frequency"),
    dcc.Graph(id="action-frequency-chart"),
    html.H4("📋 Categories by Action Item"),
    html.Div(id="action-category-table"),
    html.H4("🏆 Categories in Top 3 Priority Scores"),
    html.Div(id="top-priority-table")
])

def process_uploaded_file(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    df = pd.read_excel(BytesIO(decoded), sheet_name="Visual Summary")

    return df

@app.callback(
    Output('full-table-output', 'children'),
    Output('priority-score-chart', 'figure'),
    Output('action-frequency-chart', 'figure'),
    Output('action-category-table', 'children'),
    Output('top-priority-table', 'children'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def update_output(contents, filename):
    if contents is None:
        return None, {}, {}, None, None

    df = process_uploaded_file(contents, filename)

    # --- Full Table ---
    table = dash_table.DataTable(
        data=df.to_dict('records'),
        columns=[{"name": i, "id": i} for i in df.columns],
        page_size=15,
        style_table={'overflowX': 'auto'},
        style_cell={'textAlign': 'left'},
        id='full-table'
    )

    # --- Priority Score Chart ---
    score_fig = px.histogram(df, x="Priority Score", title="Number of Categories by Priority Score")

    # --- Action Frequency Chart ---
    action_columns = [
        "Low margin leverage",
        "No issues",
        "Ratio increased",
        "April outlier",
        "Statistically significant change",
        "High slope (scales with income)"
    ]
    action_counts = df[action_columns].sum().reset_index()
    action_counts.columns = ["Action Item", "Count"]
    action_fig = px.bar(action_counts, x="Action Item", y="Count", title="Frequency of Action Assignments")

    # --- Table of Categories by Action Item ---
    action_tables = []
    for col in action_columns:
        filtered = df[df[col] == 1][["Category"]]
        filtered["Action"] = col
        action_tables.append(filtered)
    actions_combined = pd.concat(action_tables)

    actions_table = dash_table.DataTable(
        data=actions_combined.to_dict('records'),
        columns=[{"name": i, "id": i} for i in actions_combined.columns],
        page_size=10
    )

    # --- Top Priority Scores Table (Top 3 only) ---
    top_scores = df[df["Priority Score"] >= df["Priority Score"].max() - 2][["Category", "Priority Score"]]
    top_priority_table = dash_table.DataTable(
        data=top_scores.to_dict('records'),
        columns=[{"name": i, "id": i} for i in top_scores.columns],
        page_size=10
    )

    return table, score_fig, action_fig, actions_table, top_priority_table

@app.callback(
    Output("download-dataframe-csv", "data"),
    Input("btn_csv", "n_clicks"),
    State("upload-data", "contents"),
    prevent_initial_call=True
)
def download_csv(n_clicks, contents):
    if contents is None:
        return None
    df = process_uploaded_file(contents, "file")
    return dcc.send_data_frame(df.to_csv, "clinic_analysis_results.csv")

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8050))
    app.run(host="0.0.0.0", port=port, debug=True)
