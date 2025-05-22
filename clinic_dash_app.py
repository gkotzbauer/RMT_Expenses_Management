
# -*- coding: utf-8 -*-
"""
Clinic P&L Analyzer - Dash App
Upload a P&L Excel file, analyze key cost metrics, and display results interactively.
"""

import pandas as pd
import numpy as np
from dash import Dash, dcc, html, dash_table, Input, Output, State
from sklearn.linear_model import LinearRegression
from scipy.stats import ttest_ind
import plotly.express as px

app = Dash(__name__)
server = app.server  # for deployment

app.layout = html.Div([
    html.H2("Clinic P&L Analyzer"),
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
    html.Div(id='output-table'),
    html.Div(id='charts-output')
])

def analyze_pnl(contents, filename):
    from io import BytesIO
    import base64

    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    pnl_df = pd.read_excel(BytesIO(decoded), sheet_name="Profit and Loss")

    # Step 1: Extract data
    expenses_start = pnl_df[pnl_df.iloc[:, 0] == "Expenses"].index[0]
    expense_data = pnl_df.iloc[expenses_start + 1:].copy()
    expense_data.columns = ['Category', 'Jan', 'Feb', 'Mar', 'Apr', 'Total']
    expense_data = expense_data[['Category', 'Jan', 'Feb', 'Mar', 'Apr']].dropna(subset=['Category'])
    expense_data = expense_data[~expense_data['Category'].str.contains("Total", na=False)]
    expense_data = expense_data.reset_index(drop=True)

    for col in ['Jan', 'Feb', 'Mar', 'Apr']:
        expense_data[col] = pd.to_numeric(expense_data[col], errors='coerce').fillna(0)

    monthly_income = {
        'Jan': 141764.07,
        'Feb': 115152.98,
        'Mar': 144090.18,
        'Apr': 147081.04
    }

    results = []

    for _, row in expense_data.iterrows():
        cat, jan, feb, mar, apr = row['Category'], row['Jan'], row['Feb'], row['Mar'], row['Apr']
        rj, rf, rm, ra = jan / monthly_income['Jan'], feb / monthly_income['Feb'], mar / monthly_income['Mar'], apr / monthly_income['Apr']
        pct_cost = (apr - jan) / jan if jan != 0 else np.nan
        pct_ratio = (ra - rj) / rj if rj != 0 else np.nan

        X = np.array(list(monthly_income.values())).reshape(-1, 1)
        y = np.array([jan, feb, mar, apr])
        reg = LinearRegression().fit(X, y)
        slope, intercept, r2 = reg.coef_[0], reg.intercept_, reg.score(X, y)
        inv_score = 1 / (1 + max(slope, 0))
        leverage = inv_score * r2

        mean_jfm = np.mean([jan, feb, mar])
        std_jfm = np.std([jan, feb, mar], ddof=0)
        z_apr = (apr - mean_jfm) / std_jfm if std_jfm != 0 else np.nan

        pre, post = [jan, feb], [mar, apr]
        t_stat, p_val = ttest_ind(pre, post, equal_var=False)
        abs_change = np.mean(post) - np.mean(pre)
        pct_chg = (abs_change / np.mean(pre)) if np.mean(pre) != 0 else np.nan
        sig = "Yes" if p_val < 0.05 else "No"

        may_reg = LinearRegression().fit(np.array([[1], [2], [3], [4]]), np.array([jan, feb, mar, apr]))
        may_forecast = may_reg.predict(np.array([[5]]))[0]
        may_slope, may_intercept, may_r2 = may_reg.coef_[0], may_reg.intercept_, may_reg.score(np.array([[1], [2], [3], [4]]), np.array([jan, feb, mar, apr]))

        results.append({
            "Category": cat,
            "Expense Ratio Apr": ra,
            "Z-Score April": z_apr,
            "Slope": slope,
            "Intercept": intercept,
            "R²": r2,
            "Margin Leverage Score": leverage,
            "T-Test P-Value": p_val,
            "T-Test Significant": sig,
            "May Forecast": may_forecast
        })

    df = pd.DataFrame(results)

    # Table
    table = dash_table.DataTable(
        data=df.to_dict('records'),
        columns=[{"name": i, "id": i} for i in df.columns],
        page_size=15,
        style_table={'overflowX': 'auto'},
        style_cell={'textAlign': 'left'}
    )

    # Charts
    fig1 = px.scatter(df, x="R²", y="Margin Leverage Score", color="Category", title="Margin Leverage vs R²")
    fig2 = px.bar(df.sort_values("Z-Score April", ascending=False).head(10), x="Category", y="Z-Score April", title="Top April Outliers")
    fig3 = px.bar(df.sort_values("Expense Ratio Apr", ascending=False).head(10), x="Category", y="Expense Ratio Apr", title="Top Expense Ratios (April)")

    return html.Div([
        html.H4("📋 Results Table"),
        table,
        html.Br(),
        html.Div([
            dcc.Graph(figure=fig1),
            dcc.Graph(figure=fig2),
            dcc.Graph(figure=fig3)
        ])
    ])

@app.callback(
    Output('output-table', 'children'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def update_output(contents, filename):
    if contents is not None:
        return analyze_pnl(contents, filename)

if __name__ == '__main__':
    app.run_server(debug=True)
