
# -*- coding: utf-8 -*-
"""
Clinic P&L Analyzer (Full Processing Dash App)

Steps:
1. Upload raw P&L file (.xlsx with 'Profit and Loss' sheet)
2. Automatically extract expenses and income
3. Compute analysis:
   - Expense ratios
   - Slope/intercept/R²
   - Z-score
   - T-test
   - Forecast
   - Priority scoring
4. Display results: tables + charts
5. Allow user to download final analysis file
"""

import pandas as pd
import numpy as np
import base64
from io import BytesIO
from dash import Dash, dcc, html, dash_table, Input, Output, State
from sklearn.linear_model import LinearRegression
from scipy.stats import ttest_ind
import plotly.express as px

app = Dash(__name__)
server = app.server

def analyze_file(contents):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    pnl_df = pd.read_excel(BytesIO(decoded), sheet_name="Profit and Loss")

    expenses_start = pnl_df[pnl_df.iloc[:, 0] == "Expenses"].index[0]
    expense_data = pnl_df.iloc[expenses_start + 1:].copy()
    expense_data.columns = ['Category', 'Jan', 'Feb', 'Mar', 'Apr', 'Total']
    expense_data = expense_data[['Category', 'Jan', 'Feb', 'Mar', 'Apr']].dropna(subset=['Category'])
    expense_data = expense_data[~expense_data['Category'].str.contains("Total", na=False)]
    expense_data = expense_data.reset_index(drop=True)

    for col in ['Jan', 'Feb', 'Mar', 'Apr']:
        expense_data[col] = pd.to_numeric(expense_data[col], errors='coerce').fillna(0)

    monthly_income = {
        'Jan': pnl_df.iloc[10, 1],
        'Feb': pnl_df.iloc[10, 2],
        'Mar': pnl_df.iloc[10, 3],
        'Apr': pnl_df.iloc[10, 4]
    }

    results = []
    for _, row in expense_data.iterrows():
        cat, jan, feb, mar, apr = row['Category'], row['Jan'], row['Feb'], row['Mar'], row['Apr']
        rj, rf, rm, ra = jan / monthly_income['Jan'], feb / monthly_income['Feb'], mar / monthly_income['Mar'], apr / monthly_income['Apr']
        pct_cost = (apr - jan) / jan if jan else np.nan
        pct_ratio = (ra - rj) / rj if rj else np.nan

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
        pct_chg = abs_change / np.mean(pre) if np.mean(pre) else np.nan
        sig = "Yes" if p_val < 0.05 else "No"

        may_reg = LinearRegression().fit(np.array([[1], [2], [3], [4]]), np.array([jan, feb, mar, apr]))
        may_forecast = may_reg.predict(np.array([[5]]))[0]
        may_slope, may_intercept, may_r2 = may_reg.coef_[0], may_reg.intercept_, may_reg.score(np.array([[1], [2], [3], [4]]), np.array([jan, feb, mar, apr]))

        results.append({
            "Category": cat,
            "Expense Ratios Ratio_Jan": rj,
            "Expense Ratios Ratio_Feb": rf,
            "Expense Ratios Ratio_Mar": rm,
            "Expense Ratios Ratio_April": ra,
            "Expense Ratios % Change in Cost (Apr vs Jan)": pct_cost,
            "Expense Ratios % Change in Ratio (Apr vs Jan)": pct_ratio,
            "Marginal Costs vs Income Slope": slope,
            "Marginal Costs vs Income Intercept": intercept,
            "Marginal Costs vs Income R²": r2,
            "Marginal Costs vs Income Inverse Slope Score": inv_score,
            "Marginal Costs vs Income Margin Leverage Score": leverage,
            "April Z-Score Z_April": z_apr,
            "T-Test Statistically Significant Change": sig,
            "May Forecast Forecasted Expense": may_forecast,
            "Actual Jan": jan,
            "Actual Feb": feb,
            "Actual Mar": mar,
            "Actual Apr": apr
        })

    df = pd.DataFrame(results)

    def score_row(row):
        score = 0
        reasons = []
        if row["Marginal Costs vs Income Margin Leverage Score"] < 0.5:
            score += 2; reasons.append("Low margin leverage")
        if row["Marginal Costs vs Income Slope"] > 0.05:
            score += 2; reasons.append("High slope (scales with income)")
        if row["Expense Ratios % Change in Ratio (Apr vs Jan)"] > 0:
            score += 1; reasons.append("Ratio increased")
        if abs(row["April Z-Score Z_April"]) > 2:
            score += 1; reasons.append("April outlier")
        if row["T-Test Statistically Significant Change"] == "Yes":
            score += 1; reasons.append("Statistically significant change")
        return score, "; ".join(reasons) if reasons else "No issues"

    df["Priority Score"] = 0
    df["Action Needed"] = ""
    for idx, row in df.iterrows():
        score, flag = score_row(row)
        df.at[idx, "Priority Score"] = score
        df.at[idx, "Action Needed"] = flag

    action_flags = [
        "Low margin leverage", "No issues", "Ratio increased",
        "April outlier", "Statistically significant change",
        "High slope (scales with income)"
    ]
    for flag in action_flags:
        df[flag] = df["Action Needed"].apply(lambda x: 1 if flag in str(x) else 0)

    return df

app.layout = html.Div([
    html.H2("📊 Full Clinic Margin Analyzer"),
    dcc.Upload(id='upload-data', children=html.Div(['📁 Upload your raw P&L Excel file']), style={
        'width': '100%', 'height': '60px', 'lineHeight': '60px',
        'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '5px',
        'textAlign': 'center', 'margin': '10px'
    }, multiple=False),
    html.Button("Download Processed Excel File", id="btn_download_excel"),
    dcc.Download(id="download-dataframe-xlsx"),
    html.Div(id='data-table'),
    dcc.Graph(id="priority-chart"),
    dcc.Graph(id="action-chart")
])

@app.callback(
    Output('data-table', 'children'),
    Output('priority-chart', 'figure'),
    Output('action-chart', 'figure'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def update_output(contents, filename):
    if contents is None:
        return None, {}, {}

    df = analyze_file(contents)

    table = dash_table.DataTable(
        data=df.to_dict("records"),
        columns=[{"name": i, "id": i} for i in df.columns],
        page_size=15,
        style_table={'overflowX': 'auto'}
    )

    
filtered_df = df[df["Priority Score"] > 0]
score_fig = px.histogram(filtered_df, x="Priority Score", title="Categories by Priority Score")
score_fig.update_layout(
    annotations=[dict(
        text="This chart shows how many expense categories fall into each priority level (excluding 0). "
             "Higher scores indicate stronger margin optimization opportunities.",
        xref="paper", yref="paper", showarrow=False, x=0, y=1.08, font=dict(size=12)
    )]
)

    actions = ["Low margin leverage", "No issues", "Ratio increased",
               "April outlier", "Statistically significant change", "High slope (scales with income)"]
    action_counts = df[actions].sum().reset_index()
    action_counts.columns = ["Action", "Count"]
    action_fig = px.bar(action_counts, x="Action", y="Count", title="Action Item Frequency")
action_fig.update_layout(
    annotations=[dict(
        text="This chart counts how often each action flag is assigned.\n"
             "Low margin leverage: Low efficiency at higher income.\n"
             "Ratio increased: April cost rose relative to income.\n"
             "April outlier: Cost in April was unusually high.\n"
             "Statistically significant change: Confirmed real change in cost behavior.\n"
             "High slope: Cost increases quickly with income.",
        xref="paper", yref="paper", showarrow=False, x=0, y=1.05, align='left', font=dict(size=12)
    )]
)

    
    # Table: Categories by Action Needed
    action_needed_table = dash_table.DataTable(
        data=df[["Category", "Action Needed"]].to_dict("records"),
        columns=[{"name": "Category", "id": "Category"}, {"name": "Action Needed", "id": "Action Needed"}],
        page_size=10,
        style_table={"marginTop": "20px"}
    )

    # Table: Categories in Top 3 Priority Scores
    top_scores_df = df[df["Priority Score"] >= df["Priority Score"].max() - 2]
    grouped = top_scores_df.groupby("Priority Score")["Category"].apply(lambda x: "; ".join(x)).reset_index()
    grouped.columns = ["Priority Score", "Categories"]
    top_score_table = dash_table.DataTable(
        data=grouped.to_dict("records"),
        columns=[{"name": i, "id": i} for i in grouped.columns],
        page_size=10,
        style_table={"marginTop": "20px"}
    )

    return html.Div([
        table,
        html.H4("📋 Categories by Action Needed"),
        action_needed_table,
        html.H4("🏆 Categories in Top 3 Priority Scores"),
        top_score_table
    ]), score_fig, action_fig


@app.callback(
    Output("download-dataframe-xlsx", "data"),
    Input("btn_download_excel", "n_clicks"),
    State("upload-data", "contents"),
    prevent_initial_call=True
)
def download_xlsx(n_clicks, contents):
    if contents is None:
        return None
    df = analyze_file(contents)
    return dcc.send_data_frame(df.to_excel, "RMT_Visual_Clinic_Analysis_With_Priorities_Expanded.xlsx", index=False, sheet_name="Visual Summary")

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 8050))
    app.run(host="0.0.0.0", port=port, debug=True)
