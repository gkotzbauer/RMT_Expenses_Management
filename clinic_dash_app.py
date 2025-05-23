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
        try:
            # Always attempt t-test, even with identical values
            t_stat, p_val = ttest_ind(pre, post, equal_var=False)
            # Handle case where p_val might be NaN due to identical groups
            if np.isnan(p_val):
                # If groups are identical, no significant change
                sig = "No"
            else:
                sig = "Yes" if p_val < 0.05 else "No"
        except:
            t_stat, p_val = np.nan, np.nan
            sig = "No"

        may_reg = LinearRegression().fit(np.array([[1], [2], [3], [4]]), np.array([jan, feb, mar, apr]))
        may_forecast = may_reg.predict(np.array([[5]]))[0]

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
            "T-Test P-Value": p_val,
            "May Forecast Forecasted Expense": may_forecast,
            "Actual Jan": jan,
            "Actual Feb": feb,
            "Actual Mar": mar,
            "Actual Apr": apr
        })

    df = pd.DataFrame(results)
    df["Category"] = df["Category"].astype(str).str.strip()

    headers_to_remove = [
        "600100 Payroll expenses", "610000 Supplies", "613000 Insurance",
        "616000 Employee Benefits", "612000 Professional Fees",
        "620000 Taxes", "650000 Depreciation Expense",
        "Net Operating Income", "Net Income"
    ]
    df = df[~df['Category'].isin(headers_to_remove)]

    def score_row(row):
        score = 0
        reasons = []
        flags = {
            "Low margin leverage": False,
            "High slope (scales with income)": False,
            "Ratio increased": False,
            "April outlier": False,
            "Statistically significant change": False,
            "No issues": False
        }
        
        if row["Marginal Costs vs Income Margin Leverage Score"] < 0.5:
            score += 2
            reasons.append("Low margin leverage")
            flags["Low margin leverage"] = True
            
        if row["Marginal Costs vs Income Slope"] > 0.05:
            score += 2
            reasons.append("High slope (scales with income)")
            flags["High slope (scales with income)"] = True
            
        if row["Expense Ratios % Change in Ratio (Apr vs Jan)"] > 0:
            score += 1
            reasons.append("Ratio increased")
            flags["Ratio increased"] = True
            
        if abs(row["April Z-Score Z_April"]) > 2:
            score += 1
            reasons.append("April outlier")
            flags["April outlier"] = True
            
        if row["T-Test Statistically Significant Change"].strip().lower() == "yes" and not np.isnan(row["T-Test P-Value"]) and row["T-Test P-Value"] < 0.05:
            score += 1
            reasons.append("Statistically significant change")
            flags["Statistically significant change"] = True
            
        if not reasons:
            flags["No issues"] = True
            
        return score, "; ".join(reasons) if reasons else "No issues", flags

    df["Priority Score"] = 0
    df["Action Needed"] = ""
    
    # Initialize action columns
    actions = [
        "Low margin leverage", "No issues", "Ratio increased",
        "April outlier", "Statistically significant change",
        "High slope (scales with income)"
    ]
    for action in actions:
        df[action] = 0
        
    for idx, row in df.iterrows():
        score, flag_text, flags = score_row(row)
        df.at[idx, "Priority Score"] = score
        df.at[idx, "Action Needed"] = flag_text
        
        # Set individual action flags
        for action in actions:
            df.at[idx, action] = 1 if flags[action] else 0

    return df

app.layout = html.Div([
    html.H1("Clinic Expense Analysis Dashboard"),
    dcc.Upload(id='upload-data', children=html.Button('Upload Profit and Loss Excel File'), multiple=False),
    html.Br(),
    html.Div(id='filter-area'),
    html.Br(),
    html.Div(id='output-tables'),
    html.Br(),
    html.Div(id='chart-area'),
    html.Br(),
    html.Button("Download Results", id="download-btn"),
    dcc.Download(id="download-output")
])

@app.callback(
    Output('filter-area', 'children'),
    Input('upload-data', 'contents'),
    prevent_initial_call=True
)
def update_filter(contents):
    if not contents:
        return html.Div()
    
    df = analyze_file(contents)
    categories = sorted(df['Category'].unique())
    
    return html.Div([
        html.Label("Filter by Categories:", style={'fontWeight': 'bold'}),
        dcc.Dropdown(
            id='category-filter',
            options=[{'label': cat, 'value': cat} for cat in categories],
            value=categories,  # All categories selected by default
            multi=True,
            placeholder="Select categories to display..."
        )
    ])

@app.callback(
    Output('output-tables', 'children'),
    Output('chart-area', 'children'),
    Output('download-output', 'data'),
    Input('upload-data', 'contents'),
    Input('category-filter', 'value'),
    State('upload-data', 'filename'),
    Input('download-btn', 'n_clicks'),
    prevent_initial_call=True
)
def update_output(contents, selected_categories, filename, download_clicks):
    if not contents:
        return html.Div("No file uploaded."), None, None

    df = analyze_file(contents)
    
    # Apply category filter
    if selected_categories:
        df_filtered = df[df['Category'].isin(selected_categories)].copy()
    else:
        df_filtered = df.copy()

    results_table = dash_table.DataTable(
        columns=[{"name": i, "id": i} for i in df_filtered.columns],
        data=df_filtered.to_dict('records'),
        style_table={'overflowX': 'auto'},
        style_cell={'textAlign': 'left', 'whiteSpace': 'normal', 'height': 'auto'},
        page_size=20
    )

    priority_chart = px.histogram(df_filtered[df_filtered["Priority Score"] > 0], x="Priority Score", nbins=10)
    priority_chart.update_layout(title="Categories by Priority Score")

    action_counts = df_filtered[
        ["Low margin leverage", "Ratio increased", "April outlier", 
         "Statistically significant change", "High slope (scales with income)"]
    ].sum().sort_values(ascending=False).reset_index()
    action_counts.columns = ["Action", "Count"]
    action_chart = px.bar(action_counts, x="Action", y="Count", title="Action Item Frequency")

    categories_by_action = []
    for action in action_counts["Action"]:
        matched = df_filtered[df_filtered[action] == 1]["Category"].tolist()
        categories_by_action.append({"Action Item": action, "Categories": "; ".join(matched)})
    action_table = dash_table.DataTable(
        columns=[{"name": i, "id": i} for i in ["Action Item", "Categories"]],
        data=categories_by_action,
        style_cell={'textAlign': 'left', 'whiteSpace': 'normal', 'height': 'auto'}
    )

    top_priority = df_filtered[df_filtered["Priority Score"] > 0].groupby("Priority Score")["Category"].apply(lambda x: "; ".join(x)).reset_index()
    top_priority_table = dash_table.DataTable(
        columns=[{"name": "Priority Score", "id": "Priority Score"}, {"name": "Category", "id": "Category"}],
        data=top_priority.to_dict("records"),
        style_cell={'textAlign': 'left', 'whiteSpace': 'normal', 'height': 'auto'}
    )

    # Create category explanation table
    def get_potential_action(reasons):
        actions = []
        if "Low margin leverage" in reasons:
            actions.append("Review cost structure and negotiate better rates with vendors")
        if "High slope (scales with income)" in reasons:
            actions.append("Implement cost controls to prevent expenses from growing faster than revenue")
        if "Ratio increased" in reasons:
            actions.append("Investigate why expense ratio increased and implement cost reduction measures")
        if "April outlier" in reasons:
            actions.append("Analyze April expenses for unusual items and verify accuracy")
        if "Statistically significant change" in reasons:
            actions.append("Conduct detailed expense analysis to understand the significant cost shift")
        if "No issues" in reasons or not actions:
            actions.append("Continue monitoring; no immediate action required")
        return "; ".join(actions)

    category_explanations = []
    for _, row in df_filtered.iterrows():
        potential_action = get_potential_action(row["Action Needed"])
        category_explanations.append({
            "Category": row["Category"],
            "Reasons for Priority Score": row["Action Needed"] if row["Action Needed"] != "No issues" else "No issues detected",
            "Potential Action": potential_action
        })
    
    category_explanation_table = dash_table.DataTable(
        columns=[{"name": i, "id": i} for i in ["Category", "Reasons for Priority Score", "Potential Action"]],
        data=category_explanations,
        style_cell={'textAlign': 'left', 'whiteSpace': 'normal', 'height': 'auto'},
        style_data={'whiteSpace': 'normal', 'height': 'auto'},
        css=[{
            'selector': '.dash-cell div.dash-cell-value',
            'rule': 'display: inline; white-space: inherit; overflow: inherit; text-overflow: inherit;'
        }]
    )

    if download_clicks:
        output = BytesIO()
        df_filtered.to_excel(output, index=False)
        output.seek(0)
        return dash_table.DataTable(), None, dcc.send_bytes(output.read(), "Clinic_Expense_Results.xlsx")

    return (
        html.Div([html.H3("Results Table"), results_table]),
        html.Div([
            html.H3("Categories by Priority Score"),
            dcc.Graph(figure=priority_chart),
            html.H3("Top Categories per Priority Score"),
            top_priority_table,
            html.H3("Category Priority Explanations"),
            category_explanation_table,
            html.Div([
                html.H4('Priority Score Definitions'),
                html.Table([
                    html.Tr([html.Td('Priority Score 1–2:'), html.Td('Minor issues, monitor periodically')]),
                    html.Tr([html.Td('Priority Score 3–4:'), html.Td('Moderate issues, consider corrective actions')]),
                    html.Tr([html.Td('Priority Score 5+:'), html.Td('Critical attention needed, likely inefficiencies')])
                ])
            ]),
            html.H3("Action Item Frequency"),
            dcc.Graph(figure=action_chart),
            html.H3("Categories Assigned to Each Action"),
            action_table,
            html.Div([
                html.H4('Action Item Definitions'),
                html.Table([
                    html.Tr([html.Td('Low margin leverage:'), html.Td('Cost does not scale efficiently with income')]),
                    html.Tr([html.Td('Ratio increased:'), html.Td('Expense ratio increased from Jan to Apr')]),
                    html.Tr([html.Td('April outlier:'), html.Td('April expense unusually high or low')]),
                    html.Tr([html.Td('Statistically significant change:'), html.Td('Change from Jan–Feb to Mar–Apr is statistically significant')]),
                    html.Tr([html.Td('High slope (scales with income):'), html.Td('Expense increases rapidly with income')]),
                    html.Tr([html.Td('No issues:'), html.Td('No margin concerns detected')])
                ])
            ])
        ]),
        None
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, debug=True)
