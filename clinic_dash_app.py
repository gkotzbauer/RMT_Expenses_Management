import pandas as pd
import numpy as np
import base64
from io import BytesIO
from dash import Dash, dcc, html, dash_table, Input, Output, State, ALL
from dash.exceptions import PreventUpdate
from sklearn.linear_model import LinearRegression
from scipy.stats import ttest_ind
import plotly.express as px

# Initialize the app with specific configurations
app = Dash(
    __name__,
    suppress_callback_exceptions=True,
    meta_tags=[{'name': 'viewport', 'content': 'width=device-width, initial-scale=1'}],
    title='Clinic Expense Analysis Dashboard'
)

# Configure the server
server = app.server
app.config.suppress_callback_exceptions = True

def create_layout():
    return html.Div([
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

app.layout = create_layout()

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
    Input('upload-data', 'contents'),
    State('upload-data', 'filename'),
    prevent_initial_call=True
)
def update_output_initial(contents, filename):
    if not contents:
        raise PreventUpdate
    
    df = analyze_file(contents)
    return create_output_content(df)

@app.callback(
    Output('output-tables', 'children', allow_duplicate=True),
    Output('chart-area', 'children', allow_duplicate=True),
    Input('category-filter', 'value'),
    State('upload-data', 'contents'),
    prevent_initial_call=True
)
def update_output_filtered(selected_categories, contents):
    if not contents or not selected_categories:
        raise PreventUpdate
    
    df = analyze_file(contents)
    df = df[df['Category'].isin(selected_categories)]
    return create_output_content(df)

def create_output_content(df):
    tables = []
    charts = []
    
    # Priority Score Table
    priority_df = df.sort_values('Priority Score', ascending=False)
    tables.append(html.Div([
        html.H3("Priority Score Analysis"),
        dash_table.DataTable(
            data=priority_df.to_dict('records'),
            columns=[{"name": i, "id": i} for i in priority_df.columns],
            style_table={'overflowX': 'auto'},
            style_cell={
                'textAlign': 'left',
                'padding': '10px',
                'whiteSpace': 'normal',
                'height': 'auto',
            },
            style_header={
                'backgroundColor': 'rgb(230, 230, 230)',
                'fontWeight': 'bold'
            },
            style_data_conditional=[
                {
                    'if': {'filter_query': '{Priority Score} > 0'},
                    'backgroundColor': 'rgba(255, 0, 0, 0.1)'
                }
            ]
        )
    ]))
    
    # Action Needed Table
    action_df = df[df['Action Needed'] != 'No issues'].sort_values('Priority Score', ascending=False)
    if not action_df.empty:
        tables.append(html.Div([
            html.H3("Items Requiring Action"),
            dash_table.DataTable(
                data=action_df.to_dict('records'),
                columns=[{"name": i, "id": i} for i in action_df.columns],
                style_table={'overflowX': 'auto'},
                style_cell={
                    'textAlign': 'left',
                    'padding': '10px',
                    'whiteSpace': 'normal',
                    'height': 'auto',
                },
                style_header={
                    'backgroundColor': 'rgb(230, 230, 230)',
                    'fontWeight': 'bold'
                }
            )
        ]))
    
    # Charts
    # Priority Score Distribution
    fig_priority = px.histogram(
        df, 
        x='Priority Score',
        title='Distribution of Priority Scores',
        nbins=10
    )
    charts.append(dcc.Graph(figure=fig_priority))
    
    # Action Categories
    action_counts = df['Action Needed'].value_counts()
    fig_actions = px.pie(
        values=action_counts.values,
        names=action_counts.index,
        title='Distribution of Required Actions'
    )
    charts.append(dcc.Graph(figure=fig_actions))
    
    return html.Div(tables), html.Div(charts)

@app.callback(
    Output('download-output', 'data'),
    Input('download-btn', 'n_clicks'),
    State('upload-data', 'contents'),
    State('category-filter', 'value'),
    prevent_initial_call=True
)
def download_results(download_clicks, contents, selected_categories):
    if not download_clicks or not contents:
        return None
    
    df = analyze_file(contents)
    
    # Apply category filter
    if selected_categories and len(selected_categories) > 0:
        df_filtered = df[df['Category'].isin(selected_categories)].copy()
    else:
        df_filtered = df.copy()
    
    output = BytesIO()
    df_filtered.to_excel(output, index=False)
    output.seek(0)
    return dcc.send_bytes(output.read(), "Clinic_Expense_Results.xlsx")

if __name__ == "__main__":
    app.run_server(debug=True, host="0.0.0.0", port=10000)
