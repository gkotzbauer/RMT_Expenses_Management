
# 🏥 Clinic Expense Analyzer (Dash App)

This Dash app allows clinic managers to upload a P&L Excel file and receive automated analysis of:

- Expense Ratios
- Marginal Costs vs Income
- Z-Score Outliers (April)
- T-Test Results (Jan–Feb vs Mar–Apr)
- May Forecast

Includes interactive tables and charts for insights.

---

## 🚀 Deployment Instructions (Render)

### ✅ 1. Files in this Repository

| File | Purpose |
|------|---------|
| `clinic_dash_app.py` | Main Dash app script |
| `requirements.txt`   | Python package dependencies |
| `render.yaml`        | Configuration for Render deployment |

---

### 🌐 2. Deploying to Render

1. Go to [https://render.com](https://render.com)
2. Create a **free account** and click **"New Web Service"**
3. Connect your GitHub repo
4. Select this repository
5. Render will auto-detect `render.yaml` and configure your app

---

### ⚙️ Recommended Render Settings

| Setting | Value |
|--------|-------|
| Environment | `Python` |
| Instance Type | `Free` or `Starter` |
| Build Command | `pip install -r requirements.txt` |
| Start Command | `python clinic_dash_app.py` |

---

## 🧾 P&L File Format

To ensure the app works, the uploaded Excel file must:
- Contain a worksheet named `Profit and Loss`
- Include rows labeled `Jan`, `Feb`, `Mar`, `Apr`
- Include a row named **`Total Income`** for monthly income (used in calculations)

---

## 📈 Visual Outputs

Once uploaded, the app displays:
- 📊 Expense analysis table
- 📈 Margin Leverage vs R² (scatter)
- 🔍 Z-score outliers (bar)
- 💰 Top Expense Ratios (bar)

---

## 🧰 Local Development

To run locally:

```bash
pip install -r requirements.txt
python clinic_dash_app.py
```

Visit `http://127.0.0.1:8050` in your browser.

---

## 🤝 Contact

For help or customization, reach out to your internal data analyst or development team.
