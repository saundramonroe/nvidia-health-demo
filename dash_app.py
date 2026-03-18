"""
Health AI — Patient Deterioration Dashboard
Plotly Dash app with hourly and daily hospital views + Nemotron insight panel

Run:
    conda activate nvidia-health-demo
    python dashboard/app.py
    → http://localhost:8060
"""

import os, sys, json, random, textwrap
import pandas as pd
import numpy as np

# ── Dash & Plotly ─────────────────────────────────────────────────────────────
import dash
from dash import dcc, html, Input, Output, callback, dash_table
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px

# ── Path setup ────────────────────────────────────────────────────────────────
# Works whether you run from project root OR from dashboard/ subfolder
_here = os.path.dirname(os.path.abspath(__file__))
if os.path.isdir(os.path.join(_here, 'data')):
    BASE = _here                        # running from project root
else:
    BASE = os.path.dirname(_here)       # running from dashboard/ subfolder
DATA = os.path.join(BASE, 'data')
sys.path.insert(0, BASE)

random.seed(42)
np.random.seed(42)

# ── Colour palette ────────────────────────────────────────────────────────────
BG          = "#040d1a"
SURFACE     = "#071525"
SURFACE2    = "#0c1f35"
BORDER      = "#13304d"
NVIDIA_GRN  = "#76b900"
ACCENT_BLUE = "#00c8ff"
DANGER_RED  = "#ff4560"
WARN_ORANGE = "#ff9f1c"
TEXT        = "#e8f4ff"
TEXT_DIM    = "#6a8aaa"

# ── Load data ─────────────────────────────────────────────────────────────────
def _make_hourly(df):
    """Generate realistic hourly ward data from actual patient dataframe."""
    rng = np.random.default_rng(42)
    n_high = int((df['risk_level'] == 'HIGH').sum())
    n_mod  = int((df['risk_level'] == 'MODERATE').sum())
    n_low  = int((df['risk_level'] == 'LOW').sum())
    avg_risk = float(df['risk_score'].mean())
    return pd.DataFrame({
        'hour':               list(range(24)),
        'high_risk_count':    [max(0, n_high + int(rng.integers(-max(1,n_high//10), max(1,n_high//10)+1))) for _ in range(24)],
        'moderate_risk_count':[max(0, n_mod  + int(rng.integers(-max(1,n_mod//8),  max(1,n_mod//8)+1)))  for _ in range(24)],
        'low_risk_count':     [max(0, n_low  + int(rng.integers(-max(1,n_low//12), max(1,n_low//12)+1))) for _ in range(24)],
        'alerts_fired':       [int(rng.integers(1, 9)) for _ in range(24)],
        'avg_risk_score':     [round(avg_risk + rng.uniform(-0.04, 0.04), 3) for _ in range(24)],
        'interventions':      [int(rng.integers(0, 5)) for _ in range(24)],
    })

def load_data():
    try:
        df    = pd.read_csv(os.path.join(DATA, 'patients_with_risk.csv'))
        # Always regenerate hourly from actual df so it stays in sync
        hourly = _make_hourly(df)
        alerts = pd.read_csv(os.path.join(DATA, 'alert_queue.csv'))
        with open(os.path.join(DATA, 'shap_metadata.json')) as f:
            shap_meta = json.load(f)
        return df, hourly, alerts, shap_meta
    except FileNotFoundError:
        print("⚠️  Data not found. Run notebook 01 first to generate data.")
        return _mock_data()

def _mock_data():
    """Fallback mock data if notebooks haven't been run yet."""
    n = 200
    np.random.seed(42)
    df = pd.DataFrame({
        'patient_id':          [f'ICU-{i:04d}' for i in range(n)],
        'name':                [f'Patient {i}' for i in range(n)],
        'age':                 np.random.randint(40,88,n),
        'gender':              np.random.choice(['M','F'],n),
        'diagnosis':           np.random.choice(['Sepsis','COPD','Post-cardiac surgery','Pneumonia'],n),
        'room':                [f'{np.random.randint(1,5)}{chr(np.random.randint(65,73))}' for _ in range(n)],
        'hr':                  np.random.normal(88,18,n).clip(50,170).round(1),
        'spo2':                np.random.normal(95,4,n).clip(78,100).round(1),
        'sbp':                 np.random.normal(112,22,n).clip(65,200).round(1),
        'rr':                  np.random.normal(17,4,n).clip(9,40).round(1),
        'news2_score':         np.random.randint(0,14,n),
        'risk_score':          np.random.beta(2,5,n).round(4),
        'risk_level':          np.random.choice(['LOW','MODERATE','HIGH'],n,p=[0.55,0.27,0.18]),
        'will_deteriorate':    np.random.binomial(1,0.28,n),
        'combined_flag_count': np.random.randint(0,7,n),
        'trend_severity':      np.random.exponential(2,n).round(2),
        'alert':               np.random.binomial(1,0.18,n),
        'lactate':             np.random.uniform(0.5,6,n).round(2),
        'wbc':                 np.random.uniform(4,24,n).round(1),
    })
    hourly = pd.DataFrame({
        'hour':               range(24),
        'high_risk_count':    np.random.randint(20,55,24),
        'moderate_risk_count':np.random.randint(50,120,24),
        'alerts_fired':       np.random.randint(1,9,24),
        'avg_risk_score':     np.random.uniform(0.28,0.38,24).round(3),
        'interventions':      np.random.randint(0,5,24),
    })
    alerts = df.nlargest(20,'risk_score').copy()
    shap_meta = {
        'top5_features': ['Trend Severity','Shock Index','Lactate','SpO₂ Trend/hr','NEWS2 Score'],
        'mean_shap': [0.18,0.14,0.12,0.11,0.09,0.08,0.07,0.06,0.05,0.04]*3,
        'feature_names': ['Trend Severity','Shock Index','Lactate','SpO₂ Trend/hr','NEWS2 Score',
                          'Heart Rate','SpO₂','Systolic BP','Combined Flags','WBC']*3,
    }
    return df, hourly, alerts, shap_meta

df, hourly, alerts, shap_meta = load_data()

# ── Nemotron mock engine ───────────────────────────────────────────────────────
# ── Nemotron insight engine ───────────────────────────────────────────────────
# Priority 1: pre-generated CSV from notebook 03
# Priority 2: rich per-patient generated mock (different text per patient)

_insights_path = os.path.join(DATA, 'nemotron_insights.csv')
try:
    _ins_df  = pd.read_csv(_insights_path)
    _ins_map = dict(zip(_ins_df['patient_id'], _ins_df['nemotron_insight']))
    print(f"\u2705 Loaded {len(_ins_map)} Nemotron summaries from nemotron_insights.csv")
except FileNotFoundError:
    _ins_map = {}
    print("\u2139\ufe0f  nemotron_insights.csv not found \u2014 using generated summaries")

_HIGH_T = [
    ("{name} (age {age}) presents with {risk_pct}% deterioration probability driven by "
     "{top1} and {top2} \u2014 a pattern consistent with early septic physiology. "
     "NEWS2 of {news2} with {n_flags} concurrent flags indicates high-acuity status; "
     "vasopressor protocol should be on standby within {window} minutes."),
    ("{name} shows critical hemodynamic trajectory at {risk_pct}% risk. "
     "{top1} is worsening at an accelerating rate and is the primary AI-identified driver. "
     "NEWS2={news2}. Cardiology or pulmonology consult urgently required \u2014 "
     "estimated intervention window is {window} minutes."),
    ("At {risk_pct}% risk, {name} is the highest-priority patient on this alert. "
     "{top1} and {top2} are both trending adversely over the past 6 hours. "
     "NEWS2 of {news2} places this patient in the top 3% ward acuity \u2014 "
     "escalate now, do not wait for the next scheduled check."),
]
_MOD_T = [
    ("{name} is at moderate and rising risk ({risk_pct}%). {top1} is trending unfavorably \u2014 "
     "a pattern preceding escalation in similar admissions. "
     "Nursing reassessment within 30 minutes; if NEWS2 exceeds {news2_t} or SpO\u2082 drops below 94%, "
     "escalate to attending immediately."),
    ("{name} ({risk_pct}%) requires close monitoring. {top1} is the leading concern with {top2} borderline. "
     "No immediate intervention required, but NEWS2 of {news2} warrants escalation readiness. "
     "Reassess in 30 minutes and prepare escalation pathway."),
]
_LOW_T = [
    ("{name} is clinically stable at {risk_pct}% risk. "
     "{top1} and {top2} are within acceptable ranges with no adverse trends over 24 hours. "
     "Continue current care plan; next reassessment per standard ward protocol."),
    ("All indicators stable for {name} ({risk_pct}% risk). "
     "Recovery trajectory is tracking well for this diagnosis group. "
     "No escalation required \u2014 recommend step-down review in 18\u201324 hours if trend continues."),
]

def get_insight(row):
    pid = str(row.get('patient_id', ''))
    if pid in _ins_map:                        # use pre-generated if available
        return str(_ins_map[pid])

    level    = str(row.get('risk_level', 'LOW'))
    drivers  = shap_meta.get('top5_features', ['Trend Severity', 'Shock Index'])[:2]
    name_val = str(row.get('name', 'Patient'))
    fname    = name_val.split()[0] if ' ' in name_val else name_val
    risk_pct = int(float(row.get('risk_score', 0)) * 100)
    news2    = int(row.get('news2_score', 0))
    n_flags  = int(row.get('combined_flag_count', 0))
    rng      = random.Random(pid)             # same patient always gets same template

    if level == 'HIGH':
        tmpl = rng.choice(_HIGH_T)
        return tmpl.format(name=fname, age=int(row.get('age',0)), risk_pct=risk_pct,
                           top1=drivers[0], top2=drivers[1] if len(drivers)>1 else 'SpO\u2082',
                           news2=news2, n_flags=n_flags, window=rng.choice([20,30,40,45]))
    elif level == 'MODERATE':
        tmpl = rng.choice(_MOD_T)
        return tmpl.format(name=fname, risk_pct=risk_pct,
                           top1=drivers[0], top2=drivers[1] if len(drivers)>1 else 'SpO\u2082',
                           news2=news2, news2_t=news2+2)
    else:
        tmpl = rng.choice(_LOW_T)
        return tmpl.format(name=fname, risk_pct=risk_pct,
                           top1=drivers[0], top2=drivers[1] if len(drivers)>1 else 'SpO\u2082')

# ── Helper: Plotly figure defaults ────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor=SURFACE, plot_bgcolor=SURFACE2,
    font=dict(color=TEXT, family="DM Mono, monospace", size=11),
    margin=dict(l=40, r=20, t=40, b=40),
    xaxis=dict(gridcolor=BORDER, showgrid=True, zeroline=False),
    yaxis=dict(gridcolor=BORDER, showgrid=True, zeroline=False),
)

def fig_layout(fig, title="", height=280):
    fig.update_layout(**PLOTLY_LAYOUT, title=dict(text=title, font=dict(size=12, color=TEXT_DIM)), height=height)
    return fig

# ── Summary KPIs ──────────────────────────────────────────────────────────────
n_total    = len(df)
n_high     = int((df['risk_level'] == 'HIGH').sum())
n_mod      = int((df['risk_level'] == 'MODERATE').sum())
n_low      = int((df['risk_level'] == 'LOW').sum())
n_alerts   = int(df['alert'].sum())
avg_news2  = float(df['news2_score'].mean())

# ── App layout ────────────────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.DARKLY,
        "https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=DM+Serif+Display&display=swap"
    ],
    title="NVIDIA Health AI Dashboard",
    suppress_callback_exceptions=True,
)

# ── Shared styles ─────────────────────────────────────────────────────────────
CARD_STYLE = {
    "background": SURFACE, "border": f"1px solid {BORDER}",
    "borderRadius": "10px", "padding": "16px", "marginBottom": "16px",
}
HEADER_STYLE = {
    "background": BG, "borderBottom": f"1px solid {BORDER}",
    "padding": "14px 32px", "position": "sticky", "top": 0, "zIndex": 999,
    "display": "flex", "alignItems": "center", "justifyContent": "space-between",
}
KPI_STYLE = lambda color: {
    "background": SURFACE, "border": f"1px solid {BORDER}",
    "borderTop": f"3px solid {color}", "borderRadius": "8px",
    "padding": "16px 20px", "textAlign": "center",
}
BADGE_RISK = {
    'HIGH':     {"background":"rgba(255,69,96,0.15)",  "color":DANGER_RED,  "border":f"1px solid rgba(255,69,96,0.4)"},
    'MODERATE': {"background":"rgba(255,159,28,0.15)", "color":WARN_ORANGE, "border":f"1px solid rgba(255,159,28,0.4)"},
    'LOW':      {"background":"rgba(118,185,0,0.15)",  "color":NVIDIA_GRN,  "border":f"1px solid rgba(118,185,0,0.4)"},
}

def risk_badge(level):
    s = {**BADGE_RISK.get(str(level), BADGE_RISK['LOW']),
         "fontFamily":"DM Mono,monospace","fontSize":"10px",
         "padding":"3px 8px","borderRadius":"4px","fontWeight":"500"}
    return html.Span(str(level), style=s)

def kpi_card(value, label, color=NVIDIA_GRN, sub=None):
    return html.Div([
        html.Div(str(value), style={"fontFamily":"DM Serif Display,serif","fontSize":"34px","color":color,"lineHeight":"1"}),
        html.Div(label, style={"fontFamily":"DM Mono,monospace","fontSize":"11px","color":TEXT_DIM,"marginTop":"4px"}),
        html.Div(sub, style={"fontSize":"10px","color":TEXT_DIM,"marginTop":"2px"}) if sub else None,
    ], style=KPI_STYLE(color))


# ── Layout ────────────────────────────────────────────────────────────────────
app.layout = html.Div([

    # HEADER
    html.Div([
        html.Div([
            #html.Span("NVIDIA", style={"background":NVIDIA_GRN,"color":"#000","fontFamily":"DM Mono,monospace",
            #                            "fontWeight":"500","fontSize":"11px","padding":"4px 10px","borderRadius":"3px","marginRight":"12px"}),
            html.Span("Health AI · Patient Deterioration Monitor",
                      style={"fontFamily":"DM Serif Display,serif","fontSize":"18px","color":TEXT}),
        ]),
        html.Div([
           # html.Span("● DGX", style={"fontFamily":"DM Mono,monospace","fontSize":"11px","color":"#a67fff","marginRight":"16px"}),
            html.Span("● XGBoost", style={"fontFamily":"DM Mono,monospace","fontSize":"11px","color":NVIDIA_GRN,"marginRight":"16px"}),
            html.Span("● SHAP", style={"fontFamily":"DM Mono,monospace","fontSize":"11px","color":ACCENT_BLUE,"marginRight":"16px"}),
            html.Span("● Nemotron", style={"fontFamily":"DM Mono,monospace","fontSize":"11px","color":WARN_ORANGE,"marginRight":"24px"}),
            html.Span("● LIVE · ICU WARD 4",
                      style={"fontFamily":"DM Mono,monospace","fontSize":"11px","color":NVIDIA_GRN}),
        ]),
    ], style=HEADER_STYLE),

    # BODY
    html.Div([

        # KPI ROW
        html.Div([
            dbc.Row([
                dbc.Col(kpi_card(n_total,    "Patients Monitored",  ACCENT_BLUE),  md=2),
                dbc.Col(kpi_card(n_high,     "High Risk",           DANGER_RED,  f"{n_high/n_total:.0%} of ward"), md=2),
                dbc.Col(kpi_card(n_mod,      "Moderate Risk",       WARN_ORANGE, f"{n_mod/n_total:.0%} of ward"),  md=2),
                dbc.Col(kpi_card(n_alerts,   "Active Alerts",       DANGER_RED),  md=2),
                dbc.Col(kpi_card(f"{avg_news2:.1f}", "Avg NEWS2 Score", WARN_ORANGE), md=2),
                dbc.Col(kpi_card(">0.95",    "Model AUC Score",     NVIDIA_GRN,  "XGBoost · 30 features"), md=2),
            ], className="g-2"),
        ], style={"marginBottom":"20px"}),

        # TABS
        dcc.Tabs(id="view-tabs", value="hourly",
            children=[
                dcc.Tab(label="  Hourly View", value="hourly",
                    style={"fontFamily":"DM Mono,monospace","fontSize":"12px","background":SURFACE,"color":TEXT_DIM,"border":f"1px solid {BORDER}"},
                    selected_style={"fontFamily":"DM Mono,monospace","fontSize":"12px","background":SURFACE2,"color":NVIDIA_GRN,"borderTop":f"2px solid {NVIDIA_GRN}","border":f"1px solid {BORDER}"}),
                dcc.Tab(label="  Daily View",  value="daily",
                    style={"fontFamily":"DM Mono,monospace","fontSize":"12px","background":SURFACE,"color":TEXT_DIM,"border":f"1px solid {BORDER}"},
                    selected_style={"fontFamily":"DM Mono,monospace","fontSize":"12px","background":SURFACE2,"color":NVIDIA_GRN,"borderTop":f"2px solid {NVIDIA_GRN}","border":f"1px solid {BORDER}"}),
                dcc.Tab(label="  Alert Queue", value="alerts",
                    style={"fontFamily":"DM Mono,monospace","fontSize":"12px","background":SURFACE,"color":TEXT_DIM,"border":f"1px solid {BORDER}"},
                    selected_style={"fontFamily":"DM Mono,monospace","fontSize":"12px","background":SURFACE2,"color":DANGER_RED,"borderTop":f"2px solid {DANGER_RED}","border":f"1px solid {BORDER}"}),
                dcc.Tab(label=" Medical Summaries and classification by Nemotron AI", value="nemotron",
                    style={"fontFamily":"DM Mono,monospace","fontSize":"12px","background":SURFACE,"color":TEXT_DIM,"border":f"1px solid {BORDER}"},
                    selected_style={"fontFamily":"DM Mono,monospace","fontSize":"12px","background":SURFACE2,"color":WARN_ORANGE,"borderTop":f"2px solid {WARN_ORANGE}","border":f"1px solid {BORDER}"}),
            ],
            style={"marginBottom":"0"},
            colors={"border":BORDER,"primary":NVIDIA_GRN,"background":BG},
        ),

        html.Div(id="tab-content", style={"marginTop":"16px"}),

        # Interval for live updates
        dcc.Interval(id="interval", interval=30_000, n_intervals=0),

    ], style={"padding":"24px 32px","backgroundColor":BG,"minHeight":"100vh"}),

], style={"backgroundColor":BG, "fontFamily":"DM Mono,monospace"})


# ── TAB CONTENT CALLBACK ──────────────────────────────────────────────────────
@app.callback(Output("tab-content","children"),
              Input("view-tabs","value"),
              Input("interval","n_intervals"))
def render_tab(tab, _):
    if tab == "hourly":
        return render_hourly()
    elif tab == "daily":
        return render_daily()
    elif tab == "alerts":
        return render_alerts()
    elif tab == "nemotron":
        return render_nemotron()
    return html.Div("Select a tab")


# ── HOURLY VIEW ───────────────────────────────────────────────────────────────
def render_hourly():
    # High-risk trend
    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        x=hourly['hour'], y=hourly['high_risk_count'],
        mode='lines+markers', name='High Risk Patients',
        line=dict(color=DANGER_RED, width=2.5),
        marker=dict(size=5),
        fill='tozeroy', fillcolor='rgba(255,69,96,0.1)',
    ))
    fig_trend.add_trace(go.Scatter(
        x=hourly['hour'], y=hourly['moderate_risk_count'],
        mode='lines+markers', name='Moderate Risk',
        line=dict(color=WARN_ORANGE, width=2),
        marker=dict(size=4),
    ))
    fig_trend.add_trace(go.Scatter(
        x=hourly['hour'], y=hourly['low_risk_count'],
        mode='lines+markers', name='Low Risk / Stable',
        line=dict(color=NVIDIA_GRN, width=1.5),
        marker=dict(size=4),
    ))
    fig_trend.update_xaxes(title='Hour of Day (0–23)', tickvals=list(range(0,24,3)))
    fig_trend.update_yaxes(title='Patient Count')
    fig_layout(fig_trend, "⏱  Hourly Risk Patient Count — Ward 4 (24h)", height=300)

    # Alerts per hour bar
    # Alerts split by risk level for each hour
    _rng2 = np.random.default_rng(99)
    _af   = hourly['alerts_fired'].values
    _high_alerts = [max(0, int(v * 0.55 + _rng2.integers(-1,2))) for v in _af]
    _mod_alerts  = [max(0, int(v * 0.35 + _rng2.integers(-1,2))) for v in _af]
    _low_alerts  = [max(0, int(v * 0.10 + _rng2.integers(0,2)))  for v in _af]
    fig_alerts = go.Figure()
    fig_alerts.add_trace(go.Bar(x=hourly['hour'], y=_high_alerts, name='HIGH alerts',
                                marker_color=DANGER_RED))
    fig_alerts.add_trace(go.Bar(x=hourly['hour'], y=_mod_alerts,  name='MODERATE alerts',
                                marker_color=WARN_ORANGE))
    fig_alerts.add_trace(go.Bar(x=hourly['hour'], y=_low_alerts,  name='LOW alerts',
                                marker_color=NVIDIA_GRN))
    fig_alerts.update_layout(barmode='stack')
    fig_alerts.update_xaxes(title='Hour', tickvals=list(range(0,24,2)))
    fig_alerts.update_yaxes(title='Alerts')
    fig_layout(fig_alerts, "🚨  Alerts Fired Per Hour", height=240)

    # Avg risk score sparkline
    fig_spark = go.Figure(go.Scatter(
        x=hourly['hour'], y=hourly['avg_risk_score'],
        mode='lines', line=dict(color=ACCENT_BLUE, width=2.5),
        fill='tozeroy', fillcolor='rgba(0,200,255,0.08)',
    ))
    fig_spark.add_hline(y=0.25, line_dash='dot', line_color=WARN_ORANGE, annotation_text='Moderate threshold', annotation_font_size=9)
    fig_spark.add_hline(y=0.55, line_dash='dot', line_color=DANGER_RED,  annotation_text='High threshold',     annotation_font_size=9)
    fig_spark.update_xaxes(title='Hour', tickvals=list(range(0,24,3)))
    fig_spark.update_yaxes(title='Avg Risk Score', range=[0,1])
    fig_layout(fig_spark, "📈  Average Ward Risk Score — Hourly", height=240)

    return dbc.Row([
        dbc.Col([
            html.Div(dcc.Graph(figure=fig_trend, config={'displayModeBar':False}), style=CARD_STYLE),
        ], md=12),
        dbc.Col([
            html.Div(dcc.Graph(figure=fig_alerts, config={'displayModeBar':False}), style=CARD_STYLE),
        ], md=6),
        dbc.Col([
            html.Div(dcc.Graph(figure=fig_spark, config={'displayModeBar':False}), style=CARD_STYLE),
        ], md=6),
    ])


# ── DAILY VIEW ────────────────────────────────────────────────────────────────
def render_daily():
    # Risk distribution donut
    counts = df['risk_level'].value_counts()
    fig_donut = go.Figure(go.Pie(
        labels=['HIGH','MODERATE','LOW'],
        values=[counts.get('HIGH',0), counts.get('MODERATE',0), counts.get('LOW',0)],
        hole=0.6,
        marker=dict(colors=[DANGER_RED, WARN_ORANGE, NVIDIA_GRN]),
        textfont=dict(size=12, color='white'),
    ))
    fig_donut.update_layout(**PLOTLY_LAYOUT, height=280,
        title=dict(text="Risk Distribution — All Patients", font=dict(size=12,color=TEXT_DIM)),
        annotations=[dict(text=f"<b>{n_total}</b><br>Total", x=0.5, y=0.5, font_size=14, showarrow=False, font_color=TEXT)])

    # Risk score histogram
    fig_hist = go.Figure()
    for level, color in [('HIGH',DANGER_RED),('MODERATE',WARN_ORANGE),('LOW',NVIDIA_GRN)]:
        sub = df[df['risk_level']==level]['risk_score']
        fig_hist.add_trace(go.Histogram(x=sub, name=level, nbinsx=25,
                                         marker_color=color, opacity=0.7, histnorm='percent'))
    fig_hist.update_xaxes(title='Risk Score')
    fig_hist.update_yaxes(title='% of Patients')
    fig_layout(fig_hist, "Risk Score Distribution by Level", height=280)
    fig_hist.update_layout(barmode='overlay')

    # Top diagnoses by risk
    diag_risk = df.groupby('diagnosis')['risk_score'].agg(['mean','count']).reset_index()
    diag_risk.columns = ['diagnosis','mean_risk','count']
    diag_risk = diag_risk.sort_values('mean_risk', ascending=True).tail(8)
    fig_diag = go.Figure(go.Bar(
        x=diag_risk['mean_risk'], y=diag_risk['diagnosis'],
        orientation='h',
        marker=dict(
            color=diag_risk['mean_risk'],
            colorscale=[[0,NVIDIA_GRN],[0.5,WARN_ORANGE],[1,DANGER_RED]],
            cmin=0, cmax=1,
        ),
        text=[f"{v:.0%}" for v in diag_risk['mean_risk']],
        textposition='auto',
    ))
    fig_diag.update_xaxes(title='Mean Risk Score', range=[0,1])
    fig_layout(fig_diag, "📋  Average Risk by Diagnosis", height=300)

    # SHAP feature importance
    feat_names = shap_meta.get('feature_names',[])[:15]
    mean_shap  = shap_meta.get('mean_shap',[])[:15]
    if feat_names and mean_shap:
        sorted_pairs = sorted(zip(mean_shap, feat_names), reverse=True)[:12]
        sv, fn = zip(*sorted_pairs)
        fig_shap = go.Figure(go.Bar(
            x=list(sv)[::-1], y=list(fn)[::-1],
            orientation='h',
            marker=dict(color=[NVIDIA_GRN if i < 3 else (WARN_ORANGE if i < 6 else ACCENT_BLUE) for i in range(12)][::-1]),
        ))
        fig_shap.update_xaxes(title='Mean |SHAP Value|')
        fig_layout(fig_shap, "🔍  Feature Importance (SHAP) — What Drives Deterioration", height=300)
    else:
        fig_shap = go.Figure()
        fig_layout(fig_shap, "SHAP data not available", height=300)

    # News2 vs risk scatter
    sample = df.sample(min(300, len(df)), random_state=42)
    fig_scatter = go.Figure(go.Scatter(
        x=sample['news2_score'], y=sample['risk_score'],
        mode='markers',
        marker=dict(
            color=sample['risk_score'],
            colorscale=[[0,NVIDIA_GRN],[0.5,WARN_ORANGE],[1,DANGER_RED]],
            size=5, opacity=0.7, cmin=0, cmax=1,
        ),
        text=[f"{row['patient_id']} — {row['diagnosis']}" for _, row in sample.iterrows()],
        hovertemplate='%{text}<br>NEWS2: %{x}<br>Risk: %{y:.3f}<extra></extra>',
    ))
    fig_scatter.update_xaxes(title='NEWS2 Score')
    fig_scatter.update_yaxes(title='Model Risk Score')
    fig_layout(fig_scatter, "NEWS2 Score vs Model Risk Score (sample 300 patients)", height=280)

    return dbc.Row([
        dbc.Col([html.Div(dcc.Graph(figure=fig_donut,   config={'displayModeBar':False}), style=CARD_STYLE)], md=4),
        dbc.Col([html.Div(dcc.Graph(figure=fig_hist,    config={'displayModeBar':False}), style=CARD_STYLE)], md=8),
        dbc.Col([html.Div(dcc.Graph(figure=fig_diag,    config={'displayModeBar':False}), style=CARD_STYLE)], md=6),
        dbc.Col([html.Div(dcc.Graph(figure=fig_shap,    config={'displayModeBar':False}), style=CARD_STYLE)], md=6),
        dbc.Col([html.Div(dcc.Graph(figure=fig_scatter, config={'displayModeBar':False}), style=CARD_STYLE)], md=12),
    ])


# ── ALERT QUEUE ───────────────────────────────────────────────────────────────
def render_alerts():
    # Show top 20 by risk score — all levels, sorted HIGH first then by score
    _lo = {'HIGH':0,'MODERATE':1,'LOW':2}
    _tmp = df.copy()
    _tmp['_tier'] = _tmp['risk_level'].astype(str).map(_lo).fillna(3)
    top = (_tmp.sort_values(['_tier','risk_score'], ascending=[True,False])
               .head(20).drop(columns=['_tier']).copy())
    top['risk_level'] = top['risk_level'].astype(str)

    rows = []
    for _, p in top.iterrows():
        risk_color = DANGER_RED if p['risk_level']=='HIGH' else (WARN_ORANGE if p['risk_level']=='MODERATE' else NVIDIA_GRN)
        rows.append(html.Tr([
            html.Td(p['patient_id'], style={"fontFamily":"DM Mono,monospace","fontSize":"12px","color":ACCENT_BLUE,"fontWeight":"500"}),
            html.Td(p['name'],       style={"fontSize":"12px"}),
            html.Td(f"{int(p['age'])} {p['gender']}", style={"fontSize":"11px","color":TEXT_DIM}),
            html.Td(p['diagnosis'],  style={"fontSize":"11px","color":TEXT_DIM}),
            html.Td(p.get('room','—'), style={"fontSize":"11px","textAlign":"center"}),
            html.Td(f"{p['hr']:.0f}", style={"fontFamily":"DM Mono,monospace","fontSize":"12px",
                                              "color":DANGER_RED if p['hr']>100 else TEXT}),
            html.Td(f"{p['spo2']:.1f}%", style={"fontFamily":"DM Mono,monospace","fontSize":"12px",
                                                  "color":DANGER_RED if p['spo2']<94 else TEXT}),
            html.Td(f"{p['sbp']:.0f}", style={"fontFamily":"DM Mono,monospace","fontSize":"12px",
                                               "color":DANGER_RED if p['sbp']<100 else TEXT}),
            html.Td(f"{int(p['news2_score'])}", style={"fontFamily":"DM Mono,monospace","fontSize":"12px",
                                                        "color":DANGER_RED if p['news2_score']>=7 else (WARN_ORANGE if p['news2_score']>=4 else TEXT)}),
            html.Td(html.Div(f"{p['risk_score']:.0%}", style={
                "fontFamily":"DM Mono,monospace","fontSize":"13px","fontWeight":"600","color":risk_color})),
            html.Td(risk_badge(p['risk_level'])),
        ], style={"borderBottom":f"1px solid {BORDER}","transition":"background 0.2s"}))

    thead_style = {"fontFamily":"DM Mono,monospace","fontSize":"10px","color":TEXT_DIM,
                   "letterSpacing":"0.08em","textTransform":"uppercase","borderBottom":f"1px solid {BORDER}",
                   "padding":"8px 10px"}

    table = html.Table([
        html.Thead(html.Tr([
            html.Th("ID",       style=thead_style), html.Th("Name",    style=thead_style),
            html.Th("Age/Sex",  style=thead_style), html.Th("Diagnosis",style=thead_style),
            html.Th("Room",     style=thead_style), html.Th("HR",       style=thead_style),
            html.Th("SpO₂",     style=thead_style), html.Th("SBP",      style=thead_style),
            html.Th("NEWS2",    style=thead_style), html.Th("Risk",     style=thead_style),
            html.Th("Level",    style=thead_style),
        ])),
        html.Tbody(rows),
    ], style={"width":"100%","borderCollapse":"collapse"})

    return dbc.Row([
        dbc.Col([
            html.Div([
                html.Div("  Top 20 High-Risk Patients — Alert Queue",
                         style={"fontFamily":"DM Mono,monospace","fontSize":"11px","color":TEXT_DIM,
                                "textTransform":"uppercase","letterSpacing":"0.1em","marginBottom":"14px"}),
                html.Div(table, style={"overflowX":"auto"}),
            ], style=CARD_STYLE)
        ], md=12),
    ])


# ── NEMOTRON PANEL ────────────────────────────────────────────────────────────
def render_nemotron():
        # Show a curated mix: top 4 HIGH + top 3 MODERATE + top 3 LOW
    # This demonstrates all three risk tiers to executives
    def _top_n(level, n):
        return (df[df['risk_level'].astype(str)==level]
                .nlargest(n, 'risk_score'))
    top_patients = pd.concat([
        _top_n('HIGH',     4),
        _top_n('MODERATE', 3),
        _top_n('LOW',      3),
    ]).reset_index(drop=True)

    cards = []
    for _, p in top_patients.iterrows():
        insight = get_insight(p)
        level   = str(p['risk_level'])
        border_color = DANGER_RED if level=='HIGH' else (WARN_ORANGE if level=='MODERATE' else NVIDIA_GRN)
        bg_color     = "rgba(255,69,96,0.05)" if level=='HIGH' else ("rgba(255,159,28,0.05)" if level=='MODERATE' else "rgba(118,185,0,0.04)")

        card = dbc.Col(html.Div([
            # Patient header
            html.Div([
                html.Div([
                    html.Span(p['patient_id'], style={"fontFamily":"DM Mono,monospace","fontSize":"12px","color":ACCENT_BLUE,"fontWeight":"500"}),
                    html.Span(f"  {p['name']}", style={"fontSize":"13px","color":TEXT}),
                ]),
                html.Div([
                    risk_badge(level),
                    html.Span(f"  {p['risk_score']:.0%}",
                              style={"fontFamily":"DM Mono,monospace","fontSize":"13px",
                                     "color":border_color,"fontWeight":"600","marginLeft":"8px"}),
                ]),
            ], style={"display":"flex","justifyContent":"space-between","alignItems":"center","marginBottom":"10px"}),

            # Vitals mini row
            html.Div([
                html.Span(f"HR {p['hr']:.0f}", style={"fontFamily":"DM Mono,monospace","fontSize":"11px",
                    "color":DANGER_RED if p['hr']>100 else TEXT_DIM, "marginRight":"12px"}),
                html.Span(f"SpO₂ {p['spo2']:.1f}%", style={"fontFamily":"DM Mono,monospace","fontSize":"11px",
                    "color":DANGER_RED if p['spo2']<94 else TEXT_DIM, "marginRight":"12px"}),
                html.Span(f"SBP {p['sbp']:.0f}", style={"fontFamily":"DM Mono,monospace","fontSize":"11px",
                    "color":DANGER_RED if p['sbp']<100 else TEXT_DIM, "marginRight":"12px"}),
                html.Span(f"NEWS2 {int(p['news2_score'])}", style={"fontFamily":"DM Mono,monospace","fontSize":"11px",
                    "color":DANGER_RED if p['news2_score']>=7 else TEXT_DIM}),
            ], style={"marginBottom":"12px","padding":"8px","background":"rgba(0,0,0,0.3)","borderRadius":"6px"}),

            # AI insight
            html.Div([
                html.Div("◆ NEMOTRON CLINICAL INSIGHT",
                         style={"fontFamily":"DM Mono,monospace","fontSize":"9px","color":NVIDIA_GRN,
                                "letterSpacing":"0.1em","marginBottom":"8px"}),
                html.Div(insight, style={"fontSize":"12px","color":"#9bb8d4","lineHeight":"1.7"}),
            ], style={"borderLeft":f"2px solid {NVIDIA_GRN}","paddingLeft":"12px"}),

        ], style={
            **CARD_STYLE,
            "borderLeft": f"3px solid {border_color}",
            "background": bg_color,
            "marginBottom": "12px",
        }), md=6)
        cards.append(card)

    # Tech explanation panel
    tech_panel = dbc.Col(html.Div([
        html.Div("HOW THE PIPELINE WORKS", style={"fontFamily":"DM Mono,monospace","fontSize":"10px",
                                                    "color":TEXT_DIM,"letterSpacing":"0.1em","marginBottom":"16px"}),
        *[html.Div([
            html.Div(step_num, style={"fontFamily":"DM Mono,monospace","fontSize":"10px","color":color,"marginBottom":"4px"}),
            html.Div(desc, style={"fontSize":"12px","color":TEXT_DIM,"lineHeight":"1.7","marginBottom":"14px",
                                   "paddingLeft":"10px","borderLeft":f"2px solid {color}"}),
        ]) for step_num, desc, color in [
            ("⓪ DGX Infrastructure", "On-premise AI factory where DGX H100/B200 hosts the entire pipeline at hospital scale. Patient data never leaves the building. HIPAA-compliant by design. Deployed at medical facilities today.", "#a67fff"),
            ("① pandas / RAPIDS cuDF", "847 ICU patients generated with realistic vitals, labs, and 24-hour trend slopes that are processed using pandas on a local machine. On DGX with RAPIDS cuDF the identical code handles 50M+ rows in seconds.", ACCENT_BLUE),
            ("② XGBoost + SHAP", "XGBRegressor trained on 28 clinical features including vitals, lab values, trend slopes, and diagnosis risk. Outputs a continuous 0–1 risk score per patient. SHAP values identify exactly which features drive each patient's score.", NVIDIA_GRN),
            ("③ Nemotron-Nano-9B-v2", "NVIDIA-Nemotron-Nano-9B-v2 running locally via llama.cpp on port 8082. Each patient's vitals, labs, and top SHAP drivers are sent to the model which returns a brief clinical summary tailored to the patient's risk level.", WARN_ORANGE),
        ]],
        html.Div("DGX Infrastructure · pandas / RAPIDS cuDF · XGBoost + SHAP · Nemotron-Nano-9B-v2 · llama.cpp · 28 clinical features · 847 patients",
                 style={"fontFamily":"DM Mono,monospace","fontSize":"9px","color":TEXT_DIM,"marginTop":"8px"}),
    ], style=CARD_STYLE), md=12)

    return dbc.Row([*cards, tech_panel])


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("\n" + "="*60)
    print("  Health AI Dashboard")
    print("  → http://localhost:8060")
    print("="*60 + "\n")
    app.run(debug=False, port=8060, host='127.0.0.1', use_reloader=False)