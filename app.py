import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
from pathlib import Path

# ==========================================================
# BLOODDEMAND AI — ENHANCED HEALTHCARE ANALYTICS DASHBOARD
# Keeps the existing dataset/model file names:
#   blooddemand_cleaned.csv
#   blooddemand_model.joblib
#   model_evaluation.csv (optional)
# ==========================================================

st.set_page_config(
    page_title="BloodDemand AI | Demand Intelligence",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE = Path(__file__).resolve().parent

# ------------------------------ DESIGN ------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

:root {
  --navy:#132238; --navy2:#203650; --teal:#0d9488; --teal-dark:#087f75;
  --mint:#dff5ef; --ink:#172b42; --muted:#64748b; --canvas:#f3f7fb;
  --line:#e2eaf1; --white:#ffffff; --red:#e85d67; --blue:#5b8def;
}
html, body, [class*="css"] {font-family:'DM Sans',sans-serif;}
.stApp {background:var(--canvas); color:var(--ink);}
header[data-testid="stHeader"] {background:rgba(243,247,251,.96);}
.block-container {padding-top:1.6rem;padding-bottom:3rem;max-width:1500px;}
h1,h2,h3,h4 {font-family:'Manrope',sans-serif!important;color:var(--ink)!important;}
p, label, li {color:#334155;}
[data-testid="stCaptionContainer"] {color:var(--muted)!important;}
section[data-testid="stSidebar"] {background:var(--navy);border-right:1px solid #263c54;}
section[data-testid="stSidebar"] * {color:#e9f0f7;}
section[data-testid="stSidebar"] div[role="radiogroup"] label {
  padding:.58rem .72rem;border-radius:10px;margin:.14rem 0;border:1px solid transparent;
}
section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {background:var(--navy2);}
section[data-testid="stSidebar"] div[role="radiogroup"] label p {color:#e9f0f7!important;}
.brand {padding:.8rem 0 1.35rem;border-bottom:1px solid #34465a;margin-bottom:1.2rem;}
.brand-mark {display:inline-flex;align-items:center;justify-content:center;width:43px;height:43px;
  background:var(--teal);border-radius:13px;font-size:22px;margin-bottom:12px;}
.brand-title {font-family:'Manrope',sans-serif;color:white;font-size:1.25rem;font-weight:800;letter-spacing:-.4px;}
.brand-sub {color:#b6c6d8;font-size:.78rem;margin-top:5px;}
.side-label {font-size:.66rem;color:#9bb0c6;font-weight:800;letter-spacing:1.6px;text-transform:uppercase;margin:1.1rem 0 .5rem;}
.side-status {font-size:.79rem;color:#d3dfeb;line-height:2.05;}
.topline {display:flex;align-items:center;justify-content:space-between;gap:14px;margin-bottom:1.25rem;}
.eyebrow {color:var(--teal);font-size:.7rem;font-weight:800;letter-spacing:1.8px;text-transform:uppercase;margin-bottom:7px;}
.page-title {font-family:'Manrope',sans-serif;font-size:clamp(1.7rem,3vw,2.25rem);font-weight:800;color:var(--ink);letter-spacing:-1px;line-height:1.2;}
.page-desc {color:var(--muted);font-size:.9rem;margin-top:8px;line-height:1.6;}
.status-pill {background:var(--mint);color:#087e73;border:1px solid #c4e8e1;border-radius:999px;padding:7px 12px;font-size:.69rem;font-weight:800;white-space:nowrap;}
.hero {background:linear-gradient(115deg,#132238 0%,#203b52 68%,#116f70 130%);border-radius:20px;padding:27px 29px;color:white;margin:0 0 20px;position:relative;overflow:hidden;}
.hero:after {content:"";position:absolute;width:220px;height:220px;border-radius:50%;right:-55px;top:-90px;border:1px solid rgba(255,255,255,.12);box-shadow:0 0 0 28px rgba(255,255,255,.025),0 0 0 58px rgba(255,255,255,.02);}
.hero-kicker {color:#8ce4d6;font-size:.68rem;font-weight:800;letter-spacing:1.8px;text-transform:uppercase;margin-bottom:9px;}
.hero-title {font-family:'Manrope',sans-serif;font-size:clamp(1.5rem,2.5vw,2.15rem);font-weight:800;line-height:1.22;letter-spacing:-.8px;color:white;}
.hero-title span {color:#72dfce;}
.hero-copy {color:#d2dfeb;font-size:.88rem;line-height:1.7;max-width:690px;margin-top:10px;}
.hero-foot {display:flex;flex-wrap:wrap;gap:8px;margin-top:17px;}
.hero-chip {border:1px solid rgba(255,255,255,.2);background:rgba(255,255,255,.08);border-radius:999px;padding:6px 10px;color:#e7f3f7;font-size:.7rem;}
.kpi {background:white;border:1px solid var(--line);border-radius:15px;padding:17px 18px;min-height:130px;box-shadow:0 5px 18px rgba(25,45,70,.035);}
.kpi-top {display:flex;justify-content:space-between;align-items:center;gap:8px;}
.kpi-label {color:#64748b;font-size:.78rem;font-weight:700;}
.kpi-icon {background:var(--mint);color:var(--teal);border-radius:10px;width:33px;height:33px;display:flex;align-items:center;justify-content:center;font-size:16px;}
.kpi-value {font-family:'Manrope',sans-serif;font-size:1.85rem;font-weight:800;color:var(--ink);margin-top:10px;letter-spacing:-.7px;}
.kpi-foot {color:#64748b;font-size:.7rem;margin-top:3px;}
.panel {background:white;border:1px solid var(--line);border-radius:15px;padding:17px 18px 13px;box-shadow:0 4px 14px rgba(25,45,70,.025);margin-bottom:15px;}
.panel-title {font-family:'Manrope',sans-serif;color:var(--ink);font-size:.98rem;font-weight:800;}
.panel-sub {color:var(--muted);font-size:.77rem;margin-top:5px;margin-bottom:10px;line-height:1.5;}
.section-heading {font-family:'Manrope',sans-serif;font-size:1.15rem;font-weight:800;color:var(--ink);margin:1.35rem 0 .25rem;}
.section-sub {color:var(--muted);font-size:.83rem;margin-bottom:1rem;line-height:1.5;}
.insight {background:#fff;border:1px solid var(--line);border-radius:13px;padding:15px 16px;height:100%;}
.insight-tag {font-size:.66rem;font-weight:800;letter-spacing:1.1px;text-transform:uppercase;color:var(--teal);margin-bottom:7px;}
.insight-title {font-size:.91rem;font-weight:800;color:var(--ink);margin-bottom:5px;}
.insight-copy {font-size:.79rem;color:#64748b;line-height:1.55;}
div[data-testid="stMetric"] {background:white;border:1px solid var(--line);border-radius:13px;padding:14px 16px;}
div[data-testid="stMetricLabel"] {color:#64748b!important;}
div[data-testid="stMetricValue"] {color:var(--ink)!important;}
.stButton>button,.stDownloadButton>button {background:var(--teal);color:white;border:1px solid var(--teal);border-radius:9px;font-weight:700;padding:.55rem .95rem;}
.stButton>button:hover,.stDownloadButton>button:hover {background:var(--teal-dark);color:white;border-color:var(--teal-dark);}
div[data-testid="stForm"] {background:white;border:1px solid var(--line);border-radius:14px;padding:19px;}
div[data-testid="stDataFrame"],[data-testid="stTable"] {background:white;border:1px solid var(--line);border-radius:10px;overflow:hidden;}
[data-testid="stTable"] th {background:#edf7f5!important;color:#244357!important;}
[data-testid="stTable"] td {color:#334155!important;}
div[data-testid="stSlider"] label,div[data-testid="stSelectbox"] label,div[data-testid="stNumberInput"] label {color:#334155!important;}
hr {border-color:var(--line);}
.footer {border-top:1px solid var(--line);padding-top:17px;margin-top:32px;color:#64748b;font-size:.75rem;line-height:1.8;}
@media(max-width:750px) {.block-container{padding:1rem}.hero{padding:21px}.status-pill{display:none}.kpi{padding:13px}}

/* FINAL POLISH */
div[data-testid="stPlotlyChart"] {border-radius:12px; overflow:hidden;}
div[data-testid="stPlotlyChart"] > div {border-radius:12px;}
.stSelectbox div[data-baseweb="select"] > div,
.stNumberInput div[data-baseweb="input"] {border-radius:10px;border-color:#d9e4ed;}
[data-testid="stAlert"] {border-radius:12px;}
button[kind="primaryFormSubmit"] {min-height:2.8rem;}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {line-height:1.65;}
@media (min-width: 1200px) {
  .block-container {padding-left:2.1rem;padding-right:2.1rem;}
}

</style>
""", unsafe_allow_html=True)

# ------------------------------ DATA ------------------------------

@st.cache_data
def load_data():
    path = BASE / "blooddemand_cleaned.csv"
    if not path.exists():
        raise FileNotFoundError("blooddemand_cleaned.csv is missing. Put it in the same folder as app.py.")
    data = pd.read_csv(path)
    required = {"X", "Y", "Demand"}
    if not required.issubset(data.columns):
        raise ValueError("Dataset must contain X, Y and Demand columns.")
    for col in ["X", "Y", "Demand"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=["X", "Y", "Demand"])
    data = data[data["Demand"] >= 0].drop_duplicates().reset_index(drop=True)
    if data.empty:
        raise ValueError("No valid records were found in the dataset.")
    return data

@st.cache_resource
def load_model():
    path = BASE / "blooddemand_model.joblib"
    if not path.exists():
        raise FileNotFoundError("blooddemand_model.joblib is missing. Put it in the same folder as app.py.")
    return joblib.load(path)

@st.cache_data
def load_evaluation():
    path = BASE / "model_evaluation.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()

try:
    df = load_data()
    model = load_model()
    evaluation = load_evaluation()
except Exception as error:
    st.error(f"Could not load the project files: {error}")
    st.stop()

# ------------------------------ HELPERS ------------------------------

def heading(section, title, description):
    st.markdown(f"""
    <div class="topline"><div>
      <div class="eyebrow">{section}</div>
      <div class="page-title">{title}</div>
      <div class="page-desc">{description}</div>
    </div><div class="status-pill">● ANALYTICS WORKSPACE</div></div>
    """, unsafe_allow_html=True)

def kpi(label, value, icon, foot):
    st.markdown(f"""
    <div class="kpi"><div class="kpi-top"><div class="kpi-label">{label}</div>
    <div class="kpi-icon">{icon}</div></div><div class="kpi-value">{value}</div>
    <div class="kpi-foot">{foot}</div></div>
    """, unsafe_allow_html=True)

def panel_start(title, subtitle=""):
    st.markdown(f'<div class="panel"><div class="panel-title">{title}</div><div class="panel-sub">{subtitle}</div>', unsafe_allow_html=True)

def panel_end():
    st.markdown("</div>", unsafe_allow_html=True)

def chart_layout(fig, height=330):
    fig.update_layout(
        height=height, margin=dict(l=12,r=12,t=25,b=12),
        paper_bgcolor="white", plot_bgcolor="white",
        font=dict(family="DM Sans",color="#475569",size=12),
        colorway=["#0d9488","#5b8def","#f59e0b","#8b7ce8"],
        legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1)
    )
    fig.update_xaxes(showgrid=False,linecolor="#e2e8f0",zeroline=False,tickfont=dict(color="#475569"))
    fig.update_yaxes(gridcolor="#edf1f5",zeroline=False,tickfont=dict(color="#475569"))
    return fig

def insight_card(tag, title, copy):
    st.markdown(f"""
    <div class="insight"><div class="insight-tag">{tag}</div>
    <div class="insight-title">{title}</div><div class="insight-copy">{copy}</div></div>
    """, unsafe_allow_html=True)

# ------------------------------ SIDEBAR ------------------------------

with st.sidebar:
    st.markdown("""
    <div class="brand"><div class="brand-mark">🩸</div>
    <div class="brand-title">BloodDemand AI</div>
    <div class="brand-sub">Geospatial demand intelligence</div></div>
    <div class="side-label">WORKSPACE</div>
    """, unsafe_allow_html=True)
    page = st.radio("Navigation",[
        "Overview","Demand Analytics","Prediction Studio","Model Performance","Dataset Explorer"
    ],label_visibility="collapsed")
    st.markdown("<br><div class='side-label'>PROJECT STATUS</div>",unsafe_allow_html=True)
    st.markdown(
        f"<div class='side-status'>● &nbsp; Dataset loaded<br>● &nbsp; {len(df):,} valid records<br>● &nbsp; Model loaded</div>",
        unsafe_allow_html=True
    )
    st.markdown("<br><br><div style='border-top:1px solid #34465a;padding-top:14px;color:#9bb0c6;font-size:.73rem;line-height:1.8'>BLOODDEMAND AI<br>Academic Data Science Prototype<br>Version 1.1</div>",unsafe_allow_html=True)

# ============================== OVERVIEW ==============================

if page == "Overview":
    heading("OVERVIEW / 01","Blood demand intelligence","Explore recorded demand, uncover geographic patterns, and evaluate machine-learning estimates.")

    st.markdown(f"""
    <div class="hero">
      <div class="hero-kicker">Healthcare analytics · Project overview</div>
      <div class="hero-title">Understand demand.<br><span>Make informed decisions.</span></div>
      <div class="hero-copy">Explore the geographic distribution of recorded blood demand, discover patterns in the data, and inspect experimental machine-learning estimates through one interactive workspace.</div>
      <div class="hero-foot">
        <div class="hero-chip">✦ Exploratory data analysis</div>
        <div class="hero-chip">⌖ Geographic patterns</div>
        <div class="hero-chip">⌁ Machine-learning prototype</div>
      </div>
    </div>
    """,unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    with c1: kpi("Valid records",f"{len(df):,}","▦","Records available for analysis")
    with c2: kpi("Average demand",f"{df['Demand'].mean():.2f}","↗","Mean observed demand value")
    with c3: kpi("Highest demand",f"{df['Demand'].max():g}","↑","Maximum observed value")
    with c4: kpi("Zero-demand records",f"{int((df['Demand']==0).sum()):,}","○","Records with demand equal to zero")

    st.markdown("<div class='section-heading'>At a glance</div><div class='section-sub'>A quick overview of demand distribution and the locations represented in the dataset.</div>",unsafe_allow_html=True)
    left,right = st.columns([1.1,1],gap="medium")
    with left:
        panel_start("Geographic demand pattern","Each point represents one record. Colour and marker size reflect recorded demand.")
        fig = px.scatter(df,x="Y",y="X",color="Demand",size="Demand",hover_data=["X","Y","Demand"],
                         color_continuous_scale=["#c9eee8","#0d9488","#145b62"],
                         labels={"X":"Latitude (X)","Y":"Longitude (Y)","Demand":"Observed demand"})
        fig.update_traces(marker=dict(opacity=.86,line=dict(width=0)))
        fig.update_layout(height=350,margin=dict(l=10,r=10,t=10,b=10),paper_bgcolor="white",plot_bgcolor="white",
                          font=dict(color="#475569"),coloraxis_colorbar=dict(title="Demand"))
        fig.update_xaxes(showgrid=True,gridcolor="#edf1f5")
        fig.update_yaxes(showgrid=True,gridcolor="#edf1f5")
        st.plotly_chart(fig,use_container_width=True)
        panel_end()
    with right:
        panel_start("Demand distribution","How often each observed demand value appears.")
        counts = df["Demand"].value_counts().sort_index().reset_index()
        counts.columns = ["Demand","Records"]
        fig = px.bar(counts,x="Demand",y="Records",text="Records",color_discrete_sequence=["#0d9488"])
        fig.update_traces(textposition="outside",cliponaxis=False,marker_line_color="white",marker_line_width=1)
        st.plotly_chart(chart_layout(fig,350),use_container_width=True)
        panel_end()

    st.markdown("<div class='section-heading'>Insights from your dataset</div><div class='section-sub'>These observations are calculated from the loaded records—not manually entered sample claims.</div>",unsafe_allow_html=True)
    mean_d = float(df["Demand"].mean())
    median_d = float(df["Demand"].median())
    std_d = float(df["Demand"].std()) if len(df)>1 else 0.0
    zero_pct = float((df["Demand"]==0).mean()*100)
    common = df["Demand"].mode()
    common_text = f"{common.iloc[0]:g}" if not common.empty else "Not available"
    i1,i2,i3 = st.columns(3,gap="medium")
    with i1: insight_card("CENTRAL TENDENCY",f"Typical demand: {median_d:g}","The median is the middle recorded demand value and is less sensitive to extreme observations than the mean.")
    with i2: insight_card("MOST COMMON VALUE",f"Mode: {common_text}","This is the demand value that appears most frequently in the current dataset.")
    with i3: insight_card("ZERO-DEMAND CHECK",f"{zero_pct:.1f}% of records","This percentage describes zero-demand observations in this dataset; it does not by itself indicate a stock shortage.")
    left,right = st.columns(2,gap="medium")
    with left:
        panel_start("Demand summary","Descriptive statistics calculated from the data.")
        summary = df["Demand"].describe().rename({"count":"Records","mean":"Mean","std":"Standard deviation","min":"Minimum","25%":"25th percentile","50%":"Median","75%":"75th percentile","max":"Maximum"})
        st.table(summary.round(2).to_frame("Value"))
        panel_end()
    with right:
        panel_start("How the prototype works","The workflow behind the analysis.")
        st.markdown("""
**01 · Prepare**  
Clean numeric fields and remove invalid or duplicate records.

**02 · Explore**  
Inspect distributions, summary statistics and coordinate patterns.

**03 · Estimate**  
Use geographic coordinates as model inputs to estimate recorded demand.

**04 · Evaluate**  
Review held-out error metrics and compare candidate models.
        """)
        panel_end()
    st.markdown("<div class='section-heading'>What this prototype can—and cannot—tell you</div><div class='section-sub'>A transparent summary to support responsible interpretation of the results.</div>",unsafe_allow_html=True)
    scope_a, scope_b = st.columns(2, gap="medium")
    with scope_a:
        insight_card("SUPPORTED BY THE DATA", "Observed demand + coordinates", "The dashboard summarizes the Demand field, explores X/Y coordinate patterns, and displays estimates from the trained model.")
    with scope_b:
        insight_card("NOT INCLUDED IN THIS DATASET", "No time, blood group or stock fields", "The current data cannot validate future time-based forecasts, blood-group-specific requirements, inventory levels, or real shortage alerts.")
    st.caption("Academic prototype · Results depend on the dataset and the saved model. They should not be used for clinical or blood-bank operational decisions.")

# ============================== ANALYTICS ==============================

elif page == "Demand Analytics":
    heading("ANALYTICS / 02","Explore demand patterns","Filter records and investigate how observed demand varies across the dataset.")
    lo,hi = float(df["Demand"].min()),float(df["Demand"].max())
    if lo < hi:
        selected = st.slider("Filter by demand range",lo,hi,(lo,hi))
        filtered = df[df["Demand"].between(*selected)].copy()
    else:
        filtered = df.copy()
    a,b,c = st.columns(3)
    a.metric("Filtered records",f"{len(filtered):,}")
    b.metric("Average filtered demand",f"{filtered['Demand'].mean():.2f}" if len(filtered) else "—")
    c.metric("Demand range",f"{filtered['Demand'].min():g} – {filtered['Demand'].max():g}" if len(filtered) else "—")
    left,right = st.columns(2,gap="medium")
    with left:
        panel_start("Demand frequency","Count of records at each demand value.")
        counts = filtered["Demand"].value_counts().sort_index().reset_index()
        counts.columns=["Demand","Records"]
        fig=px.bar(counts,x="Demand",y="Records",color_discrete_sequence=["#0d9488"])
        st.plotly_chart(chart_layout(fig),use_container_width=True)
        panel_end()
    with right:
        panel_start("Demand spread","Box plot showing median, quartiles and individual records.")
        fig=px.box(filtered,y="Demand",color_discrete_sequence=["#5b8def"],points="all")
        st.plotly_chart(chart_layout(fig),use_container_width=True)
        panel_end()
    panel_start("Geographic coordinate analysis","Explore the coordinates represented by the filtered records.")
    fig=px.scatter(filtered,x="Y",y="X",color="Demand",size="Demand",hover_data=["X","Y","Demand"],
                   color_continuous_scale=["#c9eee8","#0d9488","#145b62"],
                   labels={"X":"Latitude (X)","Y":"Longitude (Y)","Demand":"Observed demand"})
    st.plotly_chart(chart_layout(fig,420),use_container_width=True)
    panel_end()
    st.download_button("Download filtered dataset",filtered.to_csv(index=False).encode("utf-8"),"blooddemand_filtered.csv","text/csv")

# ============================== PREDICTION ==============================

elif page == "Prediction Studio":
    heading("MACHINE LEARNING / 03","Prediction studio","Generate an experimental demand estimate from geographic coordinates.")
    st.info("This model estimates demand from coordinates only. Its output is experimental and is not a validated medical-supply recommendation.")
    left,right=st.columns([1,1.15],gap="large")
    with left:
        panel_start("Input coordinates","Choose coordinates within the observed dataset range.")
        x_min,x_max=float(df["X"].min()),float(df["X"].max())
        y_min,y_max=float(df["Y"].min()),float(df["Y"].max())
        with st.form("prediction_form"):
            x_value=st.number_input("X coordinate",min_value=x_min,max_value=x_max,value=float(df["X"].median()),format="%.6f")
            y_value=st.number_input("Y coordinate",min_value=y_min,max_value=y_max,value=float(df["Y"].median()),format="%.6f")
            submitted=st.form_submit_button("Estimate demand →",use_container_width=True)
        panel_end()
    with right:
        panel_start("Prediction result","Your model estimate will appear here.")
        if submitted:
            try:
                estimate=float(model.predict(pd.DataFrame({"X":[x_value],"Y":[y_value]}))[0])
                estimate=max(0.0,estimate)
                st.markdown("<div style='font-size:.72rem;font-weight:800;letter-spacing:1.4px;color:#64748b'>ESTIMATED DEMAND</div>",unsafe_allow_html=True)
                st.markdown(f"<div style='font-family:Manrope,sans-serif;font-size:3.2rem;font-weight:800;letter-spacing:-1px;color:#0d9488'>{estimate:.2f}</div>",unsafe_allow_html=True)
                st.caption("Experimental estimate—not a measured value.")
                a,b=st.columns(2)
                a.metric("X coordinate",f"{x_value:.5f}")
                b.metric("Y coordinate",f"{y_value:.5f}")
                avg=float(df["Demand"].mean())
                delta=estimate-avg
                st.markdown("**Compared with the dataset average**")
                st.progress(min(1.0,max(0.0,estimate/max(float(df["Demand"].max()),1))))
                st.caption(f"Estimate is {abs(delta):.2f} {'above' if delta>=0 else 'below'} the dataset mean ({avg:.2f}). This is a descriptive comparison, not a risk classification.")
                result=pd.DataFrame({"X":[x_value],"Y":[y_value],"Estimated Demand":[estimate]})
                st.download_button("Download prediction",result.to_csv(index=False).encode("utf-8"),"blooddemand_prediction.csv","text/csv")
            except Exception as error:
                st.error(f"Prediction failed: {error}")
        else:
            st.markdown("""
            <div style="background:#f4f8fb;border:1px dashed #cbd5e1;border-radius:13px;padding:35px 18px;text-align:center">
              <div style="font-size:2rem;color:#0d9488">⌖</div>
              <div style="font-size:1rem;font-weight:800;color:#172b42;margin-top:8px">Ready to estimate</div>
              <div style="font-size:.83rem;color:#64748b;margin-top:6px">Enter coordinates and submit the form to inspect a model estimate.</div>
            </div>
            """,unsafe_allow_html=True)
        panel_end()
    panel_start("Observed data reference","Summary of the model's input features and target variable.")
    st.dataframe(df[["X","Y","Demand"]].describe().round(3),use_container_width=True)
    panel_end()

# ============================== MODEL PERFORMANCE ==============================

elif page == "Model Performance":
    heading("EVALUATION / 04","Model performance","Review saved metrics and compare the models evaluated during training.")
    if evaluation.empty:
        st.warning("model_evaluation.csv was not found or contains no results. Add the evaluation file to view saved metrics.")
    else:
        required={"Model","MAE","RMSE","R2"}
        if not required.issubset(evaluation.columns):
            st.error("Expected columns in model_evaluation.csv: Model, MAE, RMSE and R2. Check the file headers.")
        else:
            candidates=evaluation[~evaluation["Model"].astype(str).str.contains("baseline",case=False,na=False)]
            if not candidates.empty:
                best=candidates.sort_values("MAE").iloc[0]
                a,b,c=st.columns(3)
                a.metric("Lowest candidate MAE",f"{best['MAE']:.3f}")
                b.metric("Candidate RMSE",f"{best['RMSE']:.3f}")
                c.metric("Candidate R²",f"{best['R2']:.3f}")
                st.caption(f"Lowest-MAE candidate in the saved results: {best['Model']}")
            panel_start("Saved evaluation results","Metrics shown from your model_evaluation.csv file.")
            st.dataframe(evaluation,use_container_width=True)
            panel_end()
            metric=st.selectbox("Metric to compare",["MAE","RMSE","R2"])
            fig=px.bar(evaluation,x="Model",y=metric,color="Model",text=metric,title=f"Model comparison · {metric}",
                       color_discrete_sequence=["#0d9488","#5b8def","#f59e0b","#8b7ce8"])
            fig.update_traces(texttemplate="%{text:.3f}",textposition="outside")
            st.plotly_chart(chart_layout(fig,420),use_container_width=True)
            a,b,c=st.columns(3,gap="medium")
            with a:
                panel_start("MAE","Mean Absolute Error")
                st.write("Average absolute prediction error. Lower values are better.")
                panel_end()
            with b:
                panel_start("RMSE","Root Mean Squared Error")
                st.write("Penalizes larger errors more strongly. Lower values are better.")
                panel_end()
            with c:
                panel_start("R²","Coefficient of determination")
                st.write("Compares the model with a mean-based baseline. R² can be negative on unseen data.")
                panel_end()
    st.warning("With a small coordinate-only dataset, evaluation results are preliminary and do not prove reliable real-world forecasting.")
    st.download_button("Download evaluation results",evaluation.to_csv(index=False).encode("utf-8"),"model_evaluation.csv","text/csv",disabled=evaluation.empty)

# ============================== DATASET EXPLORER ==============================

elif page == "Dataset Explorer":
    heading("DATA / 05","Dataset explorer","Inspect the cleaned records, review field types and export data.")
    a,b,c=st.columns(3)
    a.metric("Valid records",f"{len(df):,}")
    b.metric("Features",str(len(df.columns)))
    c.metric("Duplicate rows remaining",str(int(df.duplicated().sum())))
    st.markdown("### Filter records")
    lo,hi=float(df["Demand"].min()),float(df["Demand"].max())
    if lo < hi:
        selected=st.slider("Demand values",lo,hi,(lo,hi),key="explorer_filter")
        view=df[df["Demand"].between(*selected)]
    else:
        view=df.copy()
    st.dataframe(view,use_container_width=True,height=380)
    left,right=st.columns(2,gap="medium")
    with left:
        panel_start("Column information","Data type and number of distinct values in each field.")
        info=pd.DataFrame({"Column":df.columns,"Data type":[str(df[c].dtype) for c in df.columns],"Unique values":[df[c].nunique() for c in df.columns]})
        st.dataframe(info,use_container_width=True,hide_index=True)
        panel_end()
    with right:
        panel_start("Summary statistics","Descriptive statistics for numeric fields.")
        st.dataframe(df.describe().T.round(3),use_container_width=True)
        panel_end()
    st.download_button("Download cleaned dataset",df.to_csv(index=False).encode("utf-8"),"blooddemand_cleaned.csv","text/csv")

# ------------------------------ FOOTER ------------------------------

st.markdown("""
<div class="footer"><b>BloodDemand AI</b> · Geospatial demand analytics<br>
Academic Data Science Prototype · Version 1.1 · Exploratory insights are based on the loaded dataset.</div>
""",unsafe_allow_html=True)
