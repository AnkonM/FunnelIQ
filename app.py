import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix,
    precision_score, recall_score, f1_score, roc_curve, auc
)
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import os, urllib.request

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FunnelIQ – E-Commerce Analytics",
    layout="wide",
    page_icon="🛒",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.main-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 60%, #0ea5e9 100%);
    padding: 1.8rem 2rem; border-radius: 16px; margin-bottom: 1.5rem;
}
.main-header h1 { color:#fff; font-size:1.9rem; font-weight:700; margin:0; letter-spacing:-.5px; }
.main-header p  { color:#94a3b8; margin:4px 0 0 0; font-size:.92rem; }
.header-badge {
    background:rgba(14,165,233,.2); border:1px solid #0ea5e9; color:#38bdf8;
    border-radius:6px; padding:2px 10px; font-size:.72rem; font-weight:600;
    letter-spacing:1px; display:inline-block; margin-top:8px;
}
.kpi-card {
    background:linear-gradient(135deg,#1e293b,#0f172a); border:1px solid #334155;
    border-radius:14px; padding:1.2rem 1.4rem; text-align:center;
    transition:transform .2s,box-shadow .2s;
}
.kpi-card:hover { transform:translateY(-3px); box-shadow:0 8px 32px rgba(14,165,233,.15); }
.kpi-value { font-size:1.8rem; font-weight:700; color:#0ea5e9; line-height:1; }
.kpi-label { color:#94a3b8; font-size:.78rem; font-weight:500; margin-top:5px;
             text-transform:uppercase; letter-spacing:.5px; }
.kpi-delta-pos { color:#22c55e; font-size:.78rem; }
.kpi-delta-neg { color:#ef4444; font-size:.78rem; }
.section-header {
    color:#e2e8f0; font-weight:700; font-size:1.15rem;
    border-left:4px solid #0ea5e9; padding-left:10px; margin:1.4rem 0 .8rem 0;
}
.insight-box {
    background:linear-gradient(135deg,rgba(14,165,233,.07),rgba(99,102,241,.05));
    border:1px solid rgba(14,165,233,.22); border-radius:12px;
    padding:.9rem 1.1rem; color:#cbd5e1; font-size:.86rem; line-height:1.6;
}
.about-feature-card {
    background:#1e293b; border:1px solid #334155; border-radius:12px;
    padding:1.1rem; height:100%;
}
.about-feature-card h4 { color:#38bdf8; margin:0 0 5px 0; font-size:.95rem; }
.about-feature-card p  { color:#94a3b8; font-size:.82rem; margin:0; line-height:1.55; }
section[data-testid="stSidebar"] { background:#0f172a !important; border-right:1px solid #1e293b; }
section[data-testid="stSidebar"] * { color:#cbd5e1 !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────
PLOT_BG = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font_color="#cbd5e1",
)

# ─────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────
DATA_PATH = "online_shoppers_intention.csv"
DATA_URL  = "https://archive.ics.uci.edu/ml/machine-learning-databases/00468/online_shoppers_intention.csv"

@st.cache_data(show_spinner=False)
def load_data():
    if not os.path.exists(DATA_PATH):
        urllib.request.urlretrieve(DATA_URL, DATA_PATH)
    df = pd.read_csv(DATA_PATH)
    df["Revenue"] = df["Revenue"].astype(int)
    return df

with st.spinner("Loading dataset…"):
    df_raw = load_data()

# ─────────────────────────────────────────────────────────────
# ML PIPELINE
# ─────────────────────────────────────────────────────────────
NUM_COLS = [
    "Administrative", "Administrative_Duration",
    "Informational", "Informational_Duration",
    "ProductRelated", "ProductRelated_Duration",
    "BounceRates", "ExitRates", "PageValues", "SpecialDay",
]
CAT_COLS = ["Month", "VisitorType"]

@st.cache_data(show_spinner=False)
def build_and_train(_df):
    df = _df.copy()
    X = df[NUM_COLS + CAT_COLS]
    y = df["Revenue"]

    num_pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
    cat_pipe = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                         ("ohe", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"))])
    pre = ColumnTransformer([("num", num_pipe, NUM_COLS), ("cat", cat_pipe, CAT_COLS)])
    clf = LogisticRegression(random_state=42, max_iter=1000, class_weight="balanced")
    pipe = Pipeline([("pre", pre), ("clf", clf)])

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    pipe.fit(X_tr, y_tr)

    y_pred   = pipe.predict(X_te)
    y_prob   = pipe.predict_proba(X_te)[:, 1]
    y_te_arr = np.array(y_te)

    metrics = {
        "accuracy":  float(accuracy_score(y_te_arr, y_pred)),
        "precision": float(precision_score(y_te_arr, y_pred)),
        "recall":    float(recall_score(y_te_arr, y_pred)),
        "f1":        float(f1_score(y_te_arr, y_pred)),
    }
    cm_list = confusion_matrix(y_te_arr, y_pred).tolist()
    fpr_raw, tpr_raw, _ = roc_curve(y_te_arr, y_prob)
    roc_auc = float(auc(fpr_raw, tpr_raw))

    # Feature importance
    ohe_names = (pipe.named_steps["pre"]
                     .named_transformers_["cat"]
                     .named_steps["ohe"]
                     .get_feature_names_out(CAT_COLS))
    all_feats = NUM_COLS + list(ohe_names)
    coefs     = pipe.named_steps["clf"].coef_[0]
    feat_imp  = (pd.DataFrame({"Feature": all_feats, "Coefficient": coefs})
                 .sort_values("Coefficient", ascending=False)
                 .reset_index(drop=True))

    prob_df = pd.DataFrame({
        "Probability": y_prob,
        "Outcome": ["Converted" if v == 1 else "Not Converted" for v in y_te_arr],
    })

    return (pipe, metrics, cm_list,
            fpr_raw.tolist(), tpr_raw.tolist(), roc_auc,
            feat_imp, prob_df)

with st.spinner("Training model…"):
    (model, metrics, cm_list,
     fpr_list, tpr_list, roc_auc,
     feat_imp, prob_df) = build_and_train(df_raw)

conf_mat = np.array(cm_list)

# ─────────────────────────────────────────────────────────────
# FUNNEL DATA
# ─────────────────────────────────────────────────────────────
def compute_funnel(df):
    s1 = len(df)
    s2 = len(df[df["ProductRelated"] > 0])
    s3 = len(df[(df["ProductRelated_Duration"] > 120) & (df["BounceRates"] < 0.05)])
    s4 = len(df[df["PageValues"] > df["PageValues"].median()])
    s5 = int(df["Revenue"].sum())
    return {
        "stages": ["1 · Site Entry", "2 · Browsed Products",
                   "3 · High Engagement", "4 · High Page Value", "5 · Converted"],
        "counts": [s1, s2, s3, s4, s5],
    }

funnel = compute_funnel(df_raw)

# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
st.sidebar.markdown("## 🛒 FunnelIQ")
st.sidebar.markdown("*AI-Powered E-Commerce Analytics*")
st.sidebar.markdown("---")

NAV_LABELS = ["🏠 Overview", "📉 Funnel Analysis", "🤖 Model & Insights",
              "🔮 Prediction Tool", "📖 About"]
NAV_KEYS   = ["overview", "funnel", "model", "predict", "about"]

# Use index-based selectbox to avoid empty label issue on older Streamlit
nav_idx = st.sidebar.selectbox(
    "Page", range(len(NAV_LABELS)),
    format_func=lambda i: NAV_LABELS[i]
)
page = NAV_KEYS[nav_idx]

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Dataset:** UCI Online Shoppers Intention")
st.sidebar.markdown(f"**Records:** {len(df_raw):,} · **Features:** {df_raw.shape[1]}")
st.sidebar.markdown(f"**Model:** Logistic Regression")
st.sidebar.markdown(f"**AUC-ROC:** `{roc_auc:.3f}`")

# ─────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <h1>📊 FunnelIQ – Conversion Intelligence</h1>
  <p>Real-time funnel analytics &amp; ML-powered buy-intent scoring for E-Commerce</p>
  <span class="header-badge">UCI DATASET · LOGISTIC REGRESSION · STREAMLIT</span>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ═══════════════════════════════════════════════════════════════
if page == "overview":
    conv_rate  = float(df_raw["Revenue"].mean() * 100)
    avg_pages  = float(df_raw["ProductRelated"].median())
    avg_dur    = float(df_raw["ProductRelated_Duration"].median())
    avg_bounce = float(df_raw["BounceRates"].mean() * 100)
    avg_exit   = float(df_raw["ExitRates"].mean() * 100)
    avg_pv     = float(df_raw["PageValues"].mean())

    kpi_cols = st.columns(6)
    kpi_data = [
        (f"{conv_rate:.1f}%",  "Conversion Rate",        "↑ Target: 20%",      True),
        (f"{avg_pages:.0f}",   "Median Product Pages",   "",                   True),
        (f"{avg_dur:.0f}s",    "Median Browse Duration", "",                   True),
        (f"{avg_bounce:.1f}%", "Avg Bounce Rate",        "↓ Lower = better",   False),
        (f"{avg_exit:.1f}%",   "Avg Exit Rate",          "↓ Lower = better",   False),
        (f"{avg_pv:.1f}",      "Avg Page Value",         "↑ Higher = better",  True),
    ]
    for col, (val, label, delta, pos) in zip(kpi_cols, kpi_data):
        dcls = "kpi-delta-pos" if pos else "kpi-delta-neg"
        col.markdown(f"""<div class="kpi-card">
            <div class="kpi-value">{val}</div>
            <div class="kpi-label">{label}</div>
            <div class="{dcls}">{delta}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    r1, r2 = st.columns(2)

    # Chart 1: Conversion by Visitor Type
    with r1:
        st.markdown('<div class="section-header">Conversion Rate by Visitor Type</div>', unsafe_allow_html=True)
        vt = df_raw.groupby("VisitorType")["Revenue"].agg(["mean", "count"]).reset_index()
        vt.columns = ["Visitor Type", "Conv Rate", "Sessions"]
        vt["Conv Rate %"] = (vt["Conv Rate"] * 100).round(2)
        fig = px.bar(
            vt.sort_values("Conv Rate %", ascending=False),
            x="Visitor Type", y="Conv Rate %",
            color="Visitor Type", text="Conv Rate %",
            color_discrete_sequence=["#0ea5e9", "#6366f1", "#22c55e"],
            labels={"Conv Rate %": "Conversion Rate (%)"},
        )
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.update_layout(**PLOT_BG, showlegend=False, height=310,
                          margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig, use_container_width=True)

    # Chart 2: Monthly Conversion Trend
    with r2:
        st.markdown('<div class="section-header">Monthly Conversion Trend</div>', unsafe_allow_html=True)
        month_order = ["Jan","Feb","Mar","Apr","May","June","Jul","Aug","Sep","Oct","Nov","Dec"]
        mo = df_raw.groupby("Month")["Revenue"].mean().reset_index()
        mo["Month"] = pd.Categorical(mo["Month"], categories=month_order, ordered=True)
        mo = mo.sort_values("Month").dropna().reset_index(drop=True)
        mo["Conv %"] = (mo["Revenue"] * 100).round(2)
        fig2 = px.line(mo, x="Month", y="Conv %", markers=True,
                       color_discrete_sequence=["#0ea5e9"],
                       labels={"Conv %": "Conversion Rate (%)"})
        fig2.update_traces(line_width=2.5, marker_size=8)
        fig2.update_layout(**PLOT_BG, height=310, margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig2, use_container_width=True)

    r3, r4 = st.columns(2)

    # Chart 3: Sessions vs Conv Rate by Traffic Type (bubble chart)
    with r3:
        st.markdown('<div class="section-header">Traffic Type – Sessions vs Conversion Rate</div>',
                    unsafe_allow_html=True)
        tt = (df_raw.groupby("TrafficType")["Revenue"]
              .agg(count="count", conversions="sum").reset_index())
        tt["Conv Rate %"] = (tt["conversions"] / tt["count"] * 100).round(1)
        tt["Label"] = "Type " + tt["TrafficType"].astype(str)
        tt["bubble"] = tt["conversions"].clip(lower=1).astype(float)
        tt_top = tt.nlargest(10, "count").reset_index(drop=True)

        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=tt_top["count"].tolist(),
            y=tt_top["Conv Rate %"].tolist(),
            mode="markers+text",
            text=tt_top["Label"].tolist(),
            textposition="top center",
            textfont=dict(color="#94a3b8", size=9),
            marker=dict(
                size=(tt_top["bubble"] / tt_top["bubble"].max() * 50 + 12).tolist(),
                color=tt_top["Conv Rate %"].tolist(),
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="Conv%", tickfont=dict(color="#cbd5e1")),
            ),
            hovertemplate="<b>%{text}</b><br>Sessions: %{x}<br>Conv Rate: %{y:.1f}%<extra></extra>",
        ))
        fig3.update_layout(**PLOT_BG, height=310,
                           xaxis_title="Total Sessions",
                           yaxis_title="Conversion Rate (%)",
                           xaxis=dict(gridcolor="#1e293b"),
                           yaxis=dict(gridcolor="#1e293b"),
                           margin=dict(t=30, b=10, l=10, r=10))
        st.plotly_chart(fig3, use_container_width=True)

    # Chart 4: Page Value distribution – converted vs not
    with r4:
        st.markdown('<div class="section-header">Page Value Distribution by Outcome</div>',
                    unsafe_allow_html=True)
        pv_df = df_raw[["PageValues", "Revenue"]].copy()
        pv_df["Outcome"] = pv_df["Revenue"].map({1: "Converted", 0: "Not Converted"})
        # clip extreme outliers for readability
        pv_clip = pv_df[pv_df["PageValues"] < pv_df["PageValues"].quantile(0.98)].copy()
        fig4 = px.box(
            pv_clip, x="Outcome", y="PageValues", color="Outcome",
            color_discrete_map={"Converted": "#0ea5e9", "Not Converted": "#64748b"},
            category_orders={"Outcome": ["Not Converted", "Converted"]},
            labels={"PageValues": "Page Value", "Outcome": ""},
            points=False,
        )
        fig4.update_layout(**PLOT_BG, showlegend=False, height=310,
                           margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig4, use_container_width=True)

# ═══════════════════════════════════════════════════════════════
# PAGE: FUNNEL ANALYSIS
# ═══════════════════════════════════════════════════════════════
elif page == "funnel":
    st.markdown('<div class="section-header">Customer Journey Funnel</div>', unsafe_allow_html=True)

    cA, cB = st.columns([3, 2])
    with cA:
        fig = go.Figure(go.Funnel(
            y=funnel["stages"],
            x=funnel["counts"],
            textinfo="value+percent initial+percent previous",
            marker=dict(
                color=["#0ea5e9", "#38bdf8", "#818cf8", "#a78bfa", "#22c55e"],
                line=dict(width=1, color="#1e293b"),
            ),
            connector=dict(line=dict(color="#334155", width=1)),
        ))
        fig.update_layout(**PLOT_BG, height=420, margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig, use_container_width=True)

    with cB:
        st.markdown('<div class="section-header">Stage Breakdown</div>', unsafe_allow_html=True)
        stages = funnel["stages"]
        counts = funnel["counts"]
        for i in range(len(stages)):
            pct = counts[i] / counts[0] * 100
            drop_str = (f"↓ {(counts[i-1]-counts[i])/counts[i-1]*100:.1f}% drop from prev"
                        if i > 0 else "→ Starting traffic")
            st.markdown(f"""
            <div style="background:#1e293b;border:1px solid #334155;border-radius:10px;
                        padding:.75rem 1rem;margin-bottom:.55rem;">
              <div style="font-weight:600;color:#e2e8f0;font-size:.9rem;">{stages[i]}</div>
              <div style="color:#0ea5e9;font-size:1.25rem;font-weight:700;">{counts[i]:,}</div>
              <div style="display:flex;align-items:center;gap:.5rem;margin-top:4px;">
                <div style="flex:1;background:#334155;border-radius:4px;height:5px;">
                  <div style="width:{pct:.1f}%;background:#0ea5e9;height:5px;border-radius:4px;"></div>
                </div>
                <span style="color:#64748b;font-size:.76rem;">{pct:.1f}%</span>
              </div>
              <div style="color:#64748b;font-size:.76rem;margin-top:2px;">{drop_str}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-header">Segment Breakdowns</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)

    # Breakdown 1: Weekend vs Weekday pie
    with c1:
        wk = df_raw.groupby("Weekend")["Revenue"].mean().reset_index()
        wk["Label"] = wk["Weekend"].map({True: "Weekend", False: "Weekday"})
        wk["Rate %"] = (wk["Revenue"] * 100).round(1)
        fig_wk = px.pie(
            wk, values="Rate %", names="Label",
            color_discrete_sequence=["#0ea5e9", "#6366f1"],
            title="Weekend vs Weekday Conv. Rate",
            hole=0.45,
        )
        fig_wk.update_traces(textinfo="label+percent", textfont_size=11)
        fig_wk.update_layout(**PLOT_BG, height=280, margin=dict(t=40, b=10, l=10, r=10),
                             title_font_color="#e2e8f0")
        st.plotly_chart(fig_wk, use_container_width=True)

    # Breakdown 2: Bounce Rate brackets
    with c2:
        bounce_bins = pd.cut(
            df_raw["BounceRates"],
            bins=[0, .02, .05, .1, .2, 1.0],
            labels=["0–2%", "2–5%", "5–10%", "10–20%", ">20%"],
        )
        bb = df_raw.groupby(bounce_bins, observed=True)["Revenue"].mean().reset_index()
        bb.columns = ["Bounce Bracket", "Conv Rate"]
        bb["Conv %"] = (bb["Conv Rate"] * 100).round(1)
        bb["Bounce Bracket"] = bb["Bounce Bracket"].astype(str)
        fig_bb = px.bar(
            bb, x="Bounce Bracket", y="Conv %",
            text="Conv %",
            color="Conv %", color_continuous_scale="RdYlGn",
            title="Conv. Rate by Bounce Rate Bracket",
            labels={"Conv %": "Conv. Rate (%)"},
        )
        fig_bb.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig_bb.update_layout(**PLOT_BG, coloraxis_showscale=False, height=280,
                             margin=dict(t=40, b=10, l=10, r=10),
                             title_font_color="#e2e8f0")
        st.plotly_chart(fig_bb, use_container_width=True)

    # Breakdown 3: Conversion by OS — use go.Bar for full control (px.bar color='col' needs
    # exact palette length match which breaks on filtered subsets)
    with c3:
        os_raw = (df_raw.groupby("OperatingSystems")["Revenue"]
                  .agg(conv="sum", total="count").reset_index())
        os_raw = os_raw[os_raw["total"] > 50].copy()
        os_raw["pct"] = (os_raw["conv"] / os_raw["total"] * 100).round(1)
        os_raw["label"] = "OS " + os_raw["OperatingSystems"].astype(str)
        os_raw = os_raw.sort_values("pct", ascending=False).reset_index(drop=True)
        palette = ["#0ea5e9", "#22c55e", "#f59e0b", "#a78bfa", "#f43f5e", "#14b8a6"]
        bar_colors = [palette[i % len(palette)] for i in range(len(os_raw))]
        fig_os = go.Figure(go.Bar(
            x=os_raw["label"].tolist(),
            y=os_raw["pct"].tolist(),
            text=[f"{v:.1f}%" for v in os_raw["pct"].tolist()],
            textposition="outside",
            marker_color=bar_colors,
            hovertemplate="%{x}: %{y:.1f}%<extra></extra>",
        ))
        fig_os.update_layout(
            **PLOT_BG, title="Conv. Rate by Operating System",
            title_font_color="#e2e8f0", showlegend=False, height=280,
            xaxis=dict(title="Operating System"),
            yaxis=dict(title="Conv. Rate (%)"),
            margin=dict(t=50, b=10, l=10, r=10),
        )
        st.plotly_chart(fig_os, use_container_width=True)

# ═══════════════════════════════════════════════════════════════
# PAGE: MODEL & INSIGHTS
# ═══════════════════════════════════════════════════════════════
elif page == "model":
    st.markdown('<div class="section-header">Model Performance Summary</div>', unsafe_allow_html=True)

    mc = st.columns(4)
    for col, (key, label) in zip(mc, [("accuracy","Accuracy"),("precision","Precision"),
                                       ("recall","Recall"),("f1","F1 Score")]):
        col.markdown(f"""<div class="kpi-card">
            <div class="kpi-value">{metrics[key]*100:.1f}%</div>
            <div class="kpi-label">{label}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1.1, 1.1, 1])

    # Chart A: Confusion Matrix — use go.Heatmap for max reliability across Plotly versions
    with col1:
        st.markdown('<div class="section-header">Confusion Matrix</div>', unsafe_allow_html=True)
        cm_labels = ["Not Converted", "Converted"]
        # Convert to nested Python float lists — no numpy types
        cm_vals = [[float(conf_mat[r][c]) for c in range(2)] for r in range(2)]
        fig_cm = go.Figure(go.Heatmap(
            z=cm_vals,
            x=cm_labels,
            y=cm_labels,
            colorscale="Blues",
            showscale=False,
            text=[[str(int(conf_mat[r][c])) for c in range(2)] for r in range(2)],
            texttemplate="%{text}",
            textfont=dict(size=20),
            hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{text}<extra></extra>",
        ))
        fig_cm.update_layout(
            **PLOT_BG, height=320,
            xaxis=dict(title="Predicted", side="bottom"),
            yaxis=dict(title="Actual", autorange="reversed"),
            margin=dict(t=10, b=40, l=80, r=10),
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    # Chart B: ROC Curve
    with col2:
        st.markdown('<div class="section-header">ROC Curve</div>', unsafe_allow_html=True)
        fpr_py = [float(v) for v in fpr_list]
        tpr_py = [float(v) for v in tpr_list]
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(
            x=fpr_py, y=tpr_py, mode="lines",
            name=f"AUC = {roc_auc:.3f}",
            line=dict(color="#0ea5e9", width=2.5),
            fill="tozeroy", fillcolor="rgba(14,165,233,0.07)",
        ))
        fig_roc.add_trace(go.Scatter(
            x=[0.0, 1.0], y=[0.0, 1.0], mode="lines", name="Random",
            line=dict(color="#64748b", dash="dash", width=1.5),
        ))
        fig_roc.update_layout(
            **PLOT_BG, height=320,
            xaxis=dict(range=[0, 1], title="False Positive Rate", gridcolor="#1e293b"),
            yaxis=dict(range=[0, 1.02], title="True Positive Rate", gridcolor="#1e293b"),
            legend=dict(bgcolor="rgba(0,0,0,0)", x=0.38, y=0.08),
            margin=dict(t=10, b=10, l=10, r=10),
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    # Chart C: Predicted Probability Distribution
    with col3:
        st.markdown('<div class="section-header">Probability Distribution</div>', unsafe_allow_html=True)
        # Use go.Histogram traces directly — more reliable than px.histogram across versions
        conv_probs    = [float(p) for p, o in zip(prob_df["Probability"], prob_df["Outcome"]) if o == "Converted"]
        nonconv_probs = [float(p) for p, o in zip(prob_df["Probability"], prob_df["Outcome"]) if o == "Not Converted"]
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=nonconv_probs, name="Not Converted",
            marker_color="#64748b", opacity=0.7, nbinsx=35,
        ))
        fig_hist.add_trace(go.Histogram(
            x=conv_probs, name="Converted",
            marker_color="#0ea5e9", opacity=0.7, nbinsx=35,
        ))
        fig_hist.update_layout(
            barmode="overlay", **PLOT_BG, height=320,
            xaxis=dict(title="Predicted Probability", gridcolor="#1e293b"),
            yaxis=dict(title="Count", gridcolor="#1e293b"),
            legend=dict(bgcolor="rgba(0,0,0,0)"),
            margin=dict(t=10, b=10, l=10, r=10),
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    # Feature Importance
    st.markdown('<div class="section-header">Feature Coefficients (Model Interpretability)</div>',
                unsafe_allow_html=True)
    st.markdown("""<div class="insight-box">
    Logistic regression coefficients show each feature's log-odds impact on the conversion outcome.
    <b style="color:#22c55e;">Positive values</b> → increase purchase probability.
    <b style="color:#ef4444;">Negative values</b> → reduce it.
    <b>PageValues</b> dominates — users who visit high Google-Analytics-value pages convert at far higher rates.</div>""",
    unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    n_pos = int(min(8, (feat_imp["Coefficient"] > 0).sum()))
    n_neg = int(min(7, (feat_imp["Coefficient"] < 0).sum()))
    fi_show = pd.concat([
        feat_imp[feat_imp["Coefficient"] > 0].head(n_pos),
        feat_imp[feat_imp["Coefficient"] < 0].tail(n_neg),
    ]).reset_index(drop=True)
    fi_show["Color"] = fi_show["Coefficient"].apply(lambda x: "#22c55e" if x > 0 else "#ef4444")
    # Force all inputs to pure Python types
    y_feats  = fi_show["Feature"].tolist()
    x_coefs  = [float(v) for v in fi_show["Coefficient"].tolist()]
    bar_clrs = fi_show["Color"].tolist()
    bar_text = [f"{v:.3f}" for v in x_coefs]

    fig_fi = go.Figure(go.Bar(
        y=y_feats, x=x_coefs,
        orientation="h",
        marker_color=bar_clrs,
        text=bar_text,
        textposition="outside",
    ))
    fig_fi.update_layout(
        **PLOT_BG, height=440,
        xaxis=dict(title="Coefficient (Log-Odds)", zeroline=True,
                   zerolinecolor="#475569", zerolinewidth=1.5, gridcolor="#1e293b"),
        yaxis=dict(gridcolor="#1e293b", autorange="reversed"),
        margin=dict(t=10, b=10, l=10, r=100),
    )
    st.plotly_chart(fig_fi, use_container_width=True)

# ═══════════════════════════════════════════════════════════════
# PAGE: PREDICTION TOOL
# ═══════════════════════════════════════════════════════════════
elif page == "predict":
    st.markdown('<div class="section-header">🔮 Live Conversion Probability Estimator</div>',
                unsafe_allow_html=True)
    st.markdown("""<div class="insight-box">
    Enter session attributes and the Logistic Regression model will estimate the probability
    that this user will <b>complete a purchase</b>. Use the recommendation to decide whether to
    trigger a discount popup, nudge with social proof, or allow an uninterrupted checkout.</div>""",
    unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    inp_col, out_col = st.columns([1.1, 1])
    with inp_col:
        with st.form("predict_form"):
            st.markdown("**📄 Page Behaviour**")
            ra, rb, rc = st.columns(3)
            in_admin     = ra.number_input("Admin Pages",      0, 100,   2)
            in_admin_dur = rb.number_input("Admin Dur. (s)",   0, 10000, 50)
            in_info      = rc.number_input("Info Pages",       0, 100,   1)
            rd, re, rf   = st.columns(3)
            in_info_dur  = rd.number_input("Info Dur. (s)",    0, 10000, 20)
            in_prod      = re.number_input("Product Pages",    0, 500,   15)
            in_prod_dur  = rf.number_input("Product Dur. (s)", 0, 50000, 600)

            st.markdown("**📊 Quality Signals**")
            rg, rh = st.columns(2)
            in_bounce   = rg.slider("Bounce Rate",  0.0, 0.20, 0.02, step=0.005, format="%.3f")
            in_exit     = rh.slider("Exit Rate",    0.0, 0.20, 0.04, step=0.005, format="%.3f")
            ri, rj      = st.columns(2)
            in_pageval  = ri.number_input("Page Value (GA)", 0.0, 500.0, 25.0, step=1.0)
            in_specday  = rj.slider("Special Day Proximity", 0.0, 1.0, 0.0)

            st.markdown("**👤 Visitor Context**")
            rk, rl = st.columns(2)
            in_month    = rk.selectbox("Month", ["Jan","Feb","Mar","Apr","May","June",
                                                  "Jul","Aug","Sep","Oct","Nov","Dec"])
            in_visitor  = rl.selectbox("Visitor Type",
                                        ["Returning_Visitor", "New_Visitor", "Other"])
            submitted = st.form_submit_button("⚡ Predict Conversion Probability",
                                              use_container_width=True)

    with out_col:
        if submitted:
            sample = pd.DataFrame([{
                "Administrative":          in_admin,
                "Administrative_Duration": in_admin_dur,
                "Informational":           in_info,
                "Informational_Duration":  in_info_dur,
                "ProductRelated":          in_prod,
                "ProductRelated_Duration": in_prod_dur,
                "BounceRates":             in_bounce,
                "ExitRates":               in_exit,
                "PageValues":              in_pageval,
                "SpecialDay":              in_specday,
                "Month":                   in_month,
                "VisitorType":             in_visitor,
            }])
            prob = float(model.predict_proba(sample)[0][1])
            pct  = prob * 100

            if pct < 30:
                bar_color, zone, icon = "#ef4444", "Low Intent", "🚨"
            elif pct < 60:
                bar_color, zone, icon = "#f59e0b", "Moderate Intent", "💡"
            else:
                bar_color, zone, icon = "#22c55e", "High Intent", "✅"

            fig_g = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=pct,
                number={"suffix": "%", "valueformat": ".1f",
                        "font": {"size": 40, "color": "#e2e8f0"}},
                delta={"reference": 15.5, "suffix": "%", "valueformat": ".1f"},
                title={"text": "Conversion Probability",
                       "font": {"color": "#94a3b8", "size": 15}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#475569"},
                    "bar":  {"color": bar_color},
                    "bgcolor": "#1e293b",
                    "bordercolor": "#334155",
                    "steps": [
                        {"range": [0,  30],  "color": "rgba(239,68,68,.14)"},
                        {"range": [30, 60],  "color": "rgba(245,158,11,.14)"},
                        {"range": [60, 100], "color": "rgba(34,197,94,.14)"},
                    ],
                    "threshold": {"line": {"color": "#fff", "width": 2},
                                  "thickness": 0.7, "value": pct},
                },
            ))
            fig_g.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#cbd5e1",
                                height=320, margin=dict(t=20, b=0, l=20, r=20))
            st.plotly_chart(fig_g, use_container_width=True)

            st.markdown(f"""<div style="text-align:center;margin-top:.3rem;">
            <span style="font-size:1.1rem;font-weight:600;color:#e2e8f0;">{icon} {zone}</span>
            </div>""", unsafe_allow_html=True)

            action_map = {
                "🚨": ("rgba(239,68,68,.1)", "#ef4444", "Exit-Intent Discount",
                        "Very low conversion likelihood. Trigger an exit-intent popup with a "
                        "15% discount code to attempt last-second cart recovery."),
                "💡": ("rgba(245,158,11,.1)", "#f59e0b", "Gentle Nudge",
                        "Moderate intent detected. Show low-stock messaging, free-shipping "
                        "banners, or social proof to tip the purchase decision."),
                "✅": ("rgba(34,197,94,.1)", "#22c55e", "Friction-Free Checkout",
                        "High intent confirmed. Suppress all popups. Ensure a fast, "
                        "seamless checkout — any friction now risks losing the sale."),
            }
            bg, border, title, desc = action_map[icon]
            st.markdown(f"""
            <div style="background:{bg};border:1px solid {border};border-radius:12px;
                        padding:1.1rem;margin-top:.9rem;">
              <div style="font-weight:700;color:{border};font-size:.95rem;
                          margin-bottom:5px;">Recommended: {title}</div>
              <div style="color:#cbd5e1;font-size:.84rem;line-height:1.6;">{desc}</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""<div style="background:#1e293b;border:2px dashed #334155;
                border-radius:16px;padding:3rem;text-align:center;margin-top:.5rem;">
              <div style="font-size:2.5rem;margin-bottom:.5rem;">🎯</div>
              <div style="color:#64748b;font-size:.9rem;">
                Fill in the session parameters on the left<br>
                and click <b style="color:#0ea5e9;">Predict Conversion Probability</b>
              </div>
            </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# PAGE: ABOUT
# ═══════════════════════════════════════════════════════════════
elif page == "about":
    st.markdown('<div class="section-header">About FunnelIQ</div>', unsafe_allow_html=True)
    st.markdown("""<div class="insight-box" style="font-size:.93rem;">
    <b>FunnelIQ</b> is an AI-powered e-commerce analytics dashboard demonstrating the integration
    of funnel analytics, machine learning, and interactive data visualization using a real-world
    dataset. The goal is to help e-commerce teams understand purchase intent and reduce cart
    abandonment through evidence-based interventions.
    </div>""", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('<div class="section-header">📦 Dataset Details</div>', unsafe_allow_html=True)
    dc1, dc2 = st.columns([1.5, 1])
    with dc1:
        st.markdown("""
| Attribute | Detail |
|---|---|
| **Name** | Online Shoppers Purchasing Intention Dataset |
| **Source** | UCI Machine Learning Repository |
| **Rows** | 12,330 sessions |
| **Features** | 18 (10 numeric, 8 categorical/boolean) |
| **Target** | `Revenue` — whether the session ended in a purchase |
| **Class Balance** | ~84.5% Non-Conversion, ~15.5% Conversion |
| **Collected From** | Real e-commerce platform over 1 year |
        """)
    with dc2:
        conv_cnt = df_raw["Revenue"].value_counts().reset_index()
        conv_cnt.columns = ["Revenue", "Count"]
        conv_cnt["Label"] = conv_cnt["Revenue"].map({0: "Not Converted", 1: "Converted"})
        fig_pie = px.pie(
            conv_cnt, values="Count", names="Label",
            color_discrete_sequence=["#334155", "#0ea5e9"],
            hole=0.55, title="Class Distribution",
        )
        fig_pie.update_traces(textinfo="label+percent", textfont_size=11)
        fig_pie.update_layout(**PLOT_BG, height=250,
                              margin=dict(t=40, b=0, l=0, r=0),
                              legend=dict(bgcolor="rgba(0,0,0,0)"),
                              title_font_color="#e2e8f0")
        st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("**Feature Glossary**")
    feat_table = pd.DataFrame({
        "Feature": ["Administrative", "Administrative_Duration", "Informational",
                    "Informational_Duration", "ProductRelated", "ProductRelated_Duration",
                    "BounceRates", "ExitRates", "PageValues", "SpecialDay",
                    "Month", "OperatingSystems", "Browser", "Region",
                    "TrafficType", "VisitorType", "Weekend", "Revenue"],
        "Type": ["int","float","int","float","int","float","float","float","float",
                 "float","str","int","int","int","int","str","bool","bool (target)"],
        "Description": [
            "# of administrative pages visited (account, policies, etc.)",
            "Total time on admin pages (seconds)",
            "# of informational pages visited (FAQs, guides)",
            "Total time on informational pages (seconds)",
            "# of product-related pages visited",
            "Total time on product pages (seconds)",
            "Avg bounce rate of pages visited in the session",
            "Avg exit rate of pages visited in the session",
            "Avg Google Analytics page value of visited pages",
            "Proximity of session date to a major shopping event (0–1)",
            "Month of the year the session occurred",
            "Operating system ID of the user",
            "Browser ID of the user",
            "Geographic region ID",
            "Traffic type / referral channel ID",
            "Returning_Visitor, New_Visitor, or Other",
            "True if the session occurred on a weekend",
            "True if the session resulted in a purchase (TARGET)",
        ],
    })
    st.dataframe(feat_table, use_container_width=True, hide_index=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">🗺️ Dashboard Features</div>', unsafe_allow_html=True)
    fc = st.columns(4)
    features = [
        ("🏠 Overview", "KPI cards from real data, monthly trend lines, traffic-source bubble charts, and page-value box plots for an executive health snapshot."),
        ("📉 Funnel Analysis", "5-stage drop-off funnel from Site Entry to Conversion, plus segment breakdowns by weekend, bounce rate, and operating system."),
        ("🤖 Model & Insights", "Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix, probability distribution, and color-coded feature coefficient chart."),
        ("🔮 Prediction Tool", "Enter any session's attributes to get a real-time probability gauge plus an actionable business recommendation."),
    ]
    for col_w, (title, desc) in zip(fc, features):
        col_w.markdown(f"""<div class="about-feature-card">
        <h4>{title}</h4><p>{desc}</p></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">⚙️ Technology Stack</div>', unsafe_allow_html=True)
    tc = st.columns(5)
    tech = [("🐍 Python 3.12", "Core language"), ("📊 Streamlit", "Dashboard framework"),
            ("🤖 scikit-learn", "ML pipeline"),  ("📈 Plotly", "Interactive charts"),
            ("🐼 Pandas / NumPy", "Data processing")]
    for cw, (name, role) in zip(tc, tech):
        cw.markdown(f"""<div class="kpi-card">
        <div style="font-size:1.4rem;">{name.split()[0]}</div>
        <div class="kpi-label" style="color:#e2e8f0;font-size:.85rem;">{' '.join(name.split()[1:])}</div>
        <div class="kpi-label">{role}</div></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">👥 Authors</div>', unsafe_allow_html=True)
    a1, a2 = st.columns(2)
    for cw, name, roll in [(a1,"Aishwarya Gawali","16014223007"),
                            (a2,"Ankon Mukherjee","16014223014")]:
        cw.markdown(f"""
        <div style="background:#1e293b;border:1px solid #334155;border-radius:14px;
                    padding:1.3rem;text-align:center;">
          <div style="font-size:2rem;">👤</div>
          <div style="font-size:1rem;font-weight:700;color:#e2e8f0;">{name}</div>
          <div style="color:#0ea5e9;font-size:.88rem;font-weight:600;">{roll}</div>
          <div style="color:#64748b;font-size:.8rem;margin-top:4px;">
            Batch A1 · AI-DS Department<br>K. J. Somaiya College of Engineering, Mumbai
          </div>
        </div>""", unsafe_allow_html=True)
