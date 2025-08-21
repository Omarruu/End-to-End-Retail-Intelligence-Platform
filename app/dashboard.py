import os
import io
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
import networkx as nx
from datetime import datetime
from google.cloud import storage

# ─── Config ───────────────────────────────────────────────────────────────────
GCS_BUCKET = os.environ.get("GCS_BUCKET")  # if set, dashboard will read processed assets from GCS
GCS_PREFIX = "processed"
LOCAL_PROCESSED_DIR = "processed"

# ─── Page State ────────────────────────────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = "Overview"

if "last_loaded_forecast" not in st.session_state:
    st.session_state.last_loaded_forecast = None
    
def navigate_to(p):
    st.session_state.page = p

# ─── Utilities: fetch parquet (GCS or local) ────────────────────────────────────
def _download_blob_to_bytes(bucket_name, blob_name):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    if not blob.exists():
        raise FileNotFoundError(f"gs://{bucket_name}/{blob_name} not found")
    data = blob.download_as_bytes()
    return data, blob.updated

def _read_parquet_from_source(path, use_gcs_when_available=True):
    """
    If GCS_BUCKET is set, try reading from GCS at <GCS_PREFIX>/<path>.
    Otherwise read from local ./processed/<path>.
    Returns (df, updated_timestamp) where updated_timestamp may be None for local files.
    """
    # prefer GCS
    if use_gcs_when_available and GCS_BUCKET:
        full_blob = f"{GCS_PREFIX}/{path}"
        try:
            data, updated = _download_blob_to_bytes(GCS_BUCKET, full_blob)
            df = pd.read_parquet(io.BytesIO(data))
            return df, updated
        except Exception as e:
            st.warning(f"Failed to load from GCS ({full_blob}): {e}. Falling back to local.")
    # fallback local
    local_path = os.path.join(LOCAL_PROCESSED_DIR, path)
    if not os.path.exists(local_path):
        return None, None
    df = pd.read_parquet(local_path)
    return df, None

# ─── Data Loaders ───────────────────────────────────────────────────────────────
@st.cache_data(ttl=24*60*60)  # cache for 24 hours
def load_transactions():
    # keep original behavior (local dataset). When deployed, you can replace cleaned parquet with GCS usage
    local = "data/cleaned_online_retail.parquet"
    if GCS_BUCKET:
        # attempt to read the raw transactions from GCS if you uploaded it there (optional)
        try:
            data, _ = _download_blob_to_bytes(GCS_BUCKET, "raw/cleaned_online_retail.parquet")
            return pd.read_parquet(io.BytesIO(data))
        except Exception:
            # fallback to local file
            pass
    return pd.read_parquet(local)

@st.cache_data(ttl=24*60*60)
def load_precomputed_forecast():
    """
    Loads the precomputed artifacts written by process_daily.py:
      - daily.parquet
      - hist_fc.parquet
      - future_fc.parquet
      - hist.parquet
      - anoms.parquet

    Returns: (daily, hist_fc, future_fc, hist, anoms, loaded_timestamp)
    """
    required = [
        ("daily.parquet", "daily"),
        ("hist_fc.parquet", "hist_fc"),
        ("future_fc.parquet", "future_fc"),
        ("hist.parquet", "hist"),
        ("anoms.parquet", "anoms"),
    ]
    results = {}
    loaded_ts = None
    for fname, key in required:
        df, updated = _read_parquet_from_source(fname)
        if df is None:
            # If any artifact missing, return None -> page will show warning
            return None, None, None, None, None, None
        results[key] = df
        if updated:
            # pick the latest updated timestamp among blobs (GCS)
            if (loaded_ts is None) or (updated > loaded_ts):
                loaded_ts = updated

    # ensure ds columns are datetimes in case read as object
    for k in ["daily", "hist_fc", "future_fc", "hist", "anoms"]:
        if "ds" in results[k].columns:
            results[k]["ds"] = pd.to_datetime(results[k]["ds"])

    return results["daily"], results["hist_fc"], results["future_fc"], results["hist"], results["anoms"], loaded_ts

@st.cache_data(ttl=24*60*60)
def load_rfm_and_profiles():
    rfm_df     = pd.read_csv("data/customer_rfm_clusters.csv")
    profile_df = pd.read_csv("data/cluster_profile.csv")
    profile_df["SegmentName"] = profile_df["Cluster"].map({
        2: "VIPs",
        3: "Loyal High-Value",
        0: "Moderate",
        1: "At-Risk/Lapsed"
    })
    return rfm_df, profile_df

@st.cache_data(ttl=24*60*60)
def load_product_segments():
    sku_df     = pd.read_csv("data/product_segments.csv")
    profile_df = pd.read_csv("data/product_segment_profile.csv")
    profile_df["SegmentName"] = profile_df["Segment"].map({
        0: "Steady Sellers",
        1: "Dead Stock",
        2: "Super Performers",
        3: "Niche Big-Tickets"
    })
    return sku_df, profile_df

@st.cache_data(ttl=24*60*60)
def load_rules(path="data/association_rules.csv"):
    df = pd.read_csv(path)
    df["antecedents"] = df["antecedents"].apply(eval)
    df["consequents"] = df["consequents"].apply(eval)
    return df

@st.cache_data(ttl=24*60*60)
def get_transactions_for_date(date):
    df = load_transactions()
    # allow selection via date object or string
    if isinstance(date, str):
        dt = pd.to_datetime(date).date()
    elif isinstance(date, datetime):
        dt = date.date()
    else:
        dt = date
    return df[df["InvoiceDay"] == pd.to_datetime(dt).date()]

# ─── UI: Force refresh helper ───────────────────────────────────────────────────
def force_refresh_all():
    st.cache_data.clear()
    st.session_state.last_loaded_forecast = None
    st.experimental_rerun()

# Put Force Refresh in a fixed place (top-right)
with st.sidebar:
    if st.button(" 🔄 Force refresh"):
        force_refresh_all()

# ─── Page: Overview ─────────────────────────────────────────────────────────────
def page_overview():
    df = load_transactions()
    st.title("📊 Online Retail Demo Dashboard")
    st.metric("Total Revenue (GBP)", f"{df.TotalSales.sum():,.0f}")
    st.metric("Total Orders",        f"{df.InvoiceNo.nunique():,}")
    st.metric("Unique Customers",     f"{df.CustomerID.nunique():,}")
    st.subheader("Daily Sales Over Time")
    daily_ts = df.groupby("InvoiceDay")["TotalSales"].sum().reset_index()
    st.line_chart(daily_ts.rename(columns={"InvoiceDay":"index"}).set_index("index"))
    st.subheader("Top 10 Products by Revenue")
    top10 = df.groupby("Description")["TotalSales"].sum().nlargest(10)
    st.bar_chart(top10)
    st.subheader("Sales Heatmap (Weekday vs Hour)")
    df["Weekday"] = pd.Categorical(
        df.InvoiceDate.dt.day_name(),
        categories=["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"],
        ordered=True
    )
    df["Hour"] = df.InvoiceDate.dt.hour
    heat = df.pivot_table(index="Weekday", columns="Hour", values="TotalSales", aggfunc="sum").fillna(0)
    fig, ax = plt.subplots(figsize=(12,6))
    im = ax.imshow(heat, aspect="auto")
    ax.set_xticks(range(0,24,2)); ax.set_xticklabels(range(0,24,2))
    ax.set_yticks(range(len(heat.index))); ax.set_yticklabels(heat.index)
    ax.set_title("Sales Heatmap by Weekday and Hour")
    ax.set_xlabel("Hour of Day"); ax.set_ylabel("Weekday")
    fig.colorbar(im, ax=ax, label="Total Sales")
    st.pyplot(fig)

# ─── Page: Forecast & Anomalies ────────────────────────────────────────────────
def page_forecast():
    st.header("🔮 Forecast & Anomaly Alerts")
    daily, hist_fc, future_fc, hist, anoms, updated = load_precomputed_forecast()

    if daily is None:
        st.warning("Processed forecast/artifacts not found. Run the daily processor (process_daily.py) locally or deploy it to the cloud and set GCS_BUCKET.")
        return

    cutoff = daily["ds"].max() if "ds" in daily.columns else None
    hist_fc["type"], future_fc["type"] = "Fitted (history)", "Forecast (30d)"
    plot_df = pd.concat([hist_fc, future_fc])

    fig_fc = px.line(plot_df, x="ds", y="yhat", color="type",
                     labels={"ds":"Date","yhat":"Sales (GBP)"},
                     title="Daily Sales Forecast", template="plotly_dark", height=600)
    fig_fc.add_trace(go.Scatter(x=daily["ds"], y=daily["y"], mode="markers", name="Actual",
                                marker=dict(color="white",size=6), opacity=0.7))
    if cutoff is not None:
        fig_fc.add_shape(type="line", x0=cutoff, x1=cutoff, y0=0, y1=1,
                         xref="x", yref="paper", line=dict(color="lightgray",dash="dash"))
        fig_fc.add_annotation(x=cutoff, y=1.02, xref="x", yref="paper",
                              text="Cutoff", showarrow=False, font=dict(color="lightgray"))

    # uncertainty bands (if available)
    if "yhat_upper" in hist_fc.columns and "yhat_lower" in hist_fc.columns:
        fig_fc.add_traces([
            go.Scatter(name="History Uncertainty",
                       x=hist_fc["ds"].tolist()+hist_fc["ds"].iloc[::-1].tolist(),
                       y=hist_fc["yhat_upper"].tolist()+hist_fc["yhat_lower"].iloc[::-1].tolist(),
                       fill="toself", fillcolor="rgba(51,207,255,0.2)",
                       line=dict(color="rgba(0,0,0,0)"), hoverinfo="skip", showlegend=False),
        ])
    if "yhat_upper" in future_fc.columns and "yhat_lower" in future_fc.columns:
        fig_fc.add_traces([
            go.Scatter(name="Forecast Uncertainty",
                       x=future_fc["ds"].tolist()+future_fc["ds"].iloc[::-1].tolist(),
                       y=future_fc["yhat_upper"].tolist()+future_fc["yhat_lower"].iloc[::-1].tolist(),
                       fill="toself", fillcolor="rgba(255,170,0,0.2)",
                       line=dict(color="rgba(0,0,0,0)"), hoverinfo="skip", showlegend=False)
        ])

    st.plotly_chart(fig_fc, use_container_width=True)

    fig_ra = go.Figure()
    fig_ra.add_trace(go.Scatter(x=hist["ds"], y=hist["residual"], mode="lines", name="Residual",
                                line=dict(color="#33CFFF",width=2)))
    fig_ra.add_trace(go.Scatter(x=anoms["ds"], y=anoms["residual"], mode="markers", name="Anomaly",
                                marker=dict(color="#FF5555",size=8)))
    fig_ra.update_layout(title="Residuals & Detected Anomalies", template="plotly_dark",
                         plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                         font_color="white", xaxis=dict(gridcolor="gray"),
                         yaxis=dict(gridcolor="gray"), hovermode="x unified", height=500)
    st.plotly_chart(fig_ra, use_container_width=True)

    st.subheader("Top 5 Anomalous Days")
    top5 = (anoms[["ds","y","yhat","residual"]]
            .sort_values("residual", key=abs, ascending=False)
            .head(5)
            .rename(columns={"ds":"Date","y":"Actual","yhat":"Predicted","residual":"Residual"}))
    st.table(top5)

    st.subheader("🔍 Drill into Anomalous Transactions")
    anoms["DateOnly"] = anoms["ds"].dt.date
    top_pos = anoms[anoms["residual"]>0].nlargest(5,"residual")
    top_neg = anoms[anoms["residual"]<0].nsmallest(5,"residual")
    mode = st.radio("Choose anomaly type:",("Above-Expectation","Below-Expectation"))
    choices = (top_pos["DateOnly"].tolist() if mode.startswith("Above") else top_neg["DateOnly"].tolist())
    if len(choices) == 0:
        st.info("No anomalies found of the selected type.")
        return
    sel = st.selectbox("Select date to inspect:", choices)
    tx = get_transactions_for_date(sel)
    st.markdown(f"### Transactions on {sel}")
    st.markdown(
        f"- Number of orders: **{tx.InvoiceNo.nunique()}**  \n"
        f"- Total transactions: **{len(tx)}**  \n"
        f"- Total sales: **£{tx.TotalSales.sum():,.2f}**  \n"
        f"- Avg. order value: **£{tx.TotalSales.sum()/tx.InvoiceNo.nunique():,.2f}**"
    )
    st.dataframe(tx[["InvoiceNo","StockCode","Description","Quantity","UnitPrice","TotalSales"]],
                 use_container_width=True)

# ─── Page: Customer Segments ────────────────────────────────────────────────────
def page_customers():
    rfm_df, profile_df = load_rfm_and_profiles()
    st.header("👥 Customer Segmentation (RFM + K-Means)")
    fig = px.scatter(rfm_df, x="Recency", y="Monetary", color="Cluster",
                     hover_data=["CustomerID","Frequency"],
                     title="Recency vs Monetary by Cluster", template="plotly_dark", height=600)
    fig.update_layout(xaxis_title="Recency (days since last purchase)",
                      yaxis_title="Monetary (£ spent)")
    st.plotly_chart(fig, use_container_width=True)
    st.subheader("Cluster Profiles")
    st.table(profile_df)
    st.subheader("🎯 Target a Segment")
    seg_names = profile_df["SegmentName"].tolist()
    sel_name  = st.selectbox("Choose a segment:", seg_names)
    cid       = profile_df.loc[profile_df["SegmentName"]==sel_name,"Cluster"].iloc[0]
    seg_customers = rfm_df[rfm_df["Cluster"]==cid]
    prof      = profile_df.query("Cluster==@cid").iloc[0]
    st.markdown(f"**Segment:** {sel_name}")
    st.markdown(
        f"- **Size:** {int(prof.Count)} customers  \n"
        f"- **Avg Recency:** {prof.AvgRecency:.1f} days  \n"
        f"- **Avg Frequency:** {prof.AvgFrequency:.1f} orders  \n"
        if False else None  # placeholder to avoid formatting issues in some editors
    )
    st.markdown(
        f"- **Avg Frequency:** {prof.AvgFrequency:.1f} orders  \n"
        f"- **Avg Monetary:** £{prof.AvgMonetary:,.2f}"
    )
    st.markdown("**Sample Customers:**")
    st.dataframe(seg_customers.head(10))
    if st.button(f"📧 Send Special Offer to {sel_name}"):
        st.success(f"Offer emailed to {prof.Count} {sel_name}!")

# ─── Page: Product Segmentation ─────────────────────────────────────────────────
def page_products():
    sku_df, profile_df = load_product_segments()
    st.header("📦 Product Segmentation")
    sku_df["SegmentName"] = sku_df["Segment"].map(profile_df.set_index("Segment")["SegmentName"])
    color_map = {
        "Steady Sellers":    "#33CFFF",
        "Dead Stock":        "#FF5555",
        "Super Performers":  "#AAFF33",
        "Niche Big-Tickets": "#FFAA00",
    }
    fig = px.scatter(sku_df, x="UnitsSold", y="AvgPrice", color="SegmentName",
                     color_discrete_map=color_map,
                     hover_data=["StockCode","Description","TotalRevenue","Recency","SalesFreq"],
                     title="Units Sold vs Average Price by Product Segment",
                     template="plotly_dark", height=600)
    fig.update_layout(xaxis_title="Units Sold", yaxis_title="Average Price (£)",
                      legend_title_text="Product Segment")
    st.plotly_chart(fig, use_container_width=True)
    st.subheader("Segment Profiles")
    st.table(profile_df.set_index("Segment")[["Count","AvgRevenue","AvgUnits","AvgRecency","AvgSalesDays","SegmentName"]])
    st.subheader("🔍 Explore SKUs in a Segment")
    seg_choice = st.selectbox("Choose a segment:", profile_df["SegmentName"])
    seg_id     = profile_df.loc[profile_df["SegmentName"]==seg_choice,"Segment"].iloc[0]
    seg_skus   = sku_df[sku_df["Segment"] == seg_id]
    st.markdown(f"**Showing {len(seg_skus)} SKUs in {seg_choice}**")
    st.dataframe(seg_skus[["StockCode","Description","TotalRevenue","UnitsSold","AvgPrice","Recency","SalesFreq"]],
                 use_container_width=True)
    st.subheader("🔍 Segment Interpretation")
    interpretations = {
        "Steady Sellers":    "Consistent revenue and volume—keep these reliably in stock.",
        "Dead Stock":        "Old, slow-moving products—consider clearance or bundles.",
        "Super Performers":  "High-velocity sellers—prioritize availability and marketing.",
        "Niche Big-Tickets": "Rare but high-value items—promote in targeted channels."
    }
    st.info(f"**{seg_choice}**: {interpretations[seg_choice]}")

# ─── Page: Market Basket Analysis ──────────────────────────────────────────────
def page_market_basket():
    st.header("🛒 Market Basket / Association Rules")
    rules_df = load_rules()

    # ─── Prepare Top-10 Rules ────────────────────────────────────────────────
    st.subheader("Top 10 Rules by Confidence")

    top10 = rules_df.nlargest(10, "confidence").copy()

    # 1) Convert frozensets to comma-joined strings
    top10["antecedents"]  = top10["antecedents"].apply(lambda s: ", ".join(sorted(s)))
    top10["consequents"]  = top10["consequents"].apply(lambda s: ", ".join(sorted(s)))

    # 2) Round numeric columns
    top10["support"]     = top10["support"].round(3)
    top10["confidence"]  = top10["confidence"].round(3)
    top10["lift"]        = top10["lift"].round(2)

    # 3) Rename for display
    display_df = top10[[
        "antecedents", "consequents", "support", "confidence", "lift"
    ]].rename(columns={
        "antecedents":   "If basket contains",
        "consequents":   "Then also contains",
        "support":       "Support",
        "confidence":    "Confidence",
        "lift":          "Lift"
    })

    # 4) Show as scrollable dataframe for better column widths
    st.dataframe(display_df, use_container_width=True, height=300)


    # ─── Recommendations by Item ─────────────────────────────────────────────
    st.subheader("💡 Get Recommendations")
    all_items = sorted({item for ants in rules_df["antecedents"] for item in ants})
    selected = st.selectbox("Select a product:", all_items)

    recs = (
        rules_df[rules_df["antecedents"].apply(lambda ants: selected in ants)]
        .sort_values("confidence", ascending=False)
        .head(5)
        .copy()
    )

    # Pre-format the same way
    recs["consequents"] = recs["consequents"].apply(lambda s: ", ".join(sorted(s)))
    recs["confidence"]  = recs["confidence"].round(3)
    recs["lift"]        = recs["lift"].round(2)

    if not recs.empty:
        st.markdown(f"**Customers who bought {selected} also bought:**")
        for _, r in recs.iterrows():
            st.markdown(
                f"- **{r['consequents']}** (conf={r['confidence']}, lift={r['lift']})"
            )
    else:
        st.info(f"No strong associations found for **{selected}**.")

    # ─── Association Network ────────────────────────────────────────────────
    st.subheader("🌐 Association Network")
    fig_net = plot_rule_network(rules_df, top_n=25)
    st.plotly_chart(fig_net, use_container_width=True, height=600)

def plot_rule_network(rules_df, top_n=25):
    # pick top_n rules
    sub = rules_df.nlargest(top_n, "confidence")
    G = nx.DiGraph()
    for _, row in sub.iterrows():
        for ant in row["antecedents"]:
            for cons in row["consequents"]:
                G.add_node(ant)
                G.add_node(cons)
                G.add_edge(ant, cons, weight=row["confidence"])
    pos = nx.spring_layout(G, k=1, seed=42)

    # build the single edge trace, now light-blue
    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(color="lightblue", width=1),
        hoverinfo="none"
    )

    # build the node trace, now lime-green
    node_x, node_y, node_text = [], [], []
    for n in G.nodes():
        x, y = pos[n]
        node_x.append(x)
        node_y.append(y)
        node_text.append(n)

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=node_text,
        textposition="top center",
        hoverinfo="text",
        marker=dict(
            color="limegreen",
            size=20,
            line_width=2
        )
    )

    fig = go.Figure(
        [edge_trace, node_trace],
        layout=go.Layout(
            title="🕸️ Association Rules Network",
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            template="plotly_dark"
        )
    )

    return fig

# ─── Sidebar Navigation ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<h3 style='color:white; margin-top:0.5rem; margin-bottom:1rem; font-size:1.50rem;'>  🔗 Navigation Bar</h3>",
        unsafe_allow_html=True
    )
    page = option_menu(
        menu_title=None,
        options=["Overview","Forecast & Anomalies","Customer Segments","Product Segments","Market Basket Analysis"],
        icons=["house","graph-up","people-fill","box-seam","cart"],
        menu_icon="cast", default_index=0, orientation="vertical",
        styles={
            "container": {"padding":"0px","background-color":"#11111F","margin-top":"0px"},
            "nav-link": {"font-size":"1rem","text-align":"left","margin":"0.1rem 0","--hover-color":"#262730"},
            "nav-link-selected": {"background-color":"#0E1C36","font-weight":"bold"},
            "icon": {"font-size":"1.2rem"}
        }
    )
    st.markdown(
        "<div style='position:absolute; bottom:1px; width:100%; text-align:center; color:#777; font-size:0.80rem;'>"
        "Dashboard © Omar Medhat</div>",
        unsafe_allow_html=True
    )

# ─── Main ───────────────────────────────────────────────────────────────────────
if page == "Overview":
    page_overview()
elif page == "Forecast & Anomalies":
    page_forecast()
elif page == "Customer Segments":
    page_customers()
elif page == "Product Segments":
    page_products()
elif page == "Market Basket Analysis":
    page_market_basket()
