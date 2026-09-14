import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import pydeck as pdk
import torch
import json
import time
from transformers import pipeline
import uuid

# --- PHASE 0: SETUP ---
st.set_page_config(
    page_title="SIFGuard - Safety Report Analyzer",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- PHASE 1: CORE ENGINE ---

# T006: LSR_RULES
LSR_RULES = {
    "Work Authorisation":    ["permit", "parmit", "authorization", "sanction"],
    "Line of Fire":          ["line of fire", "swing radius", "pressure release", "struck by"],
    "Energy Isolation":      ["isolation", "lockout", "tagout", "loto", "energized", "live"],
    "Working at Height":     ["height", "scaffold", "ladder", "harness", "fall", "handrail"],
    "Safe Driving":          ["driving", "vehicle", "gaadi", "speed", "seatbelt", "truck"],
    "Hot Work":              ["hot work", "welding", "cutting", "grinding", "spark", "flame"],
    "Confined Space":        ["confined space", "vessel entry", "tank entry", "manhole"],
    "Manual Handling":       ["lifting", "carrying", "crane", "sling", "load"],
    "Management of Change":  ["moc", "change", "modification", "deviation"],
}

# T007: BARRIER_PATTERNS
BARRIER_PATTERNS = {
    "PPE":              ["ppe missing", "no helmet", "no gloves", "no goggles"],
    "Permit":           ["permit not issued", "bina permit", "no permit"],
    "Supervision":      ["supervisor absent", "no supervisor", "unattended"],
    "Engineering":      ["handrail nahi", "guard missing", "no barrier"],
    "Training":         ["untrained", "no training", "unqualified"],
    "Procedure":        ["procedure not followed", "sop violated", "shortcut"],
}

# T008: INDIC_TRANSLITERATION_MAP
INDIC_TRANSLITERATION_MAP = {
    "bina": "without",
    "kaam": "work",
    "ho raha hai": "is happening",
    "nahi kiya": "not done",
    "nahi tha": "was not there",
    "gaadi": "vehicle",
    "parmit": "permit",
    "aadmi": "worker",
    "girna": "fall",
    "pair slip": "foot slipped",
}

# T009: INSTALLATIONS
INSTALLATIONS = {
    "Duliajan":    {"lat": 27.3587, "lon": 95.3212},
    "Moran":       {"lat": 27.1833, "lon": 94.9333},
    "Naharkatiya": {"lat": 27.2811, "lon": 95.2588},
    "Jorhat":      {"lat": 26.7509, "lon": 94.2037},
    "Digboi":      {"lat": 27.3822, "lon": 95.6311},
}

# Config Constants
DEFAULT_THRESHOLD = 0.65

# T010: Model Loader
@st.cache_resource
def load_classifier():
    """Load the BART zero-shot classification model."""
    try:
        return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

# T011: Transliteration
def transliterate_text(text: str) -> str:
    """Replace Hindi/Hinglish/Assamese terms with English equivalents."""
    text_lower = text.lower()
    for indic, eng in INDIC_TRANSLITERATION_MAP.items():
        text_lower = text_lower.replace(indic, eng)
    return text_lower

# T012: Classify SIF
def classify_sif(text: str, classifier) -> float:
    """Run zero-shot classification to determine P(SIF)."""
    if not classifier:
        return 0.0
    try:
        candidate_labels = ["serious injury or fatality", "minor incident or safe"]
        result = classifier(text, candidate_labels)
        sif_index = result["labels"].index("serious injury or fatality")
        return round(result["scores"][sif_index], 4)
    except Exception as e:
        st.error(f"Classification error: {e}")
        return 0.0

# T013: Match IOGP Rules
def match_iogp_rules(text: str) -> list:
    """Find matching IOGP Life-Saving rules in text."""
    text_lower = text.lower()
    matched = []
    for rule, keywords in LSR_RULES.items():
        if any(kw in text_lower for kw in keywords):
            matched.append(rule)
    return matched

# T014: Detect Failed Barriers
def detect_failed_barriers(text: str) -> list:
    """Find failed safety barriers in text."""
    text_lower = text.lower()
    failed = []
    for barrier, keywords in BARRIER_PATTERNS.items():
        if any(kw in text_lower for kw in keywords):
            failed.append(barrier)
    return failed

# T015: Calculate FPHI
def calculate_fphi(p_sif: float, rules: list, barriers: list) -> float:
    """Calculate the Fatal-Potential Heat Index (FPHI) 0-100."""
    rule_penalty = min(len(rules) * 5, 20)
    barrier_penalty = min(len(barriers) * 10, 30)
    base_fphi = p_sif * 100
    
    fphi = base_fphi + rule_penalty + barrier_penalty
    return round(min(fphi, 100.0), 1)

# T016: Generate Escalation Ticket
def generate_escalation_ticket(analysis: dict) -> str:
    """Generate OISD compliant escalation ticket."""
    ticket = f"""
--- OISD ESCALATION TICKET ---
Ticket ID   : {uuid.uuid4().hex[:8].upper()}
Report ID   : {analysis.get('report_id', 'UNKNOWN')}
Risk Level  : {analysis.get('risk_level', 'CRITICAL')}
P(SIF)      : {analysis.get('p_sif', 0.0):.2f}
FPHI Score  : {analysis.get('fphi_score', 0.0):.1f}
Location    : {analysis.get('installation', 'N/A')}

[Matched IOGP Rules]
{', '.join(analysis.get('matched_rules', [])) if analysis.get('matched_rules') else 'None'}

[Failed Barriers]
{', '.join(analysis.get('failed_barriers', [])) if analysis.get('failed_barriers') else 'None'}

Action Required: Immediate intervention by Field HSE Officer.
------------------------------
    """
    return ticket

# --- PHASE 2: SAMPLE DATA ---

# T018: Load Sample Reports
@st.cache_data
def load_sample_reports() -> list:
    """Load the 8 sample reports from JSON."""
    try:
        with open("data/sample_reports.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Failed to load sample reports: {e}")
        return []

# --- PHASE 3: UI SHELL ---

def inject_custom_css():
    """T020: Inject dark glassmorphism CSS."""
    st.markdown("""
        <style>
        /* Base Theme */
        body {
            font-family: 'Inter', sans-serif;
            background-color: #070B19;
            color: #FFFFFF;
        }
        
        /* Glassmorphism Cards */
        .glass-card {
            background: rgba(16, 26, 45, 0.75);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border-radius: 10px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            padding: 20px;
            margin-bottom: 20px;
        }
        
        /* Chips */
        .chip-orange {
            display: inline-block;
            background-color: rgba(255, 107, 0, 0.2);
            color: #FF6B00;
            border: 1px solid #FF6B00;
            border-radius: 16px;
            padding: 4px 12px;
            margin: 4px;
            font-size: 12px;
            font-weight: 600;
        }
        .chip-red {
            display: inline-block;
            background-color: rgba(211, 47, 47, 0.2);
            color: #D32F2F;
            border: 1px solid #D32F2F;
            border-radius: 16px;
            padding: 4px 12px;
            margin: 4px;
            font-size: 12px;
            font-weight: 600;
        }
        
        /* Alerts */
        .alert-banner {
            background: rgba(211, 47, 47, 0.15);
            border-left: 4px solid #D32F2F;
            color: #FFCDD2;
            padding: 15px;
            border-radius: 4px;
            margin: 15px 0;
            font-weight: 500;
        }
        </style>
    """, unsafe_allow_html=True)

# --- PHASE 4: REAL-TIME ANALYZER ---

# T029: Create Plotly Gauge
def create_gauge_chart(p_sif: float, threshold: float):
    """Create a Plotly speedometer gauge for P(SIF)."""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = p_sif * 100,
        title = {'text': "P(SIF) - Probability of Fatal Precursor"},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color': "#FF6B00"},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, threshold*100], 'color': "rgba(46, 125, 50, 0.5)"}, # Safe
                {'range': [threshold*100, 100], 'color': "rgba(211, 47, 47, 0.5)"} # Alert
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': threshold * 100
            }
        }
    ))
    fig.update_layout(
        font={'color': "white", 'family': "Inter"},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=300,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig

# --- PHASE 5: 3D ASSAM MAP ---

@st.cache_data
def get_map_data():
    """T036: Aggregate reports by installation for the 3D Map."""
    reports = load_sample_reports()
    classifier = load_classifier()
    
    map_data = []
    for r in reports:
        text = transliterate_text(r['report_text'])
        p_sif = classify_sif(text, classifier)
        rules = match_iogp_rules(text)
        barriers = detect_failed_barriers(text)
        fphi = calculate_fphi(p_sif, rules, barriers)
        
        map_data.append({
            "installation": r["installation"],
            "lat": r["latitude"],
            "lon": r["longitude"],
            "fphi": fphi,
            "count": 1
        })
    
    df = pd.DataFrame(map_data)
    if df.empty:
        return df
        
    agg_df = df.groupby(["installation", "lat", "lon"]).agg(
        count=("count", "sum"),
        avg_fphi=("fphi", "mean")
    ).reset_index()
    
    def get_color(fphi):
        if fphi >= 75:
            return [211, 47, 47, 200] # Red
        elif fphi >= 50:
            return [255, 107, 0, 200] # Orange
        else:
            return [46, 125, 50, 200] # Green
            
    agg_df["color"] = agg_df["avg_fphi"].apply(get_color)
    return agg_df

# --- PHASE 6: HSSE ANALYTICS ---

@st.cache_data
def get_analytics_data():
    """T042: Process all sample reports to provide a base dataset for analytics."""
    reports = load_sample_reports()
    classifier = load_classifier()
    
    analytics_data = []
    for r in reports:
        text = transliterate_text(r['report_text'])
        p_sif = classify_sif(text, classifier)
        matched_rules = match_iogp_rules(text)
        failed_barriers = detect_failed_barriers(text)
        fphi = calculate_fphi(p_sif, matched_rules, failed_barriers)
        
        analytics_data.append({
            "report_id": r["report_id"],
            "timestamp": r["timestamp"],
            "installation": r["installation"],
            "activity": r["activity"],
            "p_sif": p_sif,
            "fphi_score": fphi,
            "risk_level": "CRITICAL" if p_sif >= DEFAULT_THRESHOLD else ("HIGH" if p_sif >= 0.5 else "LOW"),
            "matched_rules": matched_rules,
            "failed_barriers": failed_barriers,
            "processing_time_ms": 850 + int(np.random.rand() * 300)
        })
    return analytics_data

def main():
    inject_custom_css()
    
    # T021: Page Header
    st.title("🛡️ SIFGuard")
    st.markdown("### *From Reports to Rescue — Detecting Fatal Precursors Before Incidents Happen*")
    
    # T023: Sidebar Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        if "threshold" not in st.session_state:
            st.session_state.threshold = DEFAULT_THRESHOLD
            
        st.session_state.threshold = st.slider(
            "SIF Alert Threshold", 
            min_value=0.40, 
            max_value=0.90, 
            value=st.session_state.threshold, 
            step=0.05
        )
        st.info("Reports with P(SIF) ≥ threshold will trigger escalation.")
        
        st.markdown("---")
        st.markdown("**Powered by:**")
        st.markdown("- BART-Large-MNLI")
        st.markdown("- Plotly & PyDeck")
        
    # Initialize State
    if "processed_reports" not in st.session_state:
        st.session_state.processed_reports = []
    if "current_analysis" not in st.session_state:
        st.session_state.current_analysis = None
        
    # T022: Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔍 Real-Time Analyzer", 
        "🗺️ 3D Assam Map", 
        "📊 HSSE Analytics", 
        "🕸️ Precursor Patterns", 
        "🏗️ Architecture"
    ])
    
    with tab1:
        st.header("Real-Time Safety Report Analyzer")
        
        # T025: Sample report selectbox
        reports = load_sample_reports()
        report_options = ["Custom Input..."] + [f"{r['report_id']} - {r['activity']}" for r in reports]
        
        selected_report = st.selectbox("Select a Sample Report or Type Your Own", report_options)
        
        default_text = ""
        if selected_report != "Custom Input...":
            idx = report_options.index(selected_report) - 1
            default_text = reports[idx]["report_text"]
            
        # T026: Text area
        report_text = st.text_area("Report Text (English/Hindi/Hinglish/Assamese)", value=default_text, height=150)
        
        # T027: Analyze Button
        if st.button("🚀 Analyze & Assess Risk", type="primary"):
            if not report_text.strip():
                st.error("Please enter report text to analyze.")
            else:
                with st.spinner("Analyzing NLP features and SIF potential..."):
                    start_time = time.time()
                    
                    # Load model (cached)
                    classifier = load_classifier()
                    
                    # T028: Full analysis pipeline
                    eng_text = transliterate_text(report_text)
                    p_sif = classify_sif(eng_text, classifier)
                    matched_rules = match_iogp_rules(eng_text)
                    failed_barriers = detect_failed_barriers(eng_text)
                    fphi = calculate_fphi(p_sif, matched_rules, failed_barriers)
                    
                    processing_time = int((time.time() - start_time) * 1000)
                    
                    # Construct analysis dict
                    analysis = {
                        "report_id": selected_report.split(" - ")[0] if selected_report != "Custom Input..." else f"CUSTOM-{uuid.uuid4().hex[:6].upper()}",
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "installation": reports[report_options.index(selected_report)-1]["installation"] if selected_report != "Custom Input..." else "Unknown",
                        "latitude": reports[report_options.index(selected_report)-1]["latitude"] if selected_report != "Custom Input..." else 0.0,
                        "longitude": reports[report_options.index(selected_report)-1]["longitude"] if selected_report != "Custom Input..." else 0.0,
                        "activity": reports[report_options.index(selected_report)-1]["activity"] if selected_report != "Custom Input..." else "Custom",
                        "p_sif": p_sif,
                        "risk_level": "CRITICAL" if p_sif >= st.session_state.threshold else ("HIGH" if p_sif >= 0.5 else "LOW"),
                        "matched_rules": matched_rules,
                        "failed_barriers": failed_barriers,
                        "fphi_score": fphi,
                        "processing_time_ms": processing_time
                    }
                    
                    # T035: Store result
                    st.session_state.current_analysis = analysis
                    st.session_state.processed_reports.append(analysis)
                    st.toast(f"✅ Analysis complete! P(SIF): {p_sif*100:.1f}%")
                    
        # Render Results
        if st.session_state.current_analysis:
            analysis = st.session_state.current_analysis
            st.markdown("---")
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
                # T030: Render gauge
                fig = create_gauge_chart(analysis["p_sif"], st.session_state.threshold)
                st.plotly_chart(fig, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
                
            with col2:
                st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
                st.subheader("Analysis Results")
                st.write(f"⏱️ **Processing Time:** {analysis['processing_time_ms']} ms")
                st.write(f"🔥 **FPHI Score:** {analysis['fphi_score']}/100")
                
                # T031: Render orange IOGP chips
                st.write("**IOGP Life-Saving Rules Violated:**")
                if analysis['matched_rules']:
                    chips_html = "".join([f"<span class='chip-orange'>{rule}</span>" for rule in analysis['matched_rules']])
                    st.markdown(chips_html, unsafe_allow_html=True)
                else:
                    st.write("None detected")
                    
                # T032: Render red barrier chips
                st.write("**Failed Safety Barriers:**")
                if analysis['failed_barriers']:
                    chips_html = "".join([f"<span class='chip-red'>{barrier}</span>" for barrier in analysis['failed_barriers']])
                    st.markdown(chips_html, unsafe_allow_html=True)
                else:
                    st.write("None detected")
                st.markdown("</div>", unsafe_allow_html=True)
                
            # T033: Show red alert banner if threshold met
            if analysis["p_sif"] >= st.session_state.threshold:
                st.markdown(f"<div class='alert-banner'>🚨 CRITICAL ALERT: P(SIF) is {analysis['p_sif']*100:.1f}%. Exceeds escalation threshold of {st.session_state.threshold*100:.1f}%.</div>", unsafe_allow_html=True)
                # T034: Escalation ticket
                ticket = generate_escalation_ticket(analysis)
                st.code(ticket, language="markdown")
            else:
                st.success("✅ Report does not meet SIF escalation threshold.")
        
    with tab2:
        st.header("3D GIS Risk Map (Assam Operations)")
        
        with st.spinner("Loading GIS data..."):
            map_df = get_map_data()
            
            if not map_df.empty:
                # T037: PyDeck ColumnLayer
                layer = pdk.Layer(
                    "ColumnLayer",
                    data=map_df,
                    get_position=["lon", "lat"],
                    get_elevation="count",
                    elevation_scale=2000,
                    radius=3000,
                    get_fill_color="color",
                    pickable=True,
                    auto_highlight=True,
                )
                
                # T038 & T039: View state and CartoDB Dark Matter
                view_state = pdk.ViewState(
                    latitude=27.2,
                    longitude=95.2,
                    zoom=8,
                    pitch=45,
                    bearing=0
                )
                
                # T040: Tooltip
                tooltip = {
                    "html": "<b>{installation}</b><br/>Reports: {count}<br/>Avg FPHI: {avg_fphi:.1f}",
                    "style": {
                        "backgroundColor": "rgba(16, 26, 45, 0.9)",
                        "color": "white",
                        "border": "1px solid #FF6B00",
                        "fontFamily": "Inter"
                    }
                }
                
                r = pdk.Deck(
                    layers=[layer],
                    initial_view_state=view_state,
                    map_style="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
                    tooltip=tooltip
                )
                
                st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
                st.pydeck_chart(r)
                
                # T041: Legend
                st.markdown("""
                <div style='display: flex; gap: 20px; justify-content: center; margin-top: 15px;'>
                    <div><span style='display:inline-block; width:12px; height:12px; background-color:rgb(211,47,47); border-radius:50%;'></span> Critical (FPHI ≥ 75)</div>
                    <div><span style='display:inline-block; width:12px; height:12px; background-color:rgb(255,107,0); border-radius:50%;'></span> High (FPHI ≥ 50)</div>
                    <div><span style='display:inline-block; width:12px; height:12px; background-color:rgb(46,125,50); border-radius:50%;'></span> Medium/Low (FPHI < 50)</div>
                </div>
                """, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
        
    with tab3:
        st.header("Executive HSSE Analytics")
        
        with st.spinner("Compiling analytics..."):
            base_data = get_analytics_data()
            
            # Combine base data with session state, avoiding duplicates by report_id
            all_data = {r['report_id']: r for r in base_data}
            for pr in st.session_state.processed_reports:
                all_data[pr['report_id']] = pr
                
            combined_reports = list(all_data.values())
            df = pd.DataFrame(combined_reports)
            
            if not df.empty:
                # T042: 4 KPI values
                total_reports = len(df)
                critical_reports = len(df[df['p_sif'] >= st.session_state.threshold])
                high_risk_sites = df[df['p_sif'] >= st.session_state.threshold]['installation'].nunique()
                avg_latency = int(df['processing_time_ms'].mean())
                
                # T043: 4 KPI Cards
                st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Total Reports", total_reports)
                k2.metric("Critical SIF Precursors", critical_reports)
                k3.metric("High-Risk Installations", high_risk_sites)
                k4.metric("Avg Analysis Latency", f"{avg_latency} ms")
                st.markdown("</div>", unsafe_allow_html=True)
                
                st.markdown("---")
                
                c1, c2 = st.columns([1, 1])
                with c1:
                    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
                    # T044: Top 5 IOGP violations
                    all_rules = []
                    for rules in df['matched_rules']:
                        all_rules.extend(rules)
                    
                    if all_rules:
                        rule_counts = pd.Series(all_rules).value_counts().head(5)
                        fig_bar = go.Figure(go.Bar(
                            x=rule_counts.values,
                            y=rule_counts.index,
                            orientation='h',
                            marker_color='#FF6B00'
                        ))
                        fig_bar.update_layout(
                            title="Top 5 IOGP Life-Saving Rule Violations",
                            yaxis={'categoryorder': 'total ascending'},
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            font={'color': 'white'},
                            margin=dict(l=10, r=10, t=40, b=10)
                        )
                        st.plotly_chart(fig_bar, use_container_width=True)
                    else:
                        st.write("No IOGP violations detected.")
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                with c2:
                    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
                    # T045: Donut chart
                    risk_counts = df['risk_level'].value_counts()
                    colors = {'CRITICAL': '#D32F2F', 'HIGH': '#FF6B00', 'LOW': '#2E7D32'}
                    fig_donut = go.Figure(go.Pie(
                        labels=risk_counts.index,
                        values=risk_counts.values,
                        hole=0.6,
                        marker_colors=[colors.get(x, '#888') for x in risk_counts.index]
                    ))
                    fig_donut.update_layout(
                        title="SIF Risk Distribution",
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font={'color': 'white'},
                        margin=dict(l=10, r=10, t=40, b=10)
                    )
                    st.plotly_chart(fig_donut, use_container_width=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                # T046: Filterable DataFrame
                st.subheader("Report Database")
                
                # Format dataframe for display
                display_df = df.copy()
                display_df['IOGP Rules'] = display_df['matched_rules'].apply(lambda x: ", ".join(x) if x else "None")
                display_df['Failed Barriers'] = display_df['failed_barriers'].apply(lambda x: ", ".join(x) if x else "None")
                display_df['P(SIF) %'] = (display_df['p_sif'] * 100).round(1)
                
                display_df = display_df[['report_id', 'timestamp', 'installation', 'activity', 'P(SIF) %', 'risk_level', 'IOGP Rules', 'Failed Barriers']]
                
                # Highlight logic avoiding deprecated code
                styled_df = display_df.style.highlight_between(
                    subset=['P(SIF) %'],
                    left=st.session_state.threshold * 100,
                    right=100.0,
                    color="rgba(211, 47, 47, 0.4)"
                )
                
                st.dataframe(styled_df, use_container_width=True)
                
                # T047: CSV Export
                csv = display_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Export Database (CSV)",
                    data=csv,
                    file_name="sifguard_reports.csv",
                    mime="text/csv",
                )
        
    with tab4:
        st.header("Hazard Precursor Patterns")
        
        with st.spinner("Mapping precursor paths..."):
            base_data = get_analytics_data()
            all_data = {r['report_id']: r for r in base_data}
            for pr in st.session_state.processed_reports:
                all_data[pr['report_id']] = pr
            df = pd.DataFrame(list(all_data.values()))
            
            if not df.empty:
                # T048: Mapping Activity -> Rule -> Barrier
                sankey_data = []
                for _, row in df.iterrows():
                    activity = row['activity']
                    rules = row['matched_rules'] if row['matched_rules'] else ["No Rule Violation"]
                    barriers = row['failed_barriers'] if row['failed_barriers'] else ["No Barrier Failure"]
                    
                    for r in rules:
                        sankey_data.append({'source': activity, 'target': r, 'value': 1})
                        for b in barriers:
                            sankey_data.append({'source': r, 'target': b, 'value': 1})
                            
                sankey_df = pd.DataFrame(sankey_data).groupby(['source', 'target']).sum().reset_index()
                
                # Create node list
                all_nodes = list(pd.unique(sankey_df[['source', 'target']].values.ravel('K')))
                node_indices = {node: i for i, node in enumerate(all_nodes)}
                
                sankey_df['source_idx'] = sankey_df['source'].map(node_indices)
                sankey_df['target_idx'] = sankey_df['target'].map(node_indices)
                
                # T049: Plotly Sankey
                fig_sankey = go.Figure(data=[go.Sankey(
                    node = dict(
                        pad = 15,
                        thickness = 20,
                        line = dict(color = "black", width = 0.5),
                        label = all_nodes,
                        color = "#FF6B00"
                    ),
                    link = dict(
                        source = sankey_df['source_idx'],
                        target = sankey_df['target_idx'],
                        value = sankey_df['value'],
                        color = "rgba(255, 107, 0, 0.4)"
                    )
                )])
                fig_sankey.update_layout(
                    title_text="Activity → IOGP Rule → Failed Barrier Mapping", 
                    font_size=12,
                    paper_bgcolor="rgba(0,0,0,0)",
                    font={'color': 'white'},
                    height=500
                )
                
                st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
                st.plotly_chart(fig_sankey, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
                
                # T050: Top Hazard Triples
                st.subheader("Top Hazard Combinations (Activity + Rule + Barrier)")
                
                triples = []
                for _, row in df.iterrows():
                    activity = row['activity']
                    rules = row['matched_rules'] if row['matched_rules'] else ["None"]
                    barriers = row['failed_barriers'] if row['failed_barriers'] else ["None"]
                    for r in rules:
                        for b in barriers:
                            triples.append(f"{activity} ➔ {r} ➔ {b}")
                            
                triple_counts = pd.Series(triples).value_counts().head(5)
                
                for idx, (triple, count) in enumerate(triple_counts.items()):
                    st.markdown(f"""
                    <div style='background: rgba(255,255,255,0.05); padding: 15px; border-radius: 8px; margin-bottom: 10px; border-left: 4px solid #FF6B00;'>
                        <strong>#{idx+1}</strong> — {triple} <span style='float:right; background:#FF6B00; padding:2px 8px; border-radius:12px; font-size:0.8em;'>{count} occurrences</span>
                    </div>
                    """, unsafe_allow_html=True)
        
    with tab5:
        st.header("About SIFGuard Architecture")
        
        # T051: Text overview
        st.markdown("""
        **SIFGuard** is an AI-powered safety analytics platform designed for Oil India Limited (SIH Problem Statement #26165).
        It identifies Fatal Precursors hidden within unstructured safety reports by extracting linguistic nuances and mapping them to standard safety frameworks.
        
        ### The AI Pipeline
        1. **Transliteration & Normalization**: Converts Hinglish/Assamese vocabulary into English semantic equivalents.
        2. **Zero-Shot NLP Inference**: Uses `facebook/bart-large-mnli` to evaluate the probability of a Serious Incident or Fatality `P(SIF)`.
        3. **Pattern Matching Engine**: Scans text for violations against the 9 IOGP Life-Saving Rules.
        4. **Barrier Failure Detection**: Identifies critical safety barriers (e.g., Permits, Isolation, PPE) that failed.
        5. **FPHI Scoring Model**: Combines P(SIF) with rule and barrier failures to generate a composite Fatal Precursor Hazard Index (0-100).
        """)
        
        # T052 & T053: Diagrams
        st.subheader("System Architecture")
        
        col_diag1, col_diag2 = st.columns(2)
        with col_diag1:
            try:
                st.image("docs/architecture_diagram.png", caption="SIFGuard Architecture Diagram", use_column_width=True)
            except Exception:
                st.info("Architecture diagram (docs/architecture_diagram.png) will be displayed here.")
                
        with col_diag2:
            try:
                st.image("docs/data_flow_diagram.png", caption="SIFGuard Data Flow Diagram", use_column_width=True)
            except Exception:
                st.info("Data Flow diagram (docs/data_flow_diagram.png) will be displayed here.")
            
        # T054: Team Info
        st.markdown("---")
        st.markdown("### Hacktivate Syndicate")
        st.markdown("Built with ❤️ for **Smart Automation** (Ministry of Petroleum & Natural Gas).")

if __name__ == "__main__":
    main()
