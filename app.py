import streamlit as st
import joblib
import pandas as pd
import plotly.graph_objects as go
import requests
from bs4 import BeautifulSoup
from datetime import datetime

# ==========================
# Load Model & Vectorizer
# ==========================

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# ==========================
# Page Config
# ==========================

st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="wide"
)

# ==========================
# Custom CSS
# ==========================

st.markdown("""
    <style>
        .main { background-color: #0e1117; }

        .title-block {
            text-align: center;
            padding: 2rem 0 1rem 0;
        }
        .title-block h1 {
            font-size: 2.8rem;
            font-weight: 800;
            color: #ffffff;
        }
        .title-block p {
            color: #aaaaaa;
            font-size: 1.05rem;
        }

        .result-fake {
            background: linear-gradient(135deg, #ff4b4b22, #ff4b4b44);
            border: 1.5px solid #ff4b4b;
            border-radius: 12px;
            padding: 1.5rem;
            text-align: center;
            font-size: 1.5rem;
            font-weight: 700;
            color: #ff4b4b;
            margin: 1rem 0;
        }
        .result-real {
            background: linear-gradient(135deg, #00c85322, #00c85344);
            border: 1.5px solid #00c853;
            border-radius: 12px;
            padding: 1.5rem;
            text-align: center;
            font-size: 1.5rem;
            font-weight: 700;
            color: #00c853;
            margin: 1rem 0;
        }

        .metric-card {
            background: #1e2130;
            border-radius: 10px;
            padding: 1rem 1.5rem;
            text-align: center;
        }
        .metric-card .label {
            font-size: 0.8rem;
            color: #aaaaaa;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .metric-card .value {
            font-size: 1.8rem;
            font-weight: 700;
            color: #ffffff;
        }

        div[data-testid="stTabs"] button {
            font-size: 1rem;
            font-weight: 600;
        }
    </style>
""", unsafe_allow_html=True)

# ==========================
# Header
# ==========================

st.markdown("""
    <div class="title-block">
        <h1>📰 Fake News Detection System</h1>
        <p>Paste an article, enter a URL, or type text — get an instant AI verdict.</p>
    </div>
""", unsafe_allow_html=True)

# ==========================
# Session State
# ==========================

if "history" not in st.session_state:
    st.session_state.history = []

# ==========================
# Input Tabs
# ==========================

tab1, tab2 = st.tabs(["✍️ Paste Text", "🔗 Enter URL"])

news_text = ""

with tab1:
    news_text_input = st.text_area(
        "Paste your news article here:",
        height=220,
        placeholder="Enter the full text of the news article..."
    )
    if news_text_input.strip():
        news_text = news_text_input

with tab2:
    url_input = st.text_input(
        "Enter a news article URL:",
        placeholder="https://example.com/news-article"
    )
    if st.button("🔍 Fetch Article"):
        if url_input.strip():
            try:
                with st.spinner("Fetching article..."):
                    headers = {"User-Agent": "Mozilla/5.0"}
                    resp = requests.get(url_input.strip(), headers=headers, timeout=10)
                    soup = BeautifulSoup(resp.text, "html.parser")
                    paragraphs = soup.find_all("p")
                    fetched = " ".join(p.get_text() for p in paragraphs)
                    if fetched.strip():
                        st.session_state["fetched_text"] = fetched
                        st.success("Article fetched successfully!")
                    else:
                        st.warning("Could not extract text from this URL.")
            except Exception as e:
                st.error(f"Failed to fetch URL: {e}")

    if "fetched_text" in st.session_state:
        st.text_area(
            "Fetched Article:",
            value=st.session_state["fetched_text"],
            height=200
        )
        news_text = st.session_state["fetched_text"]

# ==========================
# Predict Button
# ==========================

st.markdown("<br>", unsafe_allow_html=True)
predict_clicked = st.button("🔮 Analyze Article", use_container_width=True)

if predict_clicked:
    if not news_text.strip():
        st.warning("Please enter or fetch some news text first.")
    else:
        with st.spinner("Analyzing..."):

            news_vector = vectorizer.transform([news_text])
            prediction = model.predict(news_vector)[0]
            proba = model.predict_proba(news_vector)[0]
            confidence = max(proba) * 100
            fake_prob = proba[0] * 100
            real_prob = proba[1] * 100

            label = "FAKE" if prediction == 0 else "REAL"
            emoji = "❌" if prediction == 0 else "✅"

            # Result banner
            if prediction == 0:
                st.markdown(
                    f'<div class="result-fake">{emoji} FAKE NEWS &nbsp;|&nbsp; Confidence: {confidence:.1f}%</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="result-real">{emoji} REAL NEWS &nbsp;|&nbsp; Confidence: {confidence:.1f}%</div>',
                    unsafe_allow_html=True
                )

            # Metric cards
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(f"""
                    <div class="metric-card">
                        <div class="label">Verdict</div>
                        <div class="value">{emoji} {label}</div>
                    </div>
                """, unsafe_allow_html=True)
            with c2:
                st.markdown(f"""
                    <div class="metric-card">
                        <div class="label">Confidence</div>
                        <div class="value">{confidence:.1f}%</div>
                    </div>
                """, unsafe_allow_html=True)
            with c3:
                st.markdown(f"""
                    <div class="metric-card">
                        <div class="label">Words Analyzed</div>
                        <div class="value">{len(news_text.split()):,}</div>
                    </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Confidence chart
            st.subheader("📊 Confidence Breakdown")

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=["Real News", "Fake News"],
                y=[real_prob, fake_prob],
                marker_color=["#00c853", "#ff4b4b"],
                text=[f"{real_prob:.1f}%", f"{fake_prob:.1f}%"],
                textposition="outside",
                width=0.4
            ))
            fig.update_layout(
                height=320,
                yaxis=dict(range=[0, 115], showgrid=False, title="Probability (%)"),
                xaxis=dict(showgrid=False),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white", size=14),
                margin=dict(t=20, b=20)
            )
            st.plotly_chart(fig, use_container_width=True)

            # Save to history
            st.session_state.history.append({
                "Time": datetime.now().strftime("%H:%M:%S"),
                "Verdict": label,
                "Confidence": f"{confidence:.1f}%",
                "Fake Prob": f"{fake_prob:.1f}%",
                "Real Prob": f"{real_prob:.1f}%",
                "Words": len(news_text.split()),
                "Preview": news_text[:80] + "..."
            })

# ==========================
# Prediction History
# ==========================

if st.session_state.history:
    st.divider()
    st.subheader("🕓 Prediction History")

    history_df = pd.DataFrame(st.session_state.history)

    # Color verdict column
    def color_verdict(val):
        color = "#ff4b4b" if val == "FAKE" else "#00c853"
        return f"color: {color}; font-weight: bold"

    st.dataframe(
        history_df.style.applymap(color_verdict, subset=["Verdict"]),
        use_container_width=True,
        hide_index=True
    )

    if st.button("🗑️ Clear History"):
        st.session_state.history = []
        st.rerun()

# ==========================
# Sidebar
# ==========================

with st.sidebar:
    st.markdown("## 📰 About")
    st.info("""
    **Fake News Detection System**

    Analyzes news articles using Machine Learning
    and gives a real-time verdict.

    **Technologies:**
    - Python
    - Scikit-Learn
    - TF-IDF Vectorizer
    - Logistic Regression
    - Streamlit
    - Plotly
    """)

    st.markdown("---")
    st.markdown("## 📌 Tips")
    st.markdown("""
    - Paste **full articles** for best accuracy
    - Short snippets may reduce confidence
    - URL fetch works on most news sites
    - Check the confidence % — below 60% means uncertain
    """)

    st.markdown("---")
    st.caption("⚠️ Educational Project Only. Not for real-world use.")