import os, json, re
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
import joblib
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from bs4 import beautifulsoup
import requests

try:
    import mysql.connector
    from mysql.connector import Error
except Exception:
    mysql = None

try:
    from google import genai
except Exception:
    genai = None

load_dotenv()
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "models" / "fake_news_model.joblib"
DATA_PATH = BASE_DIR / "data" / "sample_news.csv"

st.set_page_config(page_title="VeriNews AI | Fake News Detector", page_icon="🛰️",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
html, body, [class*="css"] {font-family:'DM Sans',sans-serif;}
.stApp {background:radial-gradient(circle at 12% 0%, #182642 0%, #0a1020 38%, #080d18 100%); color:#e8eefc;}
h1,h2,h3 {font-family:'Space Grotesk',sans-serif; color:#f4f7ff;}
[data-testid="stSidebar"] {background:#0c1424; border-right:1px solid #24324a;}
[data-testid="stMetric"] {background:linear-gradient(145deg,#131f34,#10192a); padding:16px; border:1px solid #263954; border-radius:14px;}
div.stButton > button {background:linear-gradient(90deg,#3478f6,#795cf6); color:white; border:0; border-radius:10px; font-weight:700; padding:.65rem 1rem;}
div.stButton > button:hover {border:1px solid #83aaff; color:white;}
textarea, input {background:#111c30 !important; color:#eef3ff !important;}
.panel {background:linear-gradient(145deg,rgba(21,34,56,.96),rgba(13,22,39,.96)); border:1px solid #293d5b; border-radius:16px; padding:18px 20px; margin:8px 0;}
.muted {color:#9aabc7; font-size:.9rem;}
.pill {display:inline-block;padding:5px 11px;border-radius:20px;background:#1d3151;color:#a9c8ff;font-size:.8rem;font-weight:700;}
.smallcaps {color:#8fa6ca;letter-spacing:1.5px;font-size:.72rem;font-weight:700;}
hr {border-color:#263751;}
</style>
""", unsafe_allow_html=True)

def get_secret(name, default=""):
    try:
        return st.secrets.get(name, os.getenv(name, default))
    except Exception:
        return os.getenv(name, default)

def load_model():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH), False
    # The included tiny sample set is only a smoke-test/demo dataset, not a research-grade model.
    df = pd.read_csv(DATA_PATH)
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1,2), max_features=12000)
    X = vectorizer.fit_transform(df["text"].astype(str))
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X, df["label"].astype(str))
    def fetch_article_from_url(url: str):
        headers = {"User-Agent": "Mozilla/5.0"}
        res = requests.get(url, headers=headers, timeout=10)
        res.raise_for_status()
        soup = BeautifulSoup(res.text, "html.parser")
        title = soup.title.get_text(strip=True) if soup.title else "Untitled"
        paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p") if len(p.get_text(strip=True)) > 40]
        text = f"{title}\n\n" + "\n".join(paragraphs)
        return text.strip(), title, None
        return None, None, str(e)
    res = requests.get(url, headers=headers, timeout=10)
    res.raise_for_status()
    soup = BeautifulSoup(res.text, "html.parser")

    title = soup.title.get_text(strip=True) if soup.title else "Untitled"
    paragraphs = [
        p.get_text(" ", strip=True)
        for p in soup.find_all("p")
        if len(p.get_text(strip=True)) > 40
    ]
    article_text = f"{title}\n\n" + "\n".join(paragraphs)
    return article_text.strip(), title, None
    return None, None, str(e)
    return {"vectorizer": vectorizer, "classifier": clf}, True
@st.cache_resource
def cached_model():
    return load_model()

def ml_predict(text):
    bundle, is_demo = cached_model()
    vec = bundle["vectorizer"].transform([text])
    clf = bundle["classifier"]
    pred = str(clf.predict(vec)[0]).lower()
    probs = clf.predict_proba(vec)[0]
    prob_map = {str(label).lower(): float(p) for label, p in zip(clf.classes_, probs)}
    return pred, prob_map, is_demo

def gemini_analyze(text):
    api_key = get_secret("GEMINI_API_KEY")
    if not api_key or genai is None:
        return None, "Gemini is not configured. Add GEMINI_API_KEY to .env or Streamlit secrets."
    try:
        client = genai.Client(api_key=api_key)
        prompt = f"""
You are an assistant for preliminary news-literacy analysis, not an authoritative fact-checker.
Analyze the user-provided news text. Do not assume it is false merely because it sounds surprising.
You do not have live web access; do not invent sources, citations, dates, or verification.
Return ONLY valid JSON with these keys:
label: one of "real", "fake", "misleading", "unverified"
reason: 2-4 concise sentences explaining signals and uncertainty
red_flags: array of up to 4 specific textual warning signs (empty if none)
verification_steps: array of 2-3 practical checks the reader should perform
evidence_limit: one sentence stating that claims require independent source verification
Text:
{text[:9000]}
"""
        response = client.models.generate_content(
            model=get_secret("GEMINI_MODEL", "gemini-2.5-flash"),
            contents=prompt
        )
        raw = response.text or ""
        match = re.search(r"\{.*\}", raw, flags=re.S)
        if not match:
            raise ValueError("The model did not return valid JSON.")
        data = json.loads(match.group(0))
        allowed = {"real", "fake", "misleading", "unverified"}
        if str(data.get("label", "")).lower() not in allowed:
            data["label"] = "unverified"
        data["label"] = str(data["label"]).lower()
        data["red_flags"] = data.get("red_flags", [])[:4]
        data["verification_steps"] = data.get("verification_steps", [])[:3]
        return data, None
    except Exception as exc:
        return None, f"Gemini analysis unavailable: {exc}"

def mysql_connection():
    cfg = {
        "host": get_secret("MYSQL_HOST"),
        "user": get_secret("MYSQL_USER"),
        "password": get_secret("MYSQL_PASSWORD"),
        "database": get_secret("MYSQL_DATABASE"),
        "port": int(get_secret("MYSQL_PORT", "3306")),
        "connection_timeout": 5,
    }
    if not all([cfg["host"], cfg["user"], cfg["database"]]):
        return None
    try:
        return mysql.connector.connect(**cfg)
    except Exception:
        return None

def save_record(article, ml_label, ml_probs, llm, final_label):
    conn = mysql_connection()
    if conn is None:
        return False
    try:
        cur = conn.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS analyses (
          id BIGINT AUTO_INCREMENT PRIMARY KEY,
          article_text MEDIUMTEXT NOT NULL,
          ml_label VARCHAR(30) NOT NULL,
          ml_probabilities JSON,
          llm_label VARCHAR(30),
          llm_reason TEXT,
          final_label VARCHAR(40) NOT NULL,
          created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        cur.execute("""INSERT INTO analyses
        (article_text, ml_label, ml_probabilities, llm_label, llm_reason, final_label)
        VALUES (%s,%s,%s,%s,%s,%s)""",
        (article, ml_label, json.dumps(ml_probs),
         llm.get("label") if llm else None,
         llm.get("reason") if llm else None, final_label))
        conn.commit()
        cur.close()
        conn.close()
        return True
    except Exception:
        try: conn.close()
        except Exception: pass
        return False

def read_history(limit=100):
    conn = mysql_connection()
    if conn is None:
        return pd.DataFrame()
    try:
        cur = conn.cursor(dictionary=True)
        cur.execute("SELECT id, ml_label, llm_label, final_label, created_at, LEFT(article_text,180) AS preview FROM analyses ORDER BY id DESC LIMIT %s", (limit,))
        rows = cur.fetchall()
        cur.close(); conn.close()
        return pd.DataFrame(rows)
    except Exception:
        try: conn.close()
        except Exception: pass
        return pd.DataFrame()

def final_decision(ml_label, llm_data):
    if not llm_data:
        return ml_label, "ML-only result: Gemini was unavailable."
    llm_label = llm_data["label"]
    if llm_label == ml_label:
        return ml_label, "ML and LLM assessments agree. Agreement is not proof of truth."
    if llm_label == "unverified":
        return "Needs verification", "The LLM could not confidently assess the claim; check independent evidence."
    return "Conflicting signals", "The ML and LLM assessments disagree. Do not treat either output as a verified fact."

# Sidebar
with st.sidebar:
    st.markdown("## 🛰️ VeriNews AI")
    st.markdown('<span class="pill">HYBRID INTELLIGENCE</span>', unsafe_allow_html=True)
    page = st.radio("NAVIGATION", ["Overview", "Analyze News", "Analysis History", "About Project"], label_visibility="visible")
    st.divider()
    st.markdown("**SYSTEM STATUS**")
    st.caption("ML engine: available")
    st.caption("Gemini: " + ("configured" if get_secret("GEMINI_API_KEY") else "API key not set"))
    st.caption("MySQL: " + ("configured" if get_secret("MYSQL_HOST") else "not configured"))
    st.markdown("---")
    st.caption("BCA Final-Year Project • Academic prototype")

if page == "Overview":
    st.markdown('<div class="smallcaps">NEWS INTEGRITY INTELLIGENCE / 01</div>', unsafe_allow_html=True)
    st.title("See beyond the headline.")
    st.markdown('<p class="muted">A hybrid machine-learning and large-language-model workspace for preliminary news credibility analysis.</p>', unsafe_allow_html=True)
    c1,c2,c3 = st.columns(3)
    c1.metric("Detection engines", "02", "TF-IDF + Gemini")
    c2.metric("Storage", "MySQL", "Optional cloud DB")
    c3.metric("Mode", "Hybrid", "Agreement-aware")
    st.markdown('<div class="panel"><span class="smallcaps">HOW IT WORKS</span><h3>Two perspectives. One cautious assessment.</h3><p class="muted">The ML classifier learns language patterns from labeled training examples. Gemini reviews the article text and explains possible warning signs. The app compares both outputs and highlights disagreement rather than pretending to know the truth.</p></div>', unsafe_allow_html=True)
    left,right = st.columns([1.15, .85])
    with left:
        st.markdown("### Intelligence pipeline")
        st.markdown("""
        1. **Input** — paste a headline or article.
        2. **ML classification** — TF-IDF features + Logistic Regression.
        3. **LLM review** — Gemini provides a reasoned assessment and verification checklist.
        4. **Hybrid interpretation** — agreement, conflict, or needs verification.
        5. **Persistence** — store results in MySQL when configured.
        """)
    with right:
        st.markdown('<div class="panel"><span class="smallcaps">IMPORTANT LIMITATION</span><h4>Not a fact-checking authority</h4><p class="muted">Text-only models cannot independently verify current events. A confident prediction can still be wrong. Always check primary sources and trusted independent reporting.</p></div>', unsafe_allow_html=True)
    if cached_model()[1]:
        st.warning("Demo model active: the bundled sample dataset is tiny and synthetic. Train with a real, properly labeled dataset before evaluating or presenting accuracy.")

elif page == "Analyze News":
    st.markdown('<div class="smallcaps">CREDIBILITY WORKSPACE / 02</div>', unsafe_allow_html=True)
    st.title("Analyze a news report")
    st.markdown('<p class="muted">Paste the article text. Avoid including personal or confidential information.</p>', unsafe_allow_html=True)
    
    tab_text, tab_url = st.tabs(["📝 Paste News Text", "🔗 Fetch from News URL"])

with tab_url:
  news_url = st.text_input(
      "Enter any news article URL:",
      placeholder="https://example.com/news-story",
  )
  if st.button("Extract Article from URL"):
    if news_url.strip():
      with st.spinner("Fetching article content..."):
        fetched_text, fetched_title, err = fetch_article_from_url(
            news_url.strip()
        )
        if fetched_text and len(fetched_text) > 30:
          st.session_state["fetched_article"] = fetched_text
          st.success(f"Extracted: **{fetched_title}**")
        else:
          st.error(f"Could not extract text: {err or 'Blocked or empty page'}")
    else:
      st.warning("Please paste a valid URL first.")

with tab_text:
  article = st.text_area(
      "News headline or full article text:",
      value=st.session_state.get("fetched_article", ""),
      height=180,
      placeholder="Paste news text here, or extract it from the URL tab...",
  )
if st.button("⚡ Run hybrid analysis", use_container_width=True):
        if len(article.strip()) < 30:
            st.error("Please enter at least 30 characters for a meaningful analysis.")
        else:
            with st.spinner("Running ML classifier and requesting LLM review..."):
                ml_label, ml_probs, demo_mode = ml_predict(article)
                llm_data, llm_error = gemini_analyze(article)
                final_label, decision_note = final_decision(ml_label, llm_data)
                saved = save_record(article, ml_label, ml_probs, llm_data, final_label)
             st.session_state["latest"] = dict(ml_label=ml_label, ml_probs=ml_probs, llm=llm_data, llm_error=llm_error, final=final_label, note=decision_note, demo=demo_mode,saved=saved )
               result = st.session_state.get("latest")
if result:
        st.divider()
        st.markdown("### Analysis result")
        if result["demo"]:
            st.warning("Demo model: trained automatically on the small included sample dataset. Its probabilities are not meaningful evidence. Train on a larger real dataset.")
        st.markdown(f'<div class="panel"><span class="smallcaps">HYBRID ASSESSMENT</span><h2>{result["final"]}</h2><p class="muted">{result["note"]}</p></div>', unsafe_allow_html=True)
        a,b = st.columns(2)
        with a:
            st.markdown("**Machine-learning model**")
            st.metric("ML label", result["ml_label"].upper())
            st.bar_chart(pd.DataFrame({"Probability": result["ml_probs"]}))
        with b:
            st.markdown("**LLM review (Gemini)**")
            if result["llm"]:
                st.metric("LLM assessment", result["llm"]["label"].upper())
                st.write(result["llm"].get("reason", "No explanation returned."))
                if result["llm"].get("red_flags"):
                    st.markdown("**Potential warning signs**")
                    for item in result["llm"]["red_flags"]:
                        st.markdown(f"- {item}")
                if result["llm"].get("verification_steps"):
                    st.markdown("**Suggested verification steps**")
                    for item in result["llm"]["verification_steps"]:
                        st.markdown(f"- {item}")
                st.caption(result["llm"].get("evidence_limit", "Independent verification is required."))
            else:
                st.warning(result["llm_error"] or "LLM did not return a result.")
        st.caption("ML probabilities are model scores, not calibrated probabilities of factual truth.")
        st.caption("MySQL save: " + ("successful" if result["saved"] else "not saved (configure MySQL or check connection)."))

elif page == "Analysis History":
    st.markdown('<div class="smallcaps">AUDIT LOG / 03</div>', unsafe_allow_html=True)
    st.title("Analysis history")
    hist = read_history()
    if hist.empty:
        st.info("No MySQL records found. Configure MySQL and run an analysis to create history.")
    else:
        st.dataframe(hist, use_container_width=True, hide_index=True)
        st.download_button("Download history as CSV", hist.to_csv(index=False), "verinews_history.csv", "text/csv")

else:
    st.markdown('<div class="smallcaps">PROJECT DOCUMENTATION / 04</div>', unsafe_allow_html=True)
    st.title("About VeriNews AI")
    st.markdown("""
    **Project title:** Hybrid Fake News Detection Using Machine Learning and Large Language Models

    **Problem statement:** Online misinformation spreads quickly. This academic prototype explores how a traditional text classifier and an LLM can provide complementary preliminary signals to support news literacy.

    **Algorithms and methods**
    - TF-IDF (Term Frequency–Inverse Document Frequency) for converting text into numerical features.
    - Logistic Regression for supervised text classification.
    - Gemini LLM for natural-language review, explanation, warning signs, and verification suggestions.
    - Agreement-aware decision logic to flag disagreement between the two systems.

    **Database:** MySQL `analyses` table stores article text, model labels, ML scores, LLM explanation, final interpretation, and timestamp.

    **Evaluation recommendation:** Use a separate test set. Report accuracy, precision, recall, F1-score, confusion matrix, and class balance. Do not report the tiny demo data's score as project performance.

    **Ethical note:** Do not label a person, publisher, or article as deliberately deceptive based only on this prototype. It does not browse the web or verify claims against primary evidence.
    """)
    st.code("python train_model.py --data data/your_dataset.csv", language="bash")
