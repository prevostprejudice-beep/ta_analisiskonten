import os
import io
import csv
import re
from textwrap import shorten

import streamlit as st
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import IsolationForest
from sklearn.metrics.pairwise import cosine_similarity

import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering

# ============================================================
# OPTIONAL: LLM / HF model imports (aman kalau tidak terpasang)
# ============================================================
try:
    import torch
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        AutoConfig,
    )

    HAS_TRANSFORMERS = True
except Exception:
    HAS_TRANSFORMERS = False
    torch = None
    AutoTokenizer = AutoModelForSequenceClassification = AutoConfig = None

try:
    from sentence_transformers import SentenceTransformer

    HAS_SENTENCE_TRANSFORMERS = True
except Exception:
    HAS_SENTENCE_TRANSFORMERS = False
    SentenceTransformer = None

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Sistem Analisis Konten Radikal",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# KONFIGURASI MODEL HUGGINGFACE
# ============================================================
INDOBERT_MODEL_NAME = "w11wo/indonesian-roberta-base-sentiment-classifier"
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
LOGO_PATH = "bin.png"
PLOTLY_TEMPLATE = "plotly_dark"

# ============================================================
# STYLING
# ============================================================
CUSTOM_CSS = """
<style>
    :root {
        --bg-main: #07111f;
        --bg-card: rgba(15, 23, 42, 0.88);
        --bg-card-soft: rgba(30, 41, 59, 0.65);
        --stroke: rgba(148, 163, 184, 0.20);
        --text-main: #f8fafc;
        --text-soft: #cbd5e1;
        --danger: #ef4444;
        --warning: #f59e0b;
        --info: #38bdf8;
        --success: #22c55e;
    }

    .stApp {
        background:
            radial-gradient(circle at top left, rgba(127,29,29,0.23), transparent 28%),
            radial-gradient(circle at top right, rgba(14,165,233,0.14), transparent 24%),
            linear-gradient(180deg, #030712 0%, #07111f 35%, #0b1220 100%);
        color: var(--text-main);
    }

    .block-container {
        padding-top: 4.4rem;
        padding-bottom: 2rem;
        max-width: 1600px;
    }

    @media (max-width: 900px) {
        .block-container {
            padding-top: 3.6rem;
        }
    }

    .hero-wrap {
        background: linear-gradient(135deg, rgba(127,29,29,0.90), rgba(15,23,42,0.96) 55%, rgba(8,47,73,0.92));
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 24px;
        padding: 24px 28px;
        box-shadow: 0 18px 50px rgba(0, 0, 0, 0.32);
        margin-top: 0.25rem;
        margin-bottom: 1.1rem;
    }

    .hero-title {
        font-size: 2.4rem;
        font-weight: 800;
        line-height: 1.08;
        color: white;
        margin-bottom: 0.3rem;
    }

    .hero-subtitle {
        font-size: 1rem;
        color: #e2e8f0;
        line-height: 1.55;
        max-width: 920px;
    }

    .section-card {
        background: var(--bg-card);
        border: 1px solid var(--stroke);
        border-radius: 20px;
        padding: 18px 20px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.18);
        margin-bottom: 1rem;
    }

    .section-title {
        font-size: 1.08rem;
        font-weight: 700;
        color: var(--text-main);
        margin-bottom: 0.35rem;
    }

    .section-subtitle {
        font-size: 0.92rem;
        color: var(--text-soft);
        line-height: 1.55;
    }

    .kpi-card {
        background: linear-gradient(180deg, rgba(15,23,42,0.95), rgba(15,23,42,0.76));
        border: 1px solid var(--stroke);
        border-radius: 20px;
        padding: 16px 18px;
        min-height: 122px;
        box-shadow: 0 8px 20px rgba(0,0,0,0.16);
    }

    .kpi-label {
        color: var(--text-soft);
        font-size: 0.84rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 0.45rem;
    }

    .kpi-value {
        color: white;
        font-size: 2rem;
        font-weight: 800;
        line-height: 1.0;
        margin-bottom: 0.45rem;
    }

    .kpi-desc {
        color: #cbd5e1;
        font-size: 0.86rem;
        line-height: 1.45;
    }

    .status-pill {
        display: inline-block;
        border-radius: 999px;
        padding: 0.38rem 0.8rem;
        font-size: 0.82rem;
        font-weight: 700;
        margin-right: 0.4rem;
        margin-top: 0.35rem;
        border: 1px solid transparent;
    }

    .status-ok {
        background: rgba(34,197,94,0.16);
        color: #86efac;
        border-color: rgba(34,197,94,0.25);
    }

    .status-warn {
        background: rgba(245,158,11,0.16);
        color: #fcd34d;
        border-color: rgba(245,158,11,0.25);
    }

    .status-danger {
        background: rgba(239,68,68,0.16);
        color: #fca5a5;
        border-color: rgba(239,68,68,0.25);
    }

    .risk-high, .risk-medium, .risk-low {
        display: inline-block;
        border-radius: 999px;
        padding: 0.24rem 0.7rem;
        font-size: 0.76rem;
        font-weight: 700;
        border: 1px solid transparent;
    }

    .risk-high {
        background: rgba(239,68,68,0.16);
        color: #fca5a5;
        border-color: rgba(239,68,68,0.25);
    }

    .risk-medium {
        background: rgba(245,158,11,0.16);
        color: #fde68a;
        border-color: rgba(245,158,11,0.25);
    }

    .risk-low {
        background: rgba(34,197,94,0.16);
        color: #86efac;
        border-color: rgba(34,197,94,0.25);
    }

    .mini-note {
        color: var(--text-soft);
        font-size: 0.85rem;
        line-height: 1.55;
    }

    .priority-card {
        background: linear-gradient(180deg, rgba(15,23,42,0.95), rgba(15,23,42,0.72));
        border-left: 4px solid #ef4444;
        border-radius: 18px;
        padding: 16px 18px;
        border-top: 1px solid var(--stroke);
        border-right: 1px solid var(--stroke);
        border-bottom: 1px solid var(--stroke);
        margin-bottom: 0.9rem;
        box-shadow: 0 8px 22px rgba(0,0,0,0.15);
    }

    .priority-title {
        color: white;
        font-size: 1rem;
        font-weight: 800;
        margin-bottom: 0.35rem;
    }

    .priority-meta {
        color: #cbd5e1;
        font-size: 0.87rem;
        line-height: 1.55;
        margin-bottom: 0.45rem;
    }

    .priority-text {
        color: #e2e8f0;
        font-size: 0.9rem;
        line-height: 1.6;
    }

    .legend-dot {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 8px;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(15,23,42,0.96), rgba(30,41,59,0.96));
        border-right: 1px solid rgba(148, 163, 184, 0.14);
    }

    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] div {
        color: #f8fafc;
    }

    div[data-testid="metric-container"] {
        background: linear-gradient(180deg, rgba(15,23,42,0.95), rgba(15,23,42,0.72));
        border: 1px solid var(--stroke);
        padding: 1rem 1rem;
        border-radius: 18px;
        box-shadow: 0 8px 20px rgba(0,0,0,0.15);
    }

    .stDataFrame, .stTable {
        border-radius: 18px;
        overflow: hidden;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ============================================================
# HELPERS UI
# ============================================================
def metric_card(label: str, value, description: str = ""):
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-desc">{description}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_intro(title: str, subtitle: str):
    st.markdown(
        f"""
        <div class="section-card">
            <div class="section-title">{title}</div>
            <div class="section-subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def risk_badge(score: float) -> str:
    if score >= 0.85:
        return '<span class="risk-high">Risiko Tinggi</span>'
    if score >= 0.70:
        return '<span class="risk-medium">Perlu Ditinjau</span>'
    return '<span class="risk-low">Risiko Rendah</span>'


def render_priority_cards(data: pd.DataFrame, limit: int = 5):
    if data.empty:
        st.info("Belum ada data prioritas untuk ditampilkan.")
        return

    for _, row in data.head(limit).iterrows():
        text_preview = shorten(str(row.get("isi_postingan", "")), width=220, placeholder="...")
        badge = risk_badge(float(row.get("skor_radikal_final", 0)))
        anomaly = row.get("anomali_flag", "-")
        engagement = int(row.get("total_engagement", 0))
        sentiment = row.get("llm_sentiment_label", "-")
        st.markdown(
            f"""
            <div class="priority-card">
                <div class="priority-title">{row.get('nama', '-')} <span style="font-size:0.82rem; color:#94a3b8; font-weight:600;">({row.get('user_id','-')})</span></div>
                <div class="priority-meta">
                    {badge}&nbsp;&nbsp;
                    <span class="status-pill {'status-danger' if anomaly == 'Anomali' else 'status-ok'}">{anomaly}</span>
                    <br>
                    Sentimen LLM: <b>{sentiment}</b> &nbsp;|&nbsp;
                    Skor radikal final: <b>{float(row.get('skor_radikal_final', 0)):.2f}</b> &nbsp;|&nbsp;
                    Engagement: <b>{engagement:,}</b>
                </div>
                <div class="priority-text">{text_preview}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def apply_plotly_theme(fig: go.Figure, title: str | None = None):
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.50)",
        font=dict(color="#e2e8f0"),
        title=title if title is not None else fig.layout.title.text,
        legend_title_text="",
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


def show_logo_if_exists(width: int = 110):
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, width=width)


# ============================================================
# LEXICON / HEURISTIK MODE KETAT
# ============================================================
lexicon_radikal = [
    "thaghut", "taghut", "t4ghut", "th4ghut", "toghut", "tahgut",
    "kafir", "k@fir", "kaf1r", "murtad", "munafik", "munafiq",
    "zalim", "dzalim", "dhalim", "dholim", "jahiliyah", "jahiliah",
    "salibis", "crusader", "darul harb", "darul kufr", "wilayah perang",
    "jihad", "j1had", "j!had", "jihadist", "amaliyah", "amaliah",
    "amaliyat", "istisyhad", "istisyhadi", "istisyhadiyah", "bom syahid",
    "bom bunuh diri", "ghazwah", "ghazwat", "ghozoah", "qital",
    "perang badar", "serang", "hancurkan", "habisi", "lone wolf",
    "serigala tunggal", "daulah", "dawlah", "d4ulah", "khilafah",
    "khil4fah", "khilafiyah", "imarah", "amir", "ameer", "khalifah",
    "waliyul amri", "anshar", "ansar", "anshor", "ansharut", "muhajir",
    "muhajirin", "hijrah", "hijroh", "h1jrah", "baiat", "bayat", "b@iat",
    "b4iat", "istiqamah perjuangan", "istiqomah perjuangan", "ukhti fillah",
    "akh fi sabilillah", "tarhib", "targhib", "bergabung di jalan ini",
    "tinggalkan negeri kafir", "nashir", "nasyir", "nasher", "bayan",
    "qashidah", "anasyid jihad", "ghuraba", "ghuroba", "tauhid murni",
    "tawheed", "dhulm", "dzulm", "#zalim", "#khilafah", "#dawlah",
    "millah ibrahim", "milah ibrahim", "millahibrahim", "taghutoff",
    "#hancurkanthaghut", "syuhada", "syahid", "syahidah", "#lonewolf",
    "aktor tunggal", "idad", "syariat kaffah", "amar ma'ruf nahi munkar",
    "amar makruf nahi munkar",
]

kata_kunci_negatif = [
    "serang", "hancurkan", "musuh", "radikal", "bom", "senjata", "perang", "habisi"
]
kata_kunci_positif = [
    "damai", "toleransi", "bersatu", "harmoni", "cinta", "kebaikan", "keadilan", "kemanusiaan"
]

kata_kerja_ajakan_radikal = [
    "hancurkan", "hancurkah", "hancur kan", "serang", "serbu", "habisi", "tumpas",
    "perangi", "perangilah", "angkat senjata", "siapkan senjata", "lawan dengan senjata",
    "bom", "ledakkan", "tusuk", "tembak", "tegakkan khilafah", "tegakkan daulah",
    "tegakkan syariat kaffah", "baiat", "berbaiat", "berbai'ah", "mari baiat",
    "bergabung dengan", "bergabung di jalan ini", "ikut jihad", "berjihad", "ayo jihad",
    "wajib jihad", "dukung khilafah", "dukung daulah", "setia kepada khilafah",
    "syahid di medan jihad", "siap mati syahid",
]

frasa_penguatan_radikal = [
    "kewajiban setiap muslim sejati", "kewajiban kita", "jalan kebenaran", "satu-satunya jalan",
    "tidak ada pilihan lain selain", "ini perintah allah", "ini perintah tuhan", "demi agama",
    "demi iman", "jihad adalah solusi", "khilafah adalah solusi", "syariat kaffah satu-satunya jalan",
]

konteks_anti_radikal = [
    "bahaya radikalisme", "bahaya propaganda", "bahaya paham radikal", "bahaya ekstremisme",
    "bahaya terorisme", "bahaya lone wolf", "melawan radikalisme", "melawan ekstremisme",
    "melawan terorisme", "menolak radikalisme", "tolak radikalisme", "tolak kekerasan",
    "mencegah radikalisme", "pencegahan radikalisme", "jangan terpengaruh",
    "jangan terprovokasi", "perlu kewaspadaan terhadap", "kewaspadaan terhadap",
    "waspada terhadap", "antisipasi penyebaran", "sebagian orang menyalahgunakan istilah jihad",
    "pendistorsian makna jihad", "penyimpangan makna jihad", "bukan ajaran islam",
    "tidak sesuai ajaran islam", "kontra radikalisme", "narasi damai", "kontra narasi",
    "edukasi tentang bahaya", "sosialisasi bahaya", "menjelaskan bahaya",
    "mengkritik paham radikal",
]

# ============================================================
# STOPWORDS
# ============================================================
stopwords_id_manual = {
    "yang", "dan", "di", "ke", "dari", "dalam", "pada", "dengan", "karena",
    "sehingga", "agar", "untuk", "adalah", "ialah", "bahwa", "ini", "itu",
    "sudah", "telah", "akan", "atau", "juga", "saja", "lagi", "sebagai", "serta",
    "kami", "kita", "saya", "aku", "anda", "kamu", "engkau", "dia", "ia",
    "mereka", "para", "pun", "lah", "punya", "the", "is", "are", "of", "on",
    "in", "at", "to", "for", "an", "a", "and", "or",
}

wordcloud_default_sw = set(STOPWORDS)
STOPWORDS_WORDCLOUD = wordcloud_default_sw.union(stopwords_id_manual)
STOPWORDS_TFIDF = list(stopwords_id_manual)


# ============================================================
# LOADER MODEL HUGGINGFACE (CACHED)
# ============================================================
@st.cache_resource(show_spinner=False)
def load_indobert_model():
    if not HAS_TRANSFORMERS or torch is None:
        return None, None, None
    try:
        config = AutoConfig.from_pretrained(INDOBERT_MODEL_NAME)
        tokenizer = AutoTokenizer.from_pretrained(INDOBERT_MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(INDOBERT_MODEL_NAME)
        model.eval()
        id2label = (
            config.id2label if hasattr(config, "id2label") else {i: str(i) for i in range(model.num_labels)}
        )
        return tokenizer, model, id2label
    except Exception as e:
        st.sidebar.warning(
            f"Gagal load IndoBERT model '{INDOBERT_MODEL_NAME}': {e}\n"
            "Model LLM akan dimatikan (fallback ke rule-based saja)."
        )
        return None, None, None


@st.cache_resource(show_spinner=False)
def load_embedding_model():
    if not HAS_SENTENCE_TRANSFORMERS:
        return None
    try:
        return SentenceTransformer(EMBEDDING_MODEL_NAME)
    except Exception as e:
        st.sidebar.warning(
            f"Gagal load embedding model '{EMBEDDING_MODEL_NAME}': {e}\n"
            "SNA akan fallback ke TF-IDF."
        )
        return None


# ============================================================
# UTILITAS DATA
# ============================================================
def load_csv_safe(uploaded_file_obj):
    raw_bytes = uploaded_file_obj.read()
    if not raw_bytes:
        st.error("File kosong (0 bytes).")
        return None

    raw_text = None
    for enc in ("utf-8-sig", "utf-8", "latin1"):
        try:
            raw_text = raw_bytes.decode(enc)
            break
        except Exception:
            continue
    if raw_text is None:
        st.error("Gagal decode file.")
        return None

    try:
        dialect = csv.Sniffer().sniff(raw_text[:16384], delimiters=[",", ";", "\t", "|"])
        delimiter = dialect.delimiter
    except Exception:
        delimiter = ","

    buf = io.StringIO(raw_text)
    try:
        df_local = pd.read_csv(buf, delimiter=delimiter)
    except Exception as e:
        st.error(f"Gagal membaca CSV: {e}")
        return None

    df_local.columns = df_local.columns.astype(str).str.strip().str.lower()

    col_map = {
        "fb name": "nama",
        "fb_name": "nama",
        "uid": "user_id",
        "follower": "follower",
        "post id": "post_id",
        "post_id": "post_id",
        "konten": "isi_postingan",
        "konten_post": "isi_postingan",
        "post link": "post_link",
        "jmh like": "jumlah_like",
        "jmh comment": "jumlah_komentar",
        "jmh share": "jumlah_share",
        "like": "jumlah_like",
        "likes": "jumlah_like",
        "komentar": "jumlah_komentar",
        "comments": "jumlah_komentar",
        "share": "jumlah_share",
        "shares": "jumlah_share",
        "id": "post_id",
        "user": "user_id",
        "name": "nama",
    }
    df_local.rename(columns={k: v for k, v in col_map.items() if k in df_local.columns}, inplace=True)

    required = ["user_id", "nama", "isi_postingan", "jumlah_like", "jumlah_komentar", "jumlah_share"]
    missing = [c for c in required if c not in df_local.columns]
    if missing:
        st.error(f"Kolom wajib hilang dari dataset Anda: {missing}")
        return None

    if "follower" not in df_local.columns:
        df_local["follower"] = 0
    if "post_id" not in df_local.columns:
        df_local["post_id"] = np.arange(len(df_local)).astype(str)
    if "post_link" not in df_local.columns:
        df_local["post_link"] = ""

    df_local["isi_postingan"] = df_local["isi_postingan"].astype(str).fillna("")
    for col in ["jumlah_like", "jumlah_komentar", "jumlah_share", "follower"]:
        df_local[col] = pd.to_numeric(df_local[col], errors="coerce").fillna(0).astype(int)
    df_local["user_id"] = df_local["user_id"].astype(str)
    df_local["nama"] = df_local["nama"].astype(str)
    df_local["post_id"] = df_local["post_id"].astype(str)

    return df_local


def preprocess_text(s: str) -> str:
    s = str(s).lower()
    s = re.sub(r"http\S+|www\.\S+", " ", s)
    s = re.sub(r"[^0-9a-zA-Z\s#]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def is_radikal_strict(text: str) -> int:
    t = preprocess_text(text)
    has_lex = any(kw in t for kw in lexicon_radikal)
    if not has_lex:
        return 0
    if any(fr in t for fr in konteks_anti_radikal):
        return 0
    has_verb = any(v in t for v in kata_kerja_ajakan_radikal)
    has_praise = any(fr in t for fr in frasa_penguatan_radikal)
    if has_verb or has_praise:
        return 1
    raw_count = sum(1 for kw in lexicon_radikal if kw in t)
    if raw_count >= 3 and not any(k in t for k in ["bahaya", "waspada", "tolak", "melawan", "mencegah"]):
        return 1
    return 0


def detect_sentiment_rule(text: str) -> str:
    t = str(text).lower()
    if any(k in t for k in kata_kunci_negatif):
        return "NEGATIVE"
    if any(k in t for k in kata_kunci_positif):
        return "POSITIVE"
    return "NEUTRAL"


def plot_wordcloud_from_series(series: pd.Series, title: str | None = None):
    txt = " ".join(series.dropna().astype(str).tolist())
    if not txt.strip():
        st.info("Tidak ada teks untuk WordCloud.")
        return

    wc = WordCloud(
        width=1000,
        height=400,
        background_color="white",
        stopwords=STOPWORDS_WORDCLOUD,
        collocations=False,
    ).generate(txt)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    if title:
        ax.set_title(title)
    st.pyplot(fig)


def build_similarity_graph(feature_matrix, df_users, threshold: float = 0.35):
    df_users = df_users.reset_index(drop=True)
    sim = cosine_similarity(feature_matrix)
    G = nx.Graph()

    unique_users = df_users[["user_id", "nama"]].drop_duplicates().set_index("user_id")
    for uid, row in unique_users.iterrows():
        G.add_node(uid, label=row["nama"], posts=0)

    counts = df_users.groupby("user_id").size().to_dict()
    for uid, c in counts.items():
        if uid in G:
            G.nodes[uid]["posts"] = int(c)

    posts_by_user = df_users.groupby("user_id").apply(lambda x: x.index.tolist()).to_dict()
    user_ids = list(posts_by_user.keys())

    for i in range(len(user_ids)):
        for j in range(i + 1, len(user_ids)):
            u = user_ids[i]
            v = user_ids[j]
            idx_u = posts_by_user[u]
            idx_v = posts_by_user[v]
            if len(idx_u) == 0 or len(idx_v) == 0:
                continue
            pair_sims = sim[np.ix_(idx_u, idx_v)]
            max_sim = float(pair_sims.max())
            if max_sim >= threshold:
                G.add_edge(u, v, weight=max_sim)
    return G


def cluster_accounts_by_narrative(feature_matrix, df_users, n_clusters=3):
    user_embeddings = {}
    for uid in df_users["user_id"].unique():
        idx = df_users[df_users["user_id"] == uid].index
        user_embeddings[uid] = feature_matrix[idx].mean(axis=0)

    user_ids = list(user_embeddings.keys())
    X_user = np.vstack([user_embeddings[u] for u in user_ids])

    if len(user_ids) <= 1:
        return {uid: 0 for uid in user_ids}

    clustering = AgglomerativeClustering(
        n_clusters=min(n_clusters, len(user_ids)),
        metric="cosine",
        linkage="average"
    )
    labels = clustering.fit_predict(X_user)
    return dict(zip(user_ids, labels))


def indo_sentiment_predict(texts, tokenizer, model, id2label):
    if tokenizer is None or model is None:
        return [None] * len(texts), [None] * len(texts)

    all_labels = []
    all_scores = []
    device = "cuda" if (hasattr(torch, "cuda") and torch.cuda.is_available()) else "cpu"
    model.to(device)

    batch_size = 16
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model(**enc)
            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()

        for p in probs:
            idx = int(np.argmax(p))
            label = id2label.get(str(idx), id2label.get(idx, str(idx)))
            all_labels.append(label)
            all_scores.append(float(p[idx]))

    return all_labels, all_scores


# ============================================================
# SIDEBAR / STATUS MODEL
# ============================================================
st.sidebar.markdown("## Control Panel")

with st.sidebar:
    st.markdown("### Sistem")
    tokenizer_indobert, model_indobert, id2label_indobert = load_indobert_model()
    embedding_model = load_embedding_model()

    if tokenizer_indobert is not None:
        st.markdown('<span class="status-pill status-ok">IndoBERT Active</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-pill status-danger">IndoBERT Inactive</span>', unsafe_allow_html=True)

    if embedding_model is not None:
        st.markdown('<span class="status-pill status-ok">MiniLM Active</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-pill status-warn">MiniLM Fallback TF-IDF</span>', unsafe_allow_html=True)

    st.caption("Rule-based lexicon berjalan sebagai detektor utama. LLM dipakai sebagai faktor risiko tambahan.")

    st.markdown("---")
    st.markdown("### Data")
    uploaded = st.file_uploader(
        "Upload dataset CSV",
        type=["csv"],
        help="Format umum: Fb Name, UID, Follower, Post ID, Konten, Jmh Like, Jmh Comment, Jmh Share",
    )

    st.markdown("### Parameter Analisis")
    sim_threshold = st.slider(
        "Ambang cosine similarity",
        min_value=0.10,
        max_value=0.80,
        value=0.35,
        step=0.01,
    )
    node_size_base = st.slider(
        "Ukuran dasar node",
        min_value=10,
        max_value=60,
        value=20,
        step=1,
    )

if not uploaded:
    section_intro(
        "Unggah dataset untuk memulai analisis",
        "Aplikasi akan membangun indikator radikal rule-based, skor risiko berbasis LLM, deteksi anomali engagement, dan peta relasi antar akun."
    )
    st.stop()

df = load_csv_safe(uploaded)
if df is None:
    st.stop()

# ============================================================
# PREPROCESSING & FITUR DASAR
# ============================================================
df["clean_text"] = df["isi_postingan"].apply(preprocess_text)
df["total_engagement"] = df["jumlah_like"] + df["jumlah_komentar"] + df["jumlah_share"]
df["indikator_radikal_rule"] = df["isi_postingan"].apply(is_radikal_strict)
df["sentimen_rule"] = df["isi_postingan"].apply(detect_sentiment_rule)

tfidf = TfidfVectorizer(max_features=1500, ngram_range=(1, 2), stop_words=STOPWORDS_TFIDF)
X_tfidf = tfidf.fit_transform(df["clean_text"])

df["llm_sentiment_label"], df["llm_sentiment_score"] = indo_sentiment_predict(
    df["clean_text"].tolist(), tokenizer_indobert, model_indobert, id2label_indobert
)
df["llm_sentiment_label_norm"] = df["llm_sentiment_label"].fillna("").astype(str).str.lower()

sent2risk = {"negative": 1.0, "neutral": 0.4, "positive": 0.0}
df["llm_risk_score"] = df["llm_sentiment_label_norm"].map(sent2risk).fillna(0.0)
df["skor_radikal_final"] = 0.7 * df["indikator_radikal_rule"] + 0.3 * df["llm_risk_score"]
df["indikator_radikal_final"] = (df["skor_radikal_final"] >= 0.7).astype(int)

iso = IsolationForest(contamination=0.12, random_state=42)
try:
    df["anomali_flag"] = iso.fit_predict(df[["total_engagement"]])
    df["anomali_flag"] = df["anomali_flag"].map({1: "Normal", -1: "Anomali"})
except Exception as e:
    st.warning(f"Anomaly detection gagal: {e}")
    df["anomali_flag"] = "Unknown"

if embedding_model is not None:
    try:
        feature_matrix_sna = embedding_model.encode(
            df["clean_text"].tolist(),
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        sna_source = "MiniLM embeddings"
    except Exception as e:
        st.warning(f"Gagal membuat embeddings MiniLM, fallback ke TF-IDF: {e}")
        feature_matrix_sna = X_tfidf.toarray()
        sna_source = "TF-IDF (fallback)"
else:
    feature_matrix_sna = X_tfidf.toarray()
    sna_source = "TF-IDF (fallback)"

# ============================================================
# DATAFRAME RINGKAS
# ============================================================
df_negatif_radikal = df[
    (df["llm_sentiment_label_norm"] == "negative") &
    (df["indikator_radikal_final"] == 1)
].copy().sort_values("total_engagement", ascending=False)

df_prioritas = df_negatif_radikal[df_negatif_radikal["anomali_flag"] == "Anomali"].copy()

total_posts = len(df)
total_users = df["user_id"].nunique()
n_rad_rule = int(df["indikator_radikal_rule"].sum())
n_rad_final = int(df["indikator_radikal_final"].sum())
n_anomali_total = int((df["anomali_flag"] == "Anomali").sum())
n_prioritas = int(len(df_prioritas))

# ============================================================
# HERO HEADER
# ============================================================
hero_left, hero_right = st.columns([1.1, 4.2], gap="large")
with hero_left:
    with st.container():
        st.markdown('<div class="hero-wrap">', unsafe_allow_html=True)
        show_logo_if_exists(width=115)
        st.markdown('</div>', unsafe_allow_html=True)

with hero_right:
    st.markdown(
        f"""
        <div class="hero-wrap">
            <div class="hero-title">Sistem Analisis Konten Radikal</div>
            <div class="hero-subtitle">
                Analisis konten, anomali engagement, dan relasi akun berbasis rule-based, IndoBERT, serta embedding multilingual.
                Dashboard ini menyorot prioritas investigasi dengan fokus pada konten negative-radikal dan pola jaringan narasi.
            </div>
            <div style="margin-top:0.85rem;">
                <span class="status-pill status-ok">Rule-based Active</span>
                <span class="status-pill {'status-ok' if tokenizer_indobert is not None else 'status-danger'}">IndoBERT {'Active' if tokenizer_indobert is not None else 'Inactive'}</span>
                <span class="status-pill {'status-ok' if embedding_model is not None else 'status-warn'}">{sna_source}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ============================================================
# TABS
# ============================================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Dashboard",
    "Analisis Narasi",
    "Prioritas Risiko",
    "Klasifikasi Model",
    "Peta Jaringan",
    "Rekomendasi",
])

# ------------------------------------------------------------
# TAB 1 — DASHBOARD
# ------------------------------------------------------------
with tab1:
    section_intro(
        "Ringkasan Eksekutif",
        "Panel ini memberi gambaran cepat mengenai volume data, tingkat risiko, distribusi konten radikal, dan akun yang perlu menjadi perhatian awal."
    )

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    with k1:
        metric_card("Total Postingan", f"{total_posts:,}", "Seluruh baris data yang berhasil dimuat")
    with k2:
        metric_card("Akun Unik", f"{total_users:,}", "Jumlah user_id unik pada dataset")
    with k3:
        metric_card("Radikal Rule", f"{n_rad_rule:,}", "Terindikasi dari lexicon + heuristik ketat")
    with k4:
        metric_card("Radikal Final", f"{n_rad_final:,}", "Gabungan rule-based dan risiko LLM")
    with k5:
        metric_card("Post Anomali", f"{n_anomali_total:,}", "Engagement tidak wajar menurut IsolationForest")
    with k6:
        metric_card("Prioritas Tinggi", f"{n_prioritas:,}", "Negative + radikal + anomali")

    row1_col1, row1_col2 = st.columns([1.05, 1.15], gap="large")
    with row1_col1:
        section_intro(
            "Proporsi Hasil Deteksi",
            "Perbandingan konten radikal final terhadap seluruh postingan yang dianalisis."
        )
        overview_counts = pd.DataFrame({
            "label": ["Radikal Final", "Non-Radikal"],
            "count": [n_rad_final, total_posts - n_rad_final],
        })
        fig_overview_pie = px.pie(
            overview_counts,
            names="label",
            values="count",
            hole=0.55,
            color="label",
            color_discrete_map={"Radikal Final": "#ef4444", "Non-Radikal": "#334155"},
        )
        apply_plotly_theme(fig_overview_pie, "Proporsi Konten Radikal Final")
        fig_overview_pie.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig_overview_pie, use_container_width=True)

    with row1_col2:
        section_intro(
            "Akun dengan Paparan Radikal Tertinggi",
            "Ranking akun berdasarkan jumlah posting dengan indikator_radikal_final = 1."
        )
        top_rad_user = (
            df[df["indikator_radikal_final"] == 1]
            .groupby(["user_id", "nama"])
            .size()
            .reset_index(name="jumlah_post_radikal")
            .sort_values("jumlah_post_radikal", ascending=False)
            .head(10)
        )
        if not top_rad_user.empty:
            fig_top_users = px.bar(
                top_rad_user,
                x="jumlah_post_radikal",
                y="nama",
                orientation="h",
                text="jumlah_post_radikal",
                color="jumlah_post_radikal",
                color_continuous_scale="Reds",
            )
            apply_plotly_theme(fig_top_users, "Top Akun dengan Jumlah Postingan Radikal Terbanyak")
            fig_top_users.update_layout(yaxis_title="", xaxis_title="Jumlah Postingan")
            fig_top_users.update_coloraxes(showscale=False)
            st.plotly_chart(fig_top_users, use_container_width=True)
        else:
            st.info("Belum ada akun yang memenuhi indikator radikal final.")

    row2_left, row2_right = st.columns([1.25, 1.0], gap="large")
    with row2_left:
        section_intro(
            "Konten Prioritas Investigasi",
            "Lima postingan dengan sinyal risiko paling relevan berdasarkan skor radikal final, sentimen negative, dan engagement."
        )
        priority_source = df_negatif_radikal if not df_negatif_radikal.empty else df[df["indikator_radikal_final"] == 1].sort_values("total_engagement", ascending=False)
        render_priority_cards(priority_source, limit=5)

    with row2_right:
        section_intro(
            "Preview Dataset",
            "Cuplikan data mentah sebagai verifikasi struktur kolom dan nilai utama yang dianalisis sistem."
        )
        st.dataframe(
            df[[
                "nama", "user_id", "post_id", "isi_postingan", "jumlah_like",
                "jumlah_komentar", "jumlah_share", "total_engagement"
            ]].head(25),
            use_container_width=True,
            hide_index=True,
        )

# ------------------------------------------------------------
# TAB 2 — ANALISIS NARASI
# ------------------------------------------------------------
with tab2:
    section_intro(
        "Analisis Narasi dan WordCloud",
        "Tab ini menampilkan distribusi sentimen rule-based, proporsi deteksi radikal, serta kata-kata yang dominan pada seluruh postingan dan subset radikal."
    )

    c1, c2 = st.columns(2, gap="large")
    with c1:
        sent_counts = df["sentimen_rule"].value_counts().reset_index()
        sent_counts.columns = ["sentimen", "count"]
        fig_sent = px.bar(
            sent_counts,
            x="sentimen",
            y="count",
            color="sentimen",
            color_discrete_map={
                "NEGATIVE": "#ef4444",
                "NEUTRAL": "#38bdf8",
                "POSITIVE": "#22c55e",
            },
        )
        apply_plotly_theme(fig_sent, "Distribusi Sentimen Rule-based")
        st.plotly_chart(fig_sent, use_container_width=True)

    with c2:
        rad_counts = df["indikator_radikal_rule"].value_counts().reset_index()
        rad_counts.columns = ["radikal", "count"]
        rad_counts["label"] = rad_counts["radikal"].map({1: "Radikal", 0: "Non-Radikal"})
        fig_rad = px.pie(
            rad_counts,
            names="label",
            values="count",
            hole=0.50,
            color="label",
            color_discrete_map={"Radikal": "#ef4444", "Non-Radikal": "#334155"},
        )
        apply_plotly_theme(fig_rad, "Proporsi Radikal Berdasarkan Rule Ketat")
        fig_rad.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig_rad, use_container_width=True)

    wc1, wc2 = st.columns(2, gap="large")
    with wc1:
        section_intro("WordCloud Seluruh Postingan", "Menunjukkan term dominan dari keseluruhan dataset setelah pembersihan teks.")
        plot_wordcloud_from_series(df["clean_text"], title="WordCloud - Semua Postingan")

    with wc2:
        section_intro("WordCloud Postingan Radikal", "Kata-kata yang paling menonjol pada konten dengan indikator_radikal_rule = 1.")
        plot_wordcloud_from_series(
            df.loc[df["indikator_radikal_rule"] == 1, "clean_text"],
            title="WordCloud - Radikal (Rule Ketat)",
        )

    section_intro(
        "Contoh Postingan Terindikasi Radikal",
        "Tabel berikut menampilkan sampel postingan yang memenuhi aturan lexicon dan heuristik ketat."
    )
    st.dataframe(
        df[df["indikator_radikal_rule"] == 1][[
            "user_id", "nama", "isi_postingan", "total_engagement", "skor_radikal_final"
        ]].reset_index(drop=True),
        use_container_width=True,
        hide_index=True,
    )

# ------------------------------------------------------------
# TAB 3 — PRIORITAS RISIKO
# ------------------------------------------------------------
with tab3:
    section_intro(
        "Prioritas Risiko Negative-Radikal",
        "IsolationForest tetap dihitung dari total engagement. Namun panel ini memusatkan perhatian pada postingan berlabel negative dan terindikasi radikal agar prioritas analisis lebih operasional."
    )

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Negative + Radikal", int(len(df_negatif_radikal)))
    with m2:
        st.metric("Negative + Radikal + Anomali", int(len(df_prioritas)))
    with m3:
        st.metric("Engagement Tertinggi", int(df_negatif_radikal["total_engagement"].max()) if not df_negatif_radikal.empty else 0)

    upper_left, upper_right = st.columns([1.1, 1.0], gap="large")
    with upper_left:
        section_intro(
            "Daftar Prioritas Investigasi",
            "Card di bawah menampilkan konten negative-radikal dengan engagement tertinggi. Gunakan panel ini sebagai shortlist pemeriksaan manual."
        )
        render_priority_cards(df_negatif_radikal, limit=8)

    with upper_right:
        section_intro(
            "Distribusi Engagement",
            "Histogram memperlihatkan sebaran engagement pada subset negative-radikal dan membedakan status anomali."
        )
        if not df_negatif_radikal.empty:
            df_hist = df_negatif_radikal.copy()
            df_hist["kategori_anomali_hist"] = df_hist["anomali_flag"].map({
                "Anomali": "Anomali",
                "Normal": "Normal",
                "Unknown": "Unknown",
            })
            fig_hist = px.histogram(
                df_hist,
                x="total_engagement",
                color="kategori_anomali_hist",
                nbins=20,
                color_discrete_map={
                    "Anomali": "#ef4444",
                    "Normal": "#38bdf8",
                    "Unknown": "#94a3b8",
                },
            )
            apply_plotly_theme(fig_hist, "Histogram Engagement - Konten Negative + Radikal")
            fig_hist.update_layout(
                xaxis_title="Total Engagement",
                yaxis_title="Jumlah Postingan",
                bargap=0.06,
            )
            st.plotly_chart(fig_hist, use_container_width=True)

            fig_scatter = px.scatter(
                df_hist,
                x="total_engagement",
                y="skor_radikal_final",
                color="anomali_flag",
                hover_data=["nama", "user_id", "llm_sentiment_label"],
                color_discrete_map={
                    "Anomali": "#ef4444",
                    "Normal": "#38bdf8",
                    "Unknown": "#94a3b8",
                },
            )
            apply_plotly_theme(fig_scatter, "Sebaran Skor Radikal vs Engagement")
            fig_scatter.update_layout(xaxis_title="Total Engagement", yaxis_title="Skor Radikal Final")
            st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            st.info("Belum ada data negative + radikal untuk divisualisasikan.")

    section_intro(
        "Tabel Lengkap Prioritas",
        "Gunakan tabel ini bila ingin menelaah seluruh detail postingan negative-radikal secara tabular."
    )
    if not df_negatif_radikal.empty:
        st.dataframe(
            df_negatif_radikal[[
                "user_id", "nama", "isi_postingan", "llm_sentiment_label",
                "indikator_radikal_rule", "skor_radikal_final", "total_engagement", "anomali_flag"
            ]].reset_index(drop=True),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("Belum ada postingan yang sekaligus berlabel negative dan terindikasi radikal.")

# ------------------------------------------------------------
# TAB 4 — KLASIFIKASI MODEL
# ------------------------------------------------------------
with tab4:
    section_intro(
        "Klasifikasi Model - IndoBERT Sentiment",
        "Output model publik sentiment classifier digunakan sebagai faktor risiko tambahan. Rule-based tetap menjadi komponen dominan untuk indikator akhir."
    )

    if tokenizer_indobert is None:
        st.warning(
            "IndoBERT belum aktif. Sistem tetap berjalan dengan rule-based dan fallback tanpa label LLM."
        )

    llm_left, llm_right = st.columns([1.3, 0.9], gap="large")
    with llm_left:
        st.dataframe(
            df[[
                "user_id", "nama", "isi_postingan", "indikator_radikal_rule",
                "llm_sentiment_label", "llm_sentiment_score", "llm_risk_score",
                "skor_radikal_final", "indikator_radikal_final"
            ]].head(200),
            use_container_width=True,
            hide_index=True,
        )

    with llm_right:
        if tokenizer_indobert is not None:
            dist_llm = df["llm_sentiment_label"].value_counts(dropna=False).reset_index()
            dist_llm.columns = ["label", "count"]
            fig_llm = px.bar(
                dist_llm,
                x="label",
                y="count",
                color="label",
                color_discrete_map={
                    "negative": "#ef4444",
                    "neutral": "#38bdf8",
                    "positive": "#22c55e",
                },
            )
            apply_plotly_theme(fig_llm, "Distribusi Label IndoBERT")
            st.plotly_chart(fig_llm, use_container_width=True)

        st.markdown(
            """
            <div class="section-card">
                <div class="section-title">Catatan Metodologis</div>
                <div class="section-subtitle">
                    <ul>
                        <li>Model LLM yang dipakai adalah <i>w11wo/indonesian-roberta-base-sentiment-classifier</i>.</li>
                        <li>Label negative diberi bobot risiko lebih tinggi dibanding neutral dan positive.</li>
                        <li><b>skor_radikal_final</b> = 0.7 × rule ketat + 0.3 × risiko LLM.</li>
                        <li>Threshold 0.7 dipakai untuk menetapkan <b>indikator_radikal_final</b>.</li>
                    </ul>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ------------------------------------------------------------
# TAB 5 — PETA JARINGAN
# ------------------------------------------------------------
with tab5:
    section_intro(
        "Social Network Analysis - Akun dengan Konten Radikal",
        f"Graph dibangun dari postingan dengan indikator_radikal_final = 1. Sumber fitur similarity: {sna_source}. Ukuran node merepresentasikan jumlah posting radikal per akun."
    )

    st.markdown(
        """
        <div class="section-card">
            <div class="section-subtitle">
                <span class="legend-dot" style="background:#ef4444;"></span>Ideologis / Doktrinal &nbsp;&nbsp;
                <span class="legend-dot" style="background:#3b82f6;"></span>Kekerasan / Jihad &nbsp;&nbsp;
                <span class="legend-dot" style="background:#facc15;"></span>Rekrutmen / Hijrah
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    df_rad = df[df["indikator_radikal_final"] == 1].copy()
    if len(df_rad) < 2:
        st.info("Belum cukup data untuk membangun graph SNA.")
    else:
        rad_idx = df_rad.index.tolist()
        feature_rad = feature_matrix_sna[rad_idx, :]
        df_posts_rad = df_rad[["user_id", "nama"]].reset_index(drop=True)
        G = build_similarity_graph(feature_rad, df_posts_rad, threshold=sim_threshold)

        if G.number_of_nodes() == 0:
            st.info("Tidak ada node pada threshold saat ini.")
        else:
            account_clusters = cluster_accounts_by_narrative(feature_rad, df_posts_rad, n_clusters=3)
            cluster_info = {
                0: {"color": "#ef4444", "label": "Ideologis / Doktrinal"},
                1: {"color": "#3b82f6", "label": "Kekerasan / Jihad"},
                2: {"color": "#facc15", "label": "Rekrutmen / Hijrah"},
            }

            top_deg = sorted(G.degree, key=lambda x: x[1], reverse=True)[:5]
            max_degree = top_deg[0][1] if top_deg else 0
            graph_cols = st.columns(3)
            with graph_cols[0]:
                st.metric("Node Jaringan", G.number_of_nodes())
            with graph_cols[1]:
                st.metric("Edge Terbentuk", G.number_of_edges())
            with graph_cols[2]:
                st.metric("Degree Tertinggi", max_degree)

            pos = nx.spring_layout(G, seed=42, k=0.6)

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
                line=dict(width=0.7, color="#64748b"),
                hoverinfo="none",
                showlegend=False,
            )

            node_traces = []
            for cid, info in cluster_info.items():
                nx_, ny_, nt_, ns_, texts = [], [], [], [], []
                for n, d in G.nodes(data=True):
                    if account_clusters.get(n) != cid:
                        continue
                    x, y = pos[n]
                    nx_.append(x)
                    ny_.append(y)
                    ns_.append(node_size_base + d.get("posts", 0) * 5)
                    texts.append(G.nodes[n].get("label", ""))
                    nt_.append(
                        f"{d.get('label','')}<br>UID: {n}<br>Posting radikal: {d.get('posts',0)}<br>Kluster: {info['label']}"
                    )

                node_traces.append(
                    go.Scatter(
                        x=nx_,
                        y=ny_,
                        mode="markers+text",
                        name=info["label"],
                        text=texts,
                        textposition="top center",
                        hovertext=nt_,
                        hoverinfo="text",
                        marker=dict(
                            size=ns_,
                            color=info["color"],
                            line=dict(width=1, color="#111827"),
                            opacity=0.92,
                        ),
                    )
                )

            fig_sna = go.Figure(
                data=[edge_trace] + node_traces,
                layout=go.Layout(
                    title=(
                        f"SNA Akun Radikal - Kluster Narasi Internal"
                        f"<br><sup>Sumber fitur: {sna_source}, threshold cosine = {sim_threshold:.2f}</sup>"
                    ),
                    showlegend=True,
                    hovermode="closest",
                    margin=dict(b=20, l=5, r=5, t=70),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(15,23,42,0.50)",
                    font=dict(color="#e2e8f0"),
                ),
            )
            st.plotly_chart(fig_sna, use_container_width=True)

            rows = []
            for uid, deg in top_deg:
                rows.append({
                    "user_id": uid,
                    "nama": G.nodes[uid].get("label", ""),
                    "degree": deg,
                    "jml_post_radikal": G.nodes[uid].get("posts", 0),
                    "kluster_narasi": cluster_info.get(account_clusters.get(uid), {}).get("label", "Tidak diketahui"),
                })
            section_intro(
                "Akun dengan Konektivitas Tertinggi",
                "Degree tertinggi menunjukkan akun yang paling banyak terhubung dalam jaringan kemiripan narasi."
            )
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ------------------------------------------------------------
# TAB 6 — REKOMENDASI
# ------------------------------------------------------------
with tab6:
    section_intro(
        "Rekomendasi Cegah Dini",
        "Saran tindak lanjut berikut disusun dari hasil skor final, engagement anomali, dan struktur jaringan kemiripan narasi."
    )

    rec1, rec2, rec3 = st.columns(3)
    with rec1:
        st.metric("Postingan Radikal Final", n_rad_final)
    with rec2:
        st.metric("Postingan Anomali", n_anomali_total)
    with rec3:
        st.metric("Prioritas Tinggi", n_prioritas)

    left_rec, right_rec = st.columns([1.0, 1.1], gap="large")
    with left_rec:
        section_intro(
            "Akun dengan Indikasi Risiko Tinggi",
            "Akun berikut memiliki jumlah posting radikal final terbanyak pada dataset yang dianalisis."
        )
        top_rad_user = (
            df[df["indikator_radikal_final"] == 1]
            .groupby(["user_id", "nama"])
            .size()
            .reset_index(name="jumlah_post_radikal")
            .sort_values("jumlah_post_radikal", ascending=False)
            .head(5)
        )
        if not top_rad_user.empty:
            st.dataframe(top_rad_user, use_container_width=True, hide_index=True)
        else:
            st.info("Belum ada akun dengan posting radikal final.")

        section_intro(
            "Konten dengan Engagement Tertinggi",
            "Post berikut layak ditinjau karena memiliki jangkauan interaksi paling besar di antara subset radikal final."
        )
        top_eng_rad = (
            df[df["indikator_radikal_final"] == 1]
            .sort_values("total_engagement", ascending=False)[[
                "user_id", "nama", "isi_postingan", "total_engagement", "skor_radikal_final"
            ]]
            .head(5)
        )
        if not top_eng_rad.empty:
            st.dataframe(top_eng_rad, use_container_width=True, hide_index=True)
        else:
            st.info("Belum ada postingan radikal dengan engagement signifikan.")

    with right_rec:
        st.markdown(
            """
            <div class="section-card">
                <div class="section-title">Saran Tindak Lanjut</div>
                <div class="section-subtitle">
                    <ol>
                        <li><b>Fokus pemantauan manual</b> pada akun dengan jumlah posting radikal terbanyak, konten engagement tertinggi, dan node degree tertinggi pada grafik SNA.</li>
                        <li><b>Validasi konteks</b> untuk membedakan apakah postingan benar berupa glorifikasi/ajakan, atau justru kontra-narasi dan edukasi.</li>
                        <li><b>Gunakan kombinasi anomali engagement + SNA</b> untuk mengidentifikasi hub pengaruh dan potensi propagasi narasi.</li>
                        <li><b>Perkuat kontra narasi</b> melalui pesan damai, moderasi keagamaan, dan edukasi publik pada istilah yang sering diselewengkan.</li>
                        <li><b>Pengembangan sistem</b>: siapkan ground truth berlabel manual untuk fine-tuning model radikal khusus, serta tambahkan dimensi waktu untuk membaca eskalasi tren.</li>
                    </ol>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.success("Analisis selesai: UI telah diperbarui dengan layout dashboard yang lebih presentatif.")
