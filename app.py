"""
NewsAI — Flask Web Application
================================
Start the server:
    pip install flask pandas scikit-learn scipy
    python app.py

Then open:  http://localhost:5000
"""

import os, json, uuid, hashlib
from flask import Flask, request, jsonify, session, render_template, redirect, url_for

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import vstack
from collections import Counter

# ── App setup ──────────────────────────────────────────────────────────────────
app = Flask(__name__)

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
USERS_FILE  = os.path.join(BASE_DIR, "users.json")
SECRET_FILE = os.path.join(BASE_DIR, ".secret_key")
DATA_DIR    = os.path.join(BASE_DIR, "MINDsmall_train")

# Persist secret key so sessions survive server restarts
if os.path.exists(SECRET_FILE):
    with open(SECRET_FILE) as f:
        app.secret_key = f.read().strip()
else:
    key = os.urandom(24).hex()
    with open(SECRET_FILE, "w") as f:
        f.write(key)
    app.secret_key = key

# ── Load data at startup ───────────────────────────────────────────────────────
print("Loading news dataset...")
news_df = pd.read_csv(
    os.path.join(DATA_DIR, "news.tsv"),
    sep="\t", header=None,
    names=["news_id","category","subcategory","title","abstract",
           "url","title_entities","abstract_entities"]
)
news_df["title"]    = news_df["title"].fillna("")
news_df["abstract"] = news_df["abstract"].fillna("")
news_df["url"]      = news_df["url"].fillna("")
news_df = news_df[["news_id","category","subcategory","title","abstract","url"]]
news_df["content"]  = news_df["title"] + " " + news_df["abstract"]

print("Loading behavior data...")
behaviors_df = pd.read_csv(
    os.path.join(DATA_DIR, "behaviors.tsv"),
    sep="\t", header=None,
    names=["impression_id","user_id","time","history","impressions"]
)

print("Computing popularity scores...")
click_counts = Counter()
for imp in behaviors_df["impressions"].fillna(""):
    for item in str(imp).split():
        parts = item.split("-")
        if len(parts) == 2 and parts[1] == "1":
            click_counts[parts[0]] += 1

news_df["popularity_raw"]   = news_df["news_id"].map(click_counts).fillna(0)
mx = news_df["popularity_raw"].max()
news_df["popularity_score"] = news_df["popularity_raw"] / mx if mx > 0 else 0.0
news_df["recency_score"]    = news_df.index / len(news_df)

print("Fitting TF-IDF (this takes ~30s)...")
tfidf        = TfidfVectorizer(stop_words="english", max_features=5000)
tfidf_matrix = tfidf.fit_transform(news_df["content"])
feature_names    = tfidf.get_feature_names_out()
news_id_to_index = {nid: idx for idx, nid in enumerate(news_df["news_id"])}

# Pre-compute category hierarchy
_cat_raw = news_df[["category","subcategory"]].drop_duplicates()
CATEGORIES = {}
for _, row in _cat_raw.iterrows():
    CATEGORIES.setdefault(row["category"], []).append(row["subcategory"])
CATEGORIES = {k: sorted(v) for k, v in CATEGORIES.items()}

print(f"Ready! {len(news_df):,} articles · {len(CATEGORIES)} categories")

# ── In-memory recommendation sessions ─────────────────────────────────────────
rec_sessions = {}  # rec_sid → state dict

# ── Auth helpers ───────────────────────────────────────────────────────────────
def _hash(pw):
    return hashlib.sha256(pw.encode()).hexdigest()

def _load_users():
    if not os.path.exists(USERS_FILE):
        return {"users": []}
    with open(USERS_FILE) as f:
        return json.load(f)

def _save_users(d):
    with open(USERS_FILE, "w") as f:
        json.dump(d, f, indent=2)

def _find_user(email):
    for u in _load_users()["users"]:
        if u["email"].lower() == email.lower():
            return u
    return None

# ── Recommender core ───────────────────────────────────────────────────────────
def build_profile(clicked_ids):
    vecs, wts = [], []
    n = len(clicked_ids)
    for i, nid in enumerate(clicked_ids):
        if nid in news_id_to_index:
            vecs.append(tfidf_matrix[news_id_to_index[nid]])
            wts.append((i + 1) / n)
    if not vecs:
        return None
    s = vstack(vecs)
    w = np.array(wts).reshape(-1, 1)
    return np.asarray(s.multiply(w).sum(axis=0) / w.sum())

def neg_feedback(profile, skipped, weight=0.05):
    if not skipped or profile is None:
        return profile
    p = profile.copy().astype(float)
    for nid in skipped:
        if nid in news_id_to_index:
            p -= weight * np.asarray(
                tfidf_matrix[news_id_to_index[nid]].todense()).flatten()
    return np.clip(p, 0, None)

def get_recs(profile, exclude, category=None, top_n=5, a=0.8, b=0.1, g=0.1, pool=150):
    sims = cosine_similarity(profile, tfidf_matrix).flatten()
    idxs = sims.argsort()[::-1]
    cands = []
    for idx in idxs:
        row = news_df.iloc[idx]
        nid = row["news_id"]
        if nid in exclude:
            continue
        if category and row["category"] != category:
            continue
        s = float(sims[idx])
        r = float(row["recency_score"])
        p = float(row["popularity_score"])
        cands.append({
            "news_id":     nid,
            "title":       row["title"],
            "abstract":    str(row["abstract"]),
            "category":    row["category"],
            "subcategory": row["subcategory"],
            "url":         row["url"],
            "sim":         round(s, 4),
            "rec":         round(r, 4),
            "pop":         round(p, 4),
            "score":       round(a*s + b*r + g*p, 4),
            "is_diverse":  False,
            "explanation": None,
        })
        if len(cands) >= pool:
            break
    cands.sort(key=lambda x: x["score"], reverse=True)
    return cands[:top_n]

def inject_diversity(cands, exclude):
    if len(cands) < 2:
        return cands
    dom_sub = cands[0]["subcategory"]
    shown   = {c["news_id"] for c in cands} | set(exclude)
    pool    = news_df[
        (news_df["subcategory"] != dom_sub) &
        (~news_df["news_id"].isin(shown))
    ].sort_values("popularity_score", ascending=False)
    if pool.empty:
        return cands
    row = pool.iloc[0]
    div = {
        "news_id":     row["news_id"],
        "title":       row["title"],
        "abstract":    str(row["abstract"]),
        "category":    row["category"],
        "subcategory": row["subcategory"],
        "url":         row["url"],
        "sim":  0.0,
        "rec":  round(float(row["recency_score"]), 4),
        "pop":  round(float(row["popularity_score"]), 4),
        "score": 0.0,
        "is_diverse":  True,
        "explanation": None,
    }
    return cands[:-1] + [div]

def do_search(query, top_n=6):
    if not query.strip():
        return []
    qv   = tfidf.transform([query])
    sims = cosine_similarity(qv, tfidf_matrix).flatten()
    idxs = sims.argsort()[::-1]
    out  = []
    for idx in idxs:
        if sims[idx] == 0:
            break
        row = news_df.iloc[idx]
        out.append({
            "news_id":     row["news_id"],
            "title":       row["title"],
            "abstract":    str(row["abstract"]),
            "category":    row["category"],
            "subcategory": row["subcategory"],
            "url":         row["url"],
            "sim":   round(float(sims[idx]), 4),
            "rec":   round(float(row["recency_score"]), 4),
            "pop":   round(float(row["popularity_score"]), 4),
            "score": round(float(sims[idx]), 4),
            "is_diverse":  False,
            "explanation": None,
        })
        if len(out) >= top_n:
            break
    return out

def make_explanation(article, clicked_ids, nkw=3):
    nid = article["news_id"]
    if nid not in news_id_to_index or not clicked_ids:
        return {"because": None, "keywords": []}
    av = tfidf_matrix[news_id_to_index[nid]]
    bt, bs = None, -1.0
    for cid in clicked_ids:
        if cid not in news_id_to_index:
            continue
        sim = cosine_similarity(av, tfidf_matrix[news_id_to_index[cid]])[0][0]
        if sim > bs:
            bs = sim
            r  = news_df[news_df["news_id"] == cid]
            if not r.empty:
                bt = r.iloc[0]["title"]
    aa  = np.asarray(av.todense()).flatten()
    ti  = aa.argsort()[::-1]
    kw  = [feature_names[i] for i in ti if aa[i] > 0][:nkw]
    return {"because": bt, "keywords": kw}

def interest_summary(clicked_ids):
    rows = news_df[news_df["news_id"].isin(clicked_ids)]
    if rows.empty:
        return {"dominant": "—", "summary": {}}
    summary  = rows["category"].value_counts().to_dict()
    dominant = rows["category"].value_counts().idxmax()
    return {"dominant": dominant, "summary": summary}

# ── Page routes ────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/auth")
def auth():
    if "user_email" in session:
        return redirect(url_for("app_view"))
    return render_template("auth.html")

@app.route("/app")
def app_view():
    if "user_email" not in session:
        return redirect(url_for("auth"))
    user = _find_user(session["user_email"]) or {}
    return render_template(
        "app.html",
        user_name=user.get("name", ""),
        user_email=user.get("email", ""),
    )

# ── Auth API ───────────────────────────────────────────────────────────────────
@app.route("/api/login", methods=["POST"])
def api_login():
    d  = request.json or {}
    u  = _find_user(d.get("email", "").strip())
    pw = d.get("password", "").strip()
    if not u or u["password_hash"] != _hash(pw):
        return jsonify({"error": "Incorrect email or password."}), 401
    session["user_email"] = u["email"]
    return jsonify({
        "name":          u["name"],
        "email":         u["email"],
        "history_count": len(u.get("history", [])),
    })

@app.route("/api/register", methods=["POST"])
def api_register():
    d     = request.json or {}
    name  = d.get("name", "").strip()
    email = d.get("email", "").strip()
    pw    = d.get("password", "").strip()
    if not all([name, email, pw]):
        return jsonify({"error": "All fields are required."}), 400
    if "@" not in email or "." not in email:
        return jsonify({"error": "Please enter a valid email address."}), 400
    if len(pw) < 6:
        return jsonify({"error": "Password must be at least 6 characters."}), 400
    if _find_user(email):
        return jsonify({"error": "An account with this email already exists."}), 409
    ud = _load_users()
    nu = {"name": name, "email": email, "password_hash": _hash(pw), "history": []}
    ud["users"].append(nu)
    _save_users(ud)
    session["user_email"] = email
    return jsonify({"name": name, "email": email, "history_count": 0})

@app.route("/api/logout", methods=["POST"])
def api_logout():
    session.clear()
    return jsonify({"ok": True})

@app.route("/api/me")
def api_me():
    if "user_email" not in session:
        return jsonify({"error": "Not logged in"}), 401
    u = _find_user(session["user_email"])
    return jsonify({"name": u["name"], "email": u["email"]})

# ── Recommender API ────────────────────────────────────────────────────────────
@app.route("/api/categories")
def api_categories():
    return jsonify(CATEGORIES)

@app.route("/api/search", methods=["POST"])
def api_search():
    q = (request.json or {}).get("query", "")
    return jsonify(do_search(q))

@app.route("/api/start", methods=["POST"])
def api_start():
    if "user_email" not in session:
        return jsonify({"error": "Not logged in"}), 401
    d          = request.json or {}
    seed_ids   = d.get("seed_ids", [])
    cat_filter = d.get("category", None)
    profile    = build_profile(seed_ids)
    if profile is None:
        return jsonify({"error": "Could not build profile from seeds."}), 400
    sid = str(uuid.uuid4())
    rec_sessions[sid] = {
        "clicked_ids":   seed_ids[:],
        "profile":       profile.tolist(),
        "total_skipped": 0,
        "round":         0,
        "cat_filter":    cat_filter,
        "log":           [],
    }
    session["rec_sid"] = sid
    return jsonify({"ok": True})

@app.route("/api/recommend")
def api_recommend():
    sid = session.get("rec_sid")
    rs  = rec_sessions.get(sid) if sid else None
    if not rs:
        return jsonify({"error": "No active session. Please start again."}), 400
    profile     = np.array(rs["profile"])
    clicked_ids = rs["clicked_ids"]
    cat_filter  = rs["cat_filter"]
    rs["round"] += 1
    recs = get_recs(profile, clicked_ids, category=cat_filter)
    if not recs:
        rs["cat_filter"] = None
        recs = get_recs(profile, clicked_ids)
    recs = inject_diversity(recs, clicked_ids)
    if rs["round"] > 1:
        for rec in recs:
            rec["explanation"] = make_explanation(rec, clicked_ids)
    return jsonify({"round": rs["round"], "articles": recs})

@app.route("/api/click", methods=["POST"])
def api_click():
    sid = session.get("rec_sid")
    rs  = rec_sessions.get(sid) if sid else None
    if not rs:
        return jsonify({"error": "No active session."}), 400
    d           = request.json or {}
    clicked_id  = d.get("news_id")
    skipped_ids = d.get("skipped_ids", [])
    if clicked_id not in rs["clicked_ids"]:
        rs["clicked_ids"].append(clicked_id)
    profile = build_profile(rs["clicked_ids"])
    if skipped_ids:
        rs["total_skipped"] += len(skipped_ids)
        profile = neg_feedback(profile, skipped_ids)
    rs["profile"] = profile.tolist()
    rs["log"].append({
        "round":         rs["round"],
        "clicked":       clicked_id,
        "skipped_count": len(skipped_ids),
    })
    info = interest_summary(rs["clicked_ids"])
    return jsonify({
        "ok":             True,
        "history_count":  len(rs["clicked_ids"]),
        "total_skipped":  rs["total_skipped"],
        **info,
    })

@app.route("/api/end-session", methods=["POST"])
def api_end_session():
    sid         = session.get("rec_sid")
    rs          = rec_sessions.get(sid, {}) if sid else {}
    clicked_ids = rs.get("clicked_ids", [])
    # Persist reading history
    if "user_email" in session and clicked_ids:
        ud = _load_users()
        for u in ud["users"]:
            if u["email"].lower() == session["user_email"].lower():
                existing = set(u.get("history", []))
                u["history"] = u.get("history", []) + [
                    nid for nid in clicked_ids if nid not in existing
                ]
                break
        _save_users(ud)
    # Clean up
    if sid:
        rec_sessions.pop(sid, None)
    session.pop("rec_sid", None)
    info = interest_summary(clicked_ids)
    return jsonify({
        "ok":           True,
        "rounds":       rs.get("round", 0),
        "articles_read": len(clicked_ids),
        "skips":        rs.get("total_skipped", 0),
        **info,
        "log":          rs.get("log", []),
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)
