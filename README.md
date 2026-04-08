<div align="center">

# NewsAI

### A newspaper-style recommendation engine that learns your taste in real time

<p>
  <img src="https://img.shields.io/badge/Flask-Web_App-111111?style=flat-square" alt="Flask">
  <img src="https://img.shields.io/badge/scikit--learn-TF--IDF%20Recommender-8B1A1A?style=flat-square" alt="scikit-learn">
  <img src="https://img.shields.io/badge/Dataset-MIND%20Small-E9E4D8?style=flat-square" alt="MIND Small">
  <img src="https://img.shields.io/badge/Frontend-Jinja%20%2B%20Vanilla%20JS-3F3A34?style=flat-square" alt="Frontend">
</p>

<p>
  NewsAI turns article clicks, skips, and topic drift into a living personal edition.
</p>

</div>

---

## The Idea

Most feeds optimize for what is popular.

NewsAI tries to optimize for what feels yours.

It starts with a simple seed, learns from what you read, penalizes what you ignore, blends relevance with popularity and recency, and keeps one slot open for exploration so the experience does not become repetitive too quickly.

## At A Glance

| What it is | Why it stands out |
| --- | --- |
| A Flask-based personalized news recommender | Feels like a digital broadsheet instead of a standard dashboard |
| A content-based ranking engine using TF-IDF and cosine similarity | Updates after every interaction inside the same session |
| A session-aware experience with history tracking | Explains recommendations with similarity cues and keywords |
| A MIND Small dataset project | Includes search cold start, category browse, and diversity injection |

## Visual Tour

### Landing Page

![NewsAI landing page](UI%20Screenshots/Screenshot%202026-04-08%20at%2017.16.36.png)

### Cold Start Experience

![NewsAI cold start search and browse flow](UI%20Screenshots/Screenshot%202026-04-08%20at%2017.17.59.png)

### Adaptive Recommendation Rounds

![NewsAI recommendation rounds with preference progression](UI%20Screenshots/Screenshot%202026-04-08%20at%2017.19.03.png)

## What The App Does

NewsAI gives users a full end-to-end reading loop:

- Register or sign in
- Start with search or category browsing
- Generate a personalized first edition
- Learn from each click and skip
- Explain why stories were recommended
- Track preference shifts across rounds
- Save reading history at the end of a session

## How It Works

### 1. Dataset loading

At startup, the app reads:

- `MINDsmall_train/news.tsv`
- `MINDsmall_train/behaviors.tsv`

It then builds a working article table with:

- `news_id`
- `category`
- `subcategory`
- `title`
- `abstract`
- `url`
- a combined `content` field from title and abstract

### 2. Popularity and recency signals

Historical clicked impressions from `behaviors.tsv` are counted to estimate popularity.

- `popularity_score` is normalized from click counts
- `recency_score` is approximated from article order in the dataset

Those values are later mixed into the final ranking.

### 3. Text representation

The app vectorizes article text using:

`TfidfVectorizer(stop_words="english", max_features=5000)`

That gives the system:

- a TF-IDF matrix for all news items
- a vocabulary for keyword explanations
- fast lookup from `news_id` to vector index

### 4. User profile construction

The profile starts from seed articles and keeps evolving.

It is rebuilt from:

- the original seed articles
- everything the user has clicked in the current session

More recent clicks are weighted more heavily, so the profile adapts instead of staying frozen.

### 5. Negative feedback

When the user opens one story from a recommendation round, the rest of that round is treated as skipped.

Those skipped article vectors are subtracted from the profile with a light penalty, which means the model learns both:

- what the user likes
- what the user ignored

### 6. Recommendation scoring

Candidates are ranked using:

`score = 0.8 × similarity + 0.1 × recency + 0.1 × popularity`

Where:

- `similarity` comes from cosine similarity between the profile vector and article vectors
- `recency` is the normalized freshness proxy
- `popularity` is the normalized click-based popularity score

### 7. Diversity injection

After ranking, the system deliberately keeps one slot for a story from a different subcategory.

That one small decision helps the feed feel less narrow and adds exploration without breaking personalization.

### 8. Explainability

From round 2 onward, the app can explain a recommendation using:

- the previously read article it most resembles
- top TF-IDF keywords from the recommended article

This makes the model easier to trust because the reasoning is visible on screen.

## User Flow

### Landing page

The home page introduces the product with a newspaper-inspired layout and sample articles pulled from the dataset.

### Authentication

Users can register or sign in. Accounts are stored in `users.json`, and passwords are hashed before storage.

### Cold start

Users can begin in two ways:

- Search mode: type a topic and seed the profile from the top matching stories
- Browse mode: choose a category and subcategory, then seed the session from relevant stories while applying a category filter

### Recommendation rounds

Each round includes:

- one lead story
- several supporting stories
- one optional explore story for diversity

Opening an article updates the profile and triggers the next round.

### Profile feedback

As the session continues, the UI surfaces:

- articles read
- skipped recommendations
- dominant topic
- compact interest summary
- preference progression chart across rounds

### Session summary

Ending a session saves clicked articles to the user's persistent history and shows a summary of reading behavior and explored categories.

## Core Features

### Adaptive recommendations

The model updates after every interaction rather than waiting for an offline retrain.

### Cold-start onboarding

Users can get useful recommendations immediately, even with no prior history.

### Transparent recommendations

The system shows recommendation reasons and keywords instead of behaving like a black box.

### Diversity-aware ranking

The explore slot keeps the feed from collapsing too quickly into a single narrow topic.

### Editorial presentation

The interface is intentionally designed like a broadsheet, which makes the output feel curated rather than algorithmically dumped.

## Project Structure

```text
.
├── app.py
├── templates/
│   ├── index.html
│   ├── auth.html
│   ├── features.html
│   └── app.html
├── Notebooks/
│   ├── EDA.ipynb
│   ├── Preprocessing.ipynb
│   ├── ColdStart.ipynb
│   ├── NegativeFeedback.ipynb
│   ├── InteractiveRecommender.ipynb
│   ├── FinalRecommender.ipynb
│   └── Login.ipynb
├── UI Screenshots/
├── MINDsmall_train/
├── requirements.txt
└── users.json
```

## Tech Stack

- Python
- Flask
- Pandas
- NumPy
- scikit-learn
- SciPy
- Jinja2 templates
- HTML, CSS, and vanilla JavaScript

## Setup

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd <your-repo-folder>
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate
```

On Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add the dataset

Place the Microsoft MIND Small dataset files inside:

```text
MINDsmall_train/
```

Expected files used by the app:

- `news.tsv`
- `behaviors.tsv`

The repo already ignores the dataset directory and large `.tsv` files in `.gitignore`, which makes it safer to publish the codebase on GitHub without committing the raw dataset.

### 5. Run the app

```bash
python app.py
```

Then open:

```text
http://localhost:5000
```

## API Surface

The frontend is powered by a small JSON API:

- `POST /api/register`
- `POST /api/login`
- `POST /api/logout`
- `GET /api/me`
- `GET /api/sample-articles`
- `GET /api/categories`
- `POST /api/search`
- `POST /api/start`
- `GET /api/recommend`
- `POST /api/click`
- `POST /api/end-session`

## Why This Project Is Interesting

NewsAI is interesting because it does more than demonstrate a recommender formula. It turns that formula into an actual product experience.

It combines:

- classical NLP
- session-based personalization
- immediate feedback loops
- explainable recommendations
- strong UI identity

That makes it a solid portfolio project for showing machine learning intuition, product thinking, and full-stack execution in one place.
