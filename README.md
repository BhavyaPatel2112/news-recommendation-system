# NewsAI

An editorial-style news recommendation web app built with Flask, TF-IDF, and the Microsoft MIND Small dataset.

NewsAI learns from what a user reads, what they skip, and how their interests evolve across a session. Instead of showing a static feed, it builds a live preference profile and updates recommendations round by round.

## Overview

This project combines a lightweight recommender system with a polished newspaper-inspired interface. A user can begin in two ways:

- Search for a topic they care about
- Browse by category and subcategory

Those starting choices seed an initial profile. From there, every article the user opens becomes positive feedback, every non-selected recommendation becomes negative feedback, and the system generates the next round of suggestions.

At a high level, NewsAI is designed to answer a simple question:

How can we make news recommendations feel personal, transparent, and adaptive without requiring a heavy deep-learning pipeline?

## What The App Does

NewsAI provides:

- User registration and login
- Cold-start onboarding for new users
- Content-based recommendations using TF-IDF and cosine similarity
- Negative feedback handling for skipped articles
- Ranking that blends similarity, popularity, and recency
- A diversity injection step to reduce repetitive feeds
- Recommendation explanations such as "Because you read..."
- Session-level preference tracking and summary views
- Persistent reading history stored locally per user

## How It Works

### 1. Dataset loading

On startup, the app loads:

- `MINDsmall_train/news.tsv` for article metadata
- `MINDsmall_train/behaviors.tsv` for historical user behavior

It builds a working article table containing:

- `news_id`
- `category`
- `subcategory`
- `title`
- `abstract`
- `url`
- combined text content (`title + abstract`)

### 2. Popularity and recency signals

The app scans the behavior logs and counts clicked impressions to estimate article popularity.

- `popularity_score` is normalized from observed click counts
- `recency_score` is approximated from the article row index within the dataset

These scores are later blended into the final recommendation score.

### 3. Text representation

All articles are vectorized with `TfidfVectorizer(stop_words="english", max_features=5000)`.

This produces:

- a TF-IDF matrix for all articles
- a feature vocabulary used later for keyword-based explanations
- a fast mapping from `news_id` to matrix index

### 4. User profile construction

When a session starts, the system builds a profile vector from seed articles. As the user continues reading, the profile is rebuilt from:

- the initial seed articles
- all clicked articles in the current session

Recent clicks are weighted more heavily than earlier ones, so the profile can shift as the session evolves.

### 5. Negative feedback

Whenever a user opens one article from a recommendation round, the remaining articles from that round are treated as skipped. Their vectors are subtracted from the active profile with a small penalty weight.

That means the system does not only learn what the user likes; it also learns what they ignored.

### 6. Recommendation scoring

Candidate articles are ranked with a weighted combination of:

`score = 0.8 × similarity + 0.1 × recency + 0.1 × popularity`

Where:

- `similarity` comes from cosine similarity between the user profile and article TF-IDF vectors
- `recency` is the normalized recency proxy
- `popularity` is the normalized click popularity from behavior data

### 7. Diversity injection

After ranking, the app intentionally reserves one slot for a different subcategory. This prevents the feed from collapsing too aggressively into one narrow topic and creates a lightweight exploration path.

### 8. Explainability

From round 2 onward, recommendations can include:

- the previously read article most similar to the recommendation
- top TF-IDF keywords associated with the recommended article

This gives users a visible reason for why an article appeared in their feed.

## User Flow

### Landing page

The home page introduces the product in a newspaper-style layout and shows sample articles pulled from the dataset.

### Authentication

Users can register or sign in. Accounts are stored locally in `users.json`, and passwords are hashed with SHA-256 before storage.

### Cold start

Inside the app, the user chooses one of two onboarding modes:

- Search mode: enter a topic, then use the top matching articles as seeds
- Browse mode: choose a category and subcategory, then seed from matching search results and apply a category filter

### Recommendation rounds

Each round shows:

- one lead article
- several supporting recommendations
- one optional "Explore" article for diversity

Selecting an article updates the profile and loads the next round.

### Profile feedback

As the session progresses, the interface shows:

- number of articles read
- number of skipped items
- dominant category
- a compact interest summary
- a preference progression chart across rounds

### Session summary

Ending a session saves clicked articles into the user's persistent history and displays a summary of:

- total rounds
- total articles read
- total skips
- categories explored

## Core Features

### Adaptive recommendations

The model updates after every interaction instead of waiting for a full retrain.

### Strong cold-start handling

New users can begin immediately through search or browsing, without needing a long history first.

### Transparent recommendations

The system exposes recommendation reasons and keywords instead of acting like a black box.

### Diversity-aware feed design

The deliberate "Explore" slot helps reduce narrow repetition and encourages broader discovery.

### Editorial UI

The front end is intentionally styled like a digital broadsheet rather than a generic dashboard, which makes the recommendation output feel more like a curated edition.

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

The repository currently ignores the dataset directory and large `.tsv` files in `.gitignore`, which is useful for GitHub because these files are too large to commit normally.

### 5. Run the app

```bash
python app.py
```

Then open:

```text
http://localhost:5000
```

## Notes About Runtime

- The app precomputes TF-IDF vectors at startup, so initial launch can take a little time
- The code comments mention TF-IDF fitting may take around 30 seconds depending on machine speed
- Session state is stored in memory during runtime
- User account data and persistent reading history are stored locally in `users.json`
- The Flask secret key is persisted in `.secret_key` so browser sessions survive server restarts

## API Surface

The backend exposes simple JSON endpoints that drive the UI:

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

This project is a good example of how far a recommendation system can go with clean interaction design and thoughtful ranking logic, even without neural recommenders or large-scale infrastructure.

It combines:

- classical NLP
- session-based personalization
- transparent UX
- behavior-aware ranking
- a clear end-to-end product experience

That makes it useful both as a working application and as a portfolio project that demonstrates product thinking, machine learning intuition, and full-stack implementation.

## Current Limitations

- Authentication is local and file-based, not production-grade
- Password hashing uses plain SHA-256 without salting
- Session storage is in memory, so active recommendation sessions reset if the process stops
- Recency is approximated from dataset ordering rather than article publish timestamps
- The recommender is content-based, so it does not yet use collaborative filtering or learned embeddings
- Some original article URLs in the dataset may no longer be live

## Future Improvements

- Move user storage to a real database
- Add salted password hashing with a stronger password library
- Persist recommendation sessions beyond process memory
- Add hybrid recommendations that combine content and collaborative signals
- Improve recency with explicit publication timestamps
- Add evaluation metrics such as precision@k, diversity, novelty, and session retention
- Deploy the app publicly with production-ready configuration

## Screenshots

The repository already includes UI captures in `UI Screenshots/`, which can be added to the GitHub presentation section later if you want to curate a visual gallery.

## Author

Bhavya Patel

## License

Add your preferred license here before publishing to GitHub.
