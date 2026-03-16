from __future__ import annotations

import os
import pickle
import numpy as np
import pandas as pd
import requests
from dash import Dash, dcc, html, Input, Output, State, no_update, ALL
from dash.exceptions import PreventUpdate
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
from scipy.sparse import load_npz

# -------- PATH CONFIG --------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
load_dotenv()
TMDB_API_KEY = os.getenv("TMDB_API_KEY", "")
TMDB_IMG_BASE = "https://image.tmdb.org/t/p/w500"
POSTER_PLACEHOLDER = "https://placehold.co/150x225?text=No+Poster"

# -------- MODEL CONFIG --------
K_RECOMMENDATIONS = 5
TFIDF_WEIGHT = 0.40
SVD_WEIGHT = 0.40
RATING_WEIGHT = 0.20

movies_path = DATA_DIR / "movies.csv"
links_path = DATA_DIR / "links.csv"

# -------- DATA & PREPROCESSING --------
movies = pd.read_csv(movies_path)
links = pd.read_csv(links_path)

movies["title_lc"] = movies["title"].astype(str).str.lower()
link_map = links.set_index("movieId")[["imdbId", "tmdbId"]].to_dict("index")

# -------- LOAD PRE-TRAINED MODELS --------
tfidf_matrix = load_npz(MODEL_DIR / "tfidf_matrix.npz")
item_factors = np.load(MODEL_DIR / "item_factors.npy")
movies_tags_merged = pd.read_csv(MODEL_DIR / "movies_tags_merged.csv")

with open(MODEL_DIR / "mappings.pkl", "rb") as f:
    mappings = pickle.load(f)

tfidf_id_to_idx = mappings["tfidf_id_to_idx"]
svd_id_to_idx = mappings["svd_id_to_idx"]
common_ids_list = mappings["common_ids_list"]
tfidf_indices = mappings["tfidf_indices"]
svd_indices = mappings["svd_indices"]

def search_titles(query: str, limit: int = 20) -> list[dict]:
    """ Returns options for a Dropdown list"""
    q = (query or "").strip().lower()
    if not q:
        return []
    
    starts = movies[movies["title_lc"].str.startswith(q)]
    contains = movies[(~movies.index.isin(starts.index)) & (movies["title_lc"].str.contains(q, na=False))]
    res = pd.concat([starts, contains]).head(limit)
    return [{"label": row["title"], "value": int(row["movieId"])} for _, row in res.iterrows()]



_poster_cache: dict[int, str | None] = {}

def tmdb_poster_url(movie_id: int) -> str | None:
    if movie_id in _poster_cache:
        return _poster_cache[movie_id]

    if not TMDB_API_KEY:
        return None

    info = link_map.get(movie_id)
    if not info:
        _poster_cache[movie_id] = None
        return None

    tmdb_id = info.get("tmdbId")
    if pd.isna(tmdb_id):
        _poster_cache[movie_id] = None
        return None

    url = f"https://api.themoviedb.org/3/movie/{int(tmdb_id)}"

    try:
        response = requests.get(url, params={"api_key": TMDB_API_KEY, "language": "en-US"}, timeout=10)
        if response.status_code != 200:
            _poster_cache[movie_id] = None
            return None

        data = response.json()
        path = data.get("poster_path")

        result = f"{TMDB_IMG_BASE}{path}" if path else None
        _poster_cache[movie_id] = result
        return result

    except requests.RequestException:
        _poster_cache[movie_id] = None
        return None 

def fetch_posters(movie_ids):
    with ThreadPoolExecutor(max_workers=5) as executor:
        return list(executor.map(tmdb_poster_url, movie_ids))
    
def minmax(x):
    x = pd.Series(x)
    rng = x.max() - x.min()
    return (x - x.min()) / rng if rng else pd.Series(0.0, index=x.index)  

"""Generate hybrid recommendations using TF-IDF similarity,SVD similarity, and Bayesian weighted ratings."""
def recommend_hybrid(
        movie_id: int, 
        k: int = K_RECOMMENDATIONS, 
        tfidf_weight: float = TFIDF_WEIGHT, 
        svd_weight: float = SVD_WEIGHT, 
        rating_weight: float = RATING_WEIGHT,):
    
    idx_tfidf = tfidf_id_to_idx.get(movie_id)
    if idx_tfidf is None:
        return pd.DataFrame()

    tfidf_scores = cosine_similarity(tfidf_matrix[idx_tfidf], tfidf_matrix).flatten()

    hybrid = pd.DataFrame({"movieId": common_ids_list})
    hybrid["tfidf_score"] = minmax(tfidf_scores[tfidf_indices])

    hybrid = hybrid.merge(
        movies_tags_merged[["movieId", "weighted_rating"]],
        on="movieId", how="left"
    )
    hybrid["weighted_rating"] = minmax(hybrid["weighted_rating"])

    if movie_id in svd_id_to_idx:
        idx_svd = svd_id_to_idx[movie_id]
        svd_scores_full = cosine_similarity([item_factors[idx_svd]], item_factors).flatten()
        hybrid["svd_score"] = minmax(svd_scores_full[svd_indices])

        hybrid["score"] = (
            tfidf_weight * hybrid["tfidf_score"] +
            svd_weight * hybrid["svd_score"] +
            rating_weight * hybrid["weighted_rating"]
        )
    else:
        hybrid["score"] = (
            tfidf_weight * hybrid["tfidf_score"] + 
            rating_weight * hybrid["weighted_rating"]
        )

    hybrid = hybrid[hybrid["movieId"] != movie_id]
    top = hybrid.nlargest(k, "score")
    return top.merge(movies[["movieId", "title", "genres"]], on="movieId", how="left")[["movieId", "title", "genres"]]
    
# -------- APP --------
app = Dash(__name__)
app.title = "Movie Recommendation 3000"

app.layout = html.Div(
    style={"maxWidth": "1200px", "margin": "24px auto", "fontFamily": "system-ui"},
    children=[
        dcc.Store(id="selected_movie"),

        html.H2("Movie Recommendations"),

        html.Div(
            style={"display": "flex", "gap": "12px", "alignItems": "center"},
            children=[
                dcc.Input(
                    id="q",
                    type="text",
                    placeholder="Search for a movie (example, Star Wars)",
                    style={"flex": "1"},
                ),
            ],
        ),

        html.Div(
            id="suggestions",
            style={
                "marginTop": "8px",
                "borderRadius": "12px",
                "overflow": "hidden",
                "boxShadow": "0 6px 18px rgba(0,0,0,0.10)",
                "background": "white",
            },
        ),

        html.Hr(),
        html.Div(id="movie_panel"),

        html.H3("Recommendations (Click on a poster to continue searching)", id="recs_header", style={"display": "none"}),
        html.Div(id="recs"),

        html.Hr(),

        html.Div(
            style={"textAlign": "center", "marginTop": "24px"},
            children=[
                html.Img(
                    src="https://www.themoviedb.org/assets/2/v4/logos/v2/blue_short-8e7b30f73a4020692ccca9c88bafe5dcb6f8a62a4c6bc55cd9ba82bb2cd95f6c.svg",
                    style={"height": "40px"},
                ),
                html.P("This product uses the TMDB API but is not endorsed or certified by TMDB."),
            ],
        ),
    ],
)

@app.callback(
    Output("suggestions", "children"),
    Input("q", "value"),
)
def render_suggestions(q):
    opts = search_titles(q, limit=10)  # [{'label':..., 'value':...}]
    if not opts:
        return []

    rows = []
    for o in opts:
        rows.append(
            html.Button(
                o["label"],
                id={"type": "sug", "movieId": o["value"]},
                n_clicks=0,
                style={
                    "width": "100%",
                    "textAlign": "left",
                    "padding": "10px 12px",
                    "border": "0",
                    "borderBottom": "1px solid #eee",
                    "background": "white",
                    "cursor": "pointer",
                    "fontSize": "14px",
                },
            )
        )
    rows[-1].style["borderBottom"] = "0"
    return rows

@app.callback(
    Output("movie_panel", "children"),
    Output("recs", "children"),
    Output("recs_header", "style"),
    Input("selected_movie", "data"),
)
def update_movie(movie_id):
    if not movie_id:
        return no_update, no_update, {"display": "none"}
    
    row = movies.loc[movies["movieId"] == movie_id].head(1)
    if row.empty:
        return html.Div("Movie not found..."), ""

    title = row.iloc[0]["title"]
    genres = row.iloc[0]["genres"]

    poster = tmdb_poster_url(int(movie_id))
    poster_el = html.Img(
        src=poster or POSTER_PLACEHOLDER,
        style={"width": "220px", "borderRadius": "12px"}
    )

    recs_df = recommend_hybrid(int(movie_id))
    posters = fetch_posters(recs_df["movieId"].tolist())

    cards = []
    for r, poster in zip(recs_df.itertuples(), posters):
        cards.append(html.Div(
            style={"textAlign": "center", "width": "150px"},
            children=[
                html.Button(
                    html.Img(
                        src=poster or POSTER_PLACEHOLDER,
                        style={"width": "150px", "borderRadius": "8px"}
                    ),
                    id={"type": "sug", "movieId": int(r.movieId)},
                    n_clicks=0,
                    style={
                        "border": "0",
                        "padding": "0",
                        "background": "transparent",
                        "cursor": "pointer",
                    }
                ),
                html.P(r.title, style={"fontSize": "12px", "marginTop": "8px"}),
            ]
        ))

    rec_list = html.Div(cards, style={"display": "flex", "gap": "12px", "flexWrap": "wrap"})

    panel = html.Div(
        style={"display": "flex", "gap": "16px"},
        children=[
            poster_el,
            html.Div(
                children=[
                    html.H3(title),
                    html.Div(f"Genres: {genres}"),
                ]
            ),
        ],
    )
    return panel, rec_list, {"display": "block"}

@app.callback(
    Output("selected_movie", "data"),
    Output("q", "value"), 
    Input({"type": "sug", "movieId": ALL}, "n_clicks"),
    State({"type": "sug", "movieId": ALL}, "id"),
    prevent_initial_call=True,
)
def choose_movie(clicks, ids):
    if not clicks or max(clicks) == 0:
        raise PreventUpdate

    idx = clicks.index(max(clicks))
    return ids[idx]["movieId"], ""

if __name__ == "__main__":
    app.run()