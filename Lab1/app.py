from __future__ import annotations

import pandas as pd
import requests
from dash import Dash, dcc, html, Input, Output, State, no_update, ALL
from dash.exceptions import PreventUpdate
from pathlib import Path
from sklearn.neighbors import NearestNeighbors
from concurrent.futures import ThreadPoolExecutor

# -------- CONFIG --------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
TMDB_API_KEY = "52c330248a37671b77c03e10d5f4c57d"
TMDB_IMG_BASE = "https://image.tmdb.org/t/p/w500"

movies_path = DATA_DIR / "movies.csv"
links_path = DATA_DIR / "links.csv"
ratings_path = DATA_DIR / "ratings.csv"
tags_path = DATA_DIR / "tags.csv"

# -------- DATA --------
movies = pd.read_csv(movies_path)
links = pd.read_csv(links_path)
ratings = pd.read_csv(ratings_path)
tags = pd.read_csv(tags_path)

movies["title_lc"] = movies["title"].astype(str).str.lower()

movies["genres_clean"] = movies["genres"].replace("(no genres listed)", "").str.strip()
genre_dummies = movies["genres_clean"].str.get_dummies(sep=" ")

link_map = links.set_index("movieId")[["imdbId", "tmdbId"]].to_dict("index")

knn = NearestNeighbors(metric="cosine", algorithm="brute")
knn.fit(genre_dummies)

def search_titles(query: str, limit: int = 20) -> list[dict]:
    """ Returns options for a Dropdown list"""
    q = (query or "").strip().lower()
    if not q:
        return []
    
    starts = movies[movies["title_lc"].str.startswith(q)]
    contains = movies[(~movies.index.isin(starts.index)) & (movies["title_lc"].str.contains(q, na=False))]
    res = pd.concat([starts, contains]).head(limit)
    return [{"label": row["title"], "value": int(row["movieId"])} for _, row in res.iterrows()]



def tmdb_poster_url(movie_id: int) -> str | None:
    """ Fetch poster url from TMDB if API key exists and tmdbId is available"""
    if not TMDB_API_KEY:
        return None
    info = link_map.get(movie_id)
    if not info:
        return None
    tmdb_id = info.get("tmdbId")
    if pd.isna(tmdb_id):
        return None
    
    url = f"https://api.themoviedb.org/3/movie/{int(tmdb_id)}"
    r = requests.get(url, params={"api_key": TMDB_API_KEY, "language": "en-US"}, timeout=10)
    if r.status_code != 200:
        return None
    data = r.json()
    path = data.get("poster_path")
    if not path:
        return None
    return f"{TMDB_IMG_BASE}{path}"

""" A first dummy search , just looking at genre matching and first entry """
# def recommended_dummy(movie_id: int, k: int = 5) -> pd.DataFrame:
#     """ Placeholder dummy recommendations """
#     row = movies.loc[movies["movieId"] == movie_id].head(1)
#     if row.empty:
#         return movies.head(0)
    
#     genres = str(row.iloc[0] ["genres"])
#     g0 = genres.split("|")[0] if genres and genres != "(no genres listed)" else None
#     if not g0:
#         return movies.sample(k)

#     recs = movies[movies["genres"].astype(str).str.contains(g0, na=False)]
#     recs = recs[recs["movieId"] != movie_id].head(k)
#     return recs[["movieId", "title", "genres"]]

def recommended_knn(movie_id: int, k: int = 5) -> pd.DataFrame:
    idx = movies.index[movies["movieId"] == movie_id][0]
    distances, indices = knn.kneighbors([genre_dummies.iloc[idx]], n_neighbors=k+1)
    rec_indices = indices[0][1:]
    return movies.iloc[rec_indices][["movieId", "title", "genres"]]

def fetch_posters(movie_ids):
    with ThreadPoolExecutor(max_workers=5) as executor:
        return list(executor.map(tmdb_poster_url, movie_ids))

    
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

        html.H3("Recommendations"),
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
    Input("selected_movie", "data"),
)
def update_movie(movie_id):
    if not movie_id:
        return no_update, no_update
    
    row = movies.loc[movies["movieId"] == movie_id].head(1)
    if row.empty:
        return html.Div("Movie not found..."), ""

    title = row.iloc[0]["title"]
    genres = row.iloc[0]["genres"]

    poster = tmdb_poster_url(int(movie_id))
    poster_el = (
        html.Img(src=poster, style={"width": "220px", "borderRadius": "12px"})
        if poster
        else html.Div(
            "Ingen poster (lägg TMDB_API_KEY för posters).",
            style={"width": "220px", "height": "330px", "display": "flex", "alignItems": "center"},
        )
    )

    recs_df = recommended_knn(int(movie_id), k=5)
    posters = fetch_posters(recs_df["movieId"].tolist())  # en gång, utanför loopen

    cards = []
    for r, poster in zip(recs_df.itertuples(), posters):  # para ihop film + poster
        cards.append(html.Div(
            style={"textAlign": "center", "width": "150px"},
            children=[
                html.Img(src=poster, style={"width": "150px", "borderRadius": "8px"})
                    if poster else html.Div(style={"width": "150px", "height": "225px", "background": "#eee", "borderRadius": "8px"}),
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
                    html.Div(f"movieId: {movie_id}", style={"opacity": 0.7, "marginTop": "8px"}),
                ]
            ),
        ],
    )
    return panel, rec_list

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