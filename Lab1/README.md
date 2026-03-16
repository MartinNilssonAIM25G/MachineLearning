# Movie Recommendation System

This project implements a hybrid movie recommender system using the MovieLens dataset.  
The system combines content-based filtering using TF-IDF representations of movie genres and tags with collaborative filtering based on Truncated SVD.

The final recommendations are generated using a hybrid scoring function combining TF-IDF similarity, SVD similarity, and a Bayesian weighted rating.

---

## Dataset

The system is built using the MovieLens dataset provided by GroupLens.

Harper, F. M., & Konstan, J. A. (2015).  
*The MovieLens Datasets: History and Context.*  
ACM Transactions on Interactive Intelligent Systems, 5(4), 19.

---

## Model

The recommender combines three components:

- **Content-based filtering** using TF-IDF on genres and user tags
- **Collaborative filtering** using Truncated SVD on the user–item rating matrix
- **Hybrid scoring** combining both similarity signals with a Bayesian weighted rating

Final score:

    score = 0.40 * TF-IDF similarity
          + 0.40 * SVD similarity
          + 0.20 * Bayesian rating

---

## Project Structure

    app.py                # main application
    report.md             # written report
    eda_overview.png      # dataset analysis figure

    models/
      hyperparams.json
      item_factors.npy
      mappings.pkl
      movies_tags_merged.csv
      tfidf_matrix.npz

The trained model artifacts are stored in `models/` so the system can run without retraining.

---

## Running the Project

Run the application:

    python app.py

---

## Notes

This project was developed as part of a Machine Learning course assignment.

---

*This product uses the TMDB API but is not endorsed or certified by TMDB.*