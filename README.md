# svd-recommendation-engine
A Collaborative Filtering recommendation engine built with Python. Uses Singular Value Decomposition (SVD) and Matrix Factorization on the MovieLens dataset to predict user ratings and generate Top-10 personalized movie suggestions.
# 🎬 SVD Movie Recommendation Engine

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-%230C55A5.svg?style=for-the-badge&logo=scipy&logoColor=white)

A high-performance Collaborative Filtering (User-Item) recommendation engine. This system analyzes 100,000+ data points from the MovieLens dataset to uncover hidden behavioral patterns and provide highly personalized "Top-10" movie suggestions.

## 🧠 Methodology: Matrix Factorization (SVD)

Unlike simple popularity-based recommenders, this system uses **Singular Value Decomposition (SVD)** to predict how a user would rate a movie they haven't seen yet. 

The pipeline is broken down into four core phases:
1. **Data Pipeline:** Merges user ratings with movie metadata and pivots the data into a large User-Item Matrix. Data is demeaned (normalized) to account for "strict" vs. "generous" raters.
2. **Matrix Factorization:** Uses `scipy.sparse.linalg.svds` to decompose the matrix, extracting 50 latent features (hidden patterns) that represent underlying user tastes and movie attributes.
3. **Evaluation:** Calculates the Root Mean Squared Error (RMSE) against the known ratings to evaluate the system's predictive accuracy, achieving a highly optimized score of 1.85.
4. **Recommendation Generation:** Filters out previously watched movies and returns the Top-10 highest-predicted movies for any given user.

## 🛠️ Tech Stack
* **Language:** Python
* **Data Processing:** Pandas, NumPy
* **Machine Learning / Math:** SciPy (SVD), Scikit-learn (RMSE calculation)

## 🚀 Getting Started

### Prerequisites
* Python 3.8+
* [MovieLens 100K Dataset](https://grouplens.org/datasets/movielens/100k/) (`u.data` and `u.item` files)

### Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/dishasonawat15/svd-recommendation-engine.git](https://github.com/dishasonawat15/svd-recommendation-engine.git)
   cd svd-recommendation-engine
   
