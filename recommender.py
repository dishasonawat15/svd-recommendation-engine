import pandas as pd
import numpy as np
from scipy.sparse.linalg import svds
from sklearn.metrics import mean_squared_error

print("Loading 100,000+ data points...")

ratings_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings = pd.read_csv('u.data', sep='\t', names=ratings_cols, encoding='latin-1')

movies_cols = ['movie_id', 'title', 'release_date', 'video_release_date', 'imdb_url']
movies = pd.read_csv('u.item', sep='|', names=movies_cols, usecols=range(5), encoding='latin-1')

df = pd.merge(ratings, movies[['movie_id', 'title']], on='movie_id')
user_item_matrix = df.pivot(index='user_id', columns='movie_id', values='rating').fillna(0)

R = user_item_matrix.values
user_ratings_mean = np.mean(R, axis=1)
R_demeaned = R - user_ratings_mean.reshape(-1, 1)

print("Applying Matrix Factorization...")
U, sigma, Vt = svds(R_demeaned, k=50)
sigma = np.diag(sigma)

all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)
preds_df = pd.DataFrame(all_user_predicted_ratings, columns=user_item_matrix.columns)

print("Calculating RMSE...")
def calculate_rmse(actual, predicted):
    actual_flat = actual.flatten()
    predicted_flat = predicted.flatten()
    nonzero_indices = actual_flat.nonzero()
    actual_nonzero = actual_flat[nonzero_indices]
    predicted_nonzero = predicted_flat[nonzero_indices]
    mse = mean_squared_error(actual_nonzero, predicted_nonzero)
    return np.sqrt(mse)

rmse_score = calculate_rmse(R, all_user_predicted_ratings)
print(f"System RMSE: {rmse_score:.2f}")

def recommend_movies(preds_df, user_id, movies_df, original_ratings_df, num_recommendations=10):
    print(f"\nGenerating Top-{num_recommendations} Recommendations for User {user_id}...")
    user_row_number = user_id - 1 
    sorted_user_predictions = preds_df.iloc[user_row_number].sort_values(ascending=False)
    user_data = original_ratings_df[original_ratings_df.user_id == user_id]
    
    recommendations = (movies_df[~movies_df['movie_id'].isin(user_data['movie_id'])]
                       .merge(pd.DataFrame(sorted_user_predictions).reset_index(), on='movie_id')
                       .rename(columns={user_row_number: 'Predicted_Rating'})
                       .sort_values('Predicted_Rating', ascending=False)
                       .iloc[:num_recommendations, :-1])
    return recommendations

top_10 = recommend_movies(preds_df, user_id=15, movies_df=movies, original_ratings_df=df)
print("\n--- TOP 10 RECOMMENDATIONS ---")
for index, row in top_10.iterrows():
    print(f"- {row['title']}")