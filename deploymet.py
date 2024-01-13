import pandas as pd
import numpy as np
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity



# Assuming 'final_ratings.csv' is loaded somewhere in the code

final_ratings = pd.read_csv(r"C:\Users\Anil\Book recomendation project\final_ratings.csv")

pt=final_ratings.pivot_table(index='Book-Title',columns='User-ID',values='Book-Rating')
pt.fillna(0, inplace=True)

st.title('User-Based Book Recommendation System')

# Function to recommend similar books
def recommend(book_name):
    index = np.where(pt.index == book_name)[0][0]

    similarity_scores = cosine_similarity(pt)
    similar_items = sorted(list(enumerate(similarity_scores[index])), key=lambda x: x[1], reverse=True)[1:6]

    recommended_books = [pt.index[i[0]] for i in similar_items]
    return recommended_books

# Streamlit app
def main():
    st.title("Recommand Top 5 Books")

    user_input = st.text_input("Enter User ID or Book Title:")

    if st.button("Enter"):
        try:
            user_input = int(user_input)
            if user_input in pt.columns:
                user_ratings = pt[user_input].dropna()
                top_rated_books = user_ratings.sort_values(ascending=False).head(5)

                st.write(f"Top 5 rated books for User {user_input}:")
                st.write(top_rated_books)
            else:
                st.write("Invalid User ID. Please enter a valid User ID.")
        except ValueError:
            if user_input in pt.index:
                recommended_books = recommend(user_input)
                st.write(f"Books similar to '{user_input}':")
                st.write(recommended_books)
            else:
                st.write(f"Book '{user_input}' not found in the dataset.")

    st.title("Recommand Books")

    book_input = st.text_input("Enter Book Name:")

    if st.button("Recommend"):
        if book_input in pt.index:
            recommended_books = recommend(book_input)
            st.write(f"Books similar to '{book_input}':")
            st.write(recommended_books)
        else:
            st.write(f"Book '{book_input}' not found in the dataset.")

if __name__ == "__main__":
    main()
