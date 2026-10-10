# 📚 Book Recommender System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Notebook-Jupyter-orange?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/pandas-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/numpy-013243?logo=numpy&logoColor=white)](https://numpy.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An intelligent, item-based collaborative filtering book recommendation engine built using **Unsupervised Nearest Neighbors (k-NN)** and **Cosine Similarity**. The system analyzes historical user reading patterns and ratings to surface relevant, highly correlated book suggestions for any given title.

---

## 📌 Table of Contents

- [Project Overview](#-project-overview)
- [How It Works](#-how-it-works)
- [Dataset Architecture](#-dataset-architecture)
- [Data Pipeline & Filtering](#-data-pipeline--filtering)
- [Recommendation Algorithm](#-recommendation-algorithm)
- [Sample Recommendations](#-sample-recommendations)
- [Project Structure](#-project-structure)
- [Installation & Quickstart](#-installation--quickstart)
- [Dataset Notice](#-dataset-notice-repository-size-optimization)
- [Future Enhancements](#-future-enhancements)
- [License](#-license)

---

## 📖 Project Overview

Finding the next book to read among hundreds of thousands of titles can be challenging. This project builds an **Item-Based Collaborative Filtering** system using the popular **Book-Crossing** dataset.

Instead of relying solely on genre tags or metadata, this system examines collective reader behavior: **if readers who rated Book A also rated Book B favorably, then Book B is recommended to readers of Book A**.

### Key Highlights:
- **Noise Reduction**: Filters out inactive users and unrated/infrequently rated books to combat extreme matrix sparsity.
- **Memory Efficiency**: Converts high-dimensional user-item interaction matrices into **Compressed Sparse Row (`csr_matrix`)** format.
- **Fast Similarity Search**: Employs `NearestNeighbors(algorithm='brute')` to compute cosine distances across book rating vectors.

---

## ⚙️ How It Works

```
Raw CSV Files (Books, Users, Ratings)
       │
       ▼
Data Cleaning & Column Renaming
       │
       ▼
Filter Avid Readers (Users with > 200 ratings)
       │
       ▼
Filter Popular Books (Books with ≥ 50 ratings)
       │
       ▼
Deduplicate User-Book Pairs
       │
       ▼
Create Pivot Table (742 Books × 888 Users)
       │
       ▼
Convert to CSR Matrix (Compressed Sparse Row)
       │
       ▼
Train Unsupervised NearestNeighbors Model
       │
       ▼
Query Book ──► Top 5 Nearest Neighbor Recommendations
```

---

## 📊 Dataset Architecture

The project utilizes the classic **Book-Crossing Dataset** comprising three relational CSV files:

> 📥 **Dataset Download**: Download the dataset from [Kaggle: Book Recommendation Dataset](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset?select=Books.csv) and place the three CSV files in the project root directory.

| File | Records | Key Columns Used | Description |
| :--- | :--- | :--- | :--- |
| **`Books.csv`** | 271,360 | `ISBN`, `title`, `author`, `year`, `publisher` | Catalog containing book metadata and publication information |
| **`Users.csv`** | 278,858 | `user_id`, `location`, `age` | Anonymized user demographic profiles |
| **`Ratings.csv`** | 1,149,780 | `user_id`, `ISBN`, `rating` | Explicit ratings (0–10 scale) given by users |

---

## 🧹 Data Pipeline & Filtering

In recommendation engines, high sparsity (over 99% empty matrix entries) leads to the **cold-start problem** and skewed recommendations. The pipeline applies targeted thresholds to extract statistically sound patterns:

1. **Active Reader Filtering**:
   - Only users who have rated **more than 200 books** are retained.
   - Reduces ratings from **1,149,780** down to **526,356** high-confidence reviews across **888 avid readers**.

2. **Popular Book Filtering**:
   - Filtered ratings are merged with book records (**487,671** matches).
   - Only books with **at least 50 ratings** from active readers are kept.
   - Isolates **61,853** ratings on recognized titles.

3. **Deduplication & Pivot Table**:
   - Duplicate user ratings for identical titles are eliminated, leaving **59,850 clean records**.
   - A pivot table is generated with:
     - **Index**: Book Title (742 distinct books)
     - **Columns**: User ID (888 distinct users)
     - **Values**: Rating score (0–10, missing entries filled with `0`)

4. **Sparse Matrix Representation**:
   - The dense $742 \times 888$ pivot table is converted to `scipy.sparse.csr_matrix` to save memory and accelerate distance computations.

---

## 🤖 Recommendation Algorithm

The system uses **k-Nearest Neighbors (k-NN)** for unsupervised distance metric learning:

```python
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

# Convert pivot table to CSR sparse matrix
book_sparse = csr_matrix(book_pivot)

# Initialize and fit Nearest Neighbors model
model = NearestNeighbors(algorithm='brute')
model.fit(book_sparse)
```

### Recommendation Logic:
Given a target book title, the algorithm looks up its vector index in the matrix, extracts its rating distribution across all users, and queries the $k=6$ closest vectors (1 target book + 5 nearest neighbors) using cosine distance:

```python
def recommend_book(book_name):
    book_id = np.where(book_pivot.index == book_name)[0][0]
    distances, suggestions = model.kneighbors(
        book_pivot.iloc[book_id].values.reshape(1, -1), 
        n_neighbors=6
    )
    print(f"The suggestions for {book_name} are :")
    for i in range(len(suggestions[0])):
        if i != 0:
            print(f"{i}. {book_pivot.index[suggestions[0][i]]}")
```

---

## 🎯 Sample Recommendations

Here are examples of recommendations produced by the model in [`main.ipynb`](main.ipynb):

### 🔮 Example 1: *Harry Potter and the Chamber of Secrets (Book 2)*
```text
The suggestions for Harry Potter and the Chamber of Secrets (Book 2) are :
1. Harry Potter and the Goblet of Fire (Book 4)
2. Harry Potter and the Prisoner of Azkaban (Book 3)
3. Harry Potter and the Sorcerer's Stone (Book 1)
4. Exclusive
5. The Cradle Will Fall
```
> **Observation**: The algorithm effectively clusters related fantasy titles within the same series.

---

### 📖 Example 2: *Animal Farm*
```text
The suggestions for Animal Farm are :
1. Exclusive
2. Jacob Have I Loved
3. Second Nature
4. Pleading Guilty
5. No Safe Place
```

---

## 📁 Project Structure

```bash
Book_Recommender_System/
│
├── main.ipynb          # End-to-end Jupyter Notebook (EDA, Preprocessing, Modeling, Testing)
├── requirements.txt    # Project dependencies
├── README.md           # Project documentation
├── .gitignore          # Git ignore rules (ignores .csv, venv, checkpoints)
│
├── Books.csv           # [Local only] Book catalog (download from Kaggle)
├── Ratings.csv         # [Local only] Book ratings (download from Kaggle)
└── Users.csv           # [Local only] User profiles (download from Kaggle)
```

---

## 🚀 Installation & Quickstart

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/Book_Recommender_System.git
cd Book_Recommender_System
```

### 2. Create and Activate a Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Required Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download the Dataset
Download the [Book-Crossing Dataset from Kaggle](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset?select=Books.csv) and place the three CSV files in the project root:
- `Books.csv`
- `Ratings.csv`
- `Users.csv`

### 5. Run the Jupyter Notebook
```bash
jupyter notebook main.ipynb
```
Step through the notebook cells to reproduce data cleaning, sparse matrix transformation, and book recommendations.

---

## 💾 Dataset Notice (Repository Size Optimization)

The dataset files (`Books.csv` ~73 MB, `Ratings.csv` ~22 MB, `Users.csv` ~11 MB) are intentionally **excluded from Git tracking** via [`.gitignore`](.gitignore) to:
- Keep repository size ultra-lightweight (< 1 MB).
- Prevent GitHub's 50 MB / 100 MB large file warning and permanent `.git` history bloat.
- Follow data science best practices of decoupling code from large static datasets.

> Anyone looking to reproduce the notebook can grab the files directly from the [Kaggle Dataset Page](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset?select=Books.csv).

---

## 🔮 Future Enhancements

- [ ] **Interactive Web UI**: Build a frontend using [Streamlit](https://streamlit.io/) or Flask.
- [ ] **Model Serialization**: Export `model` and `book_pivot` as `.pkl` artifacts using `pickle` for instant inference.
- [ ] **Poster / Cover Display**: Enrich recommendations by rendering book cover images via image URLs.
- [ ] **Hybrid Recommender**: Combine collaborative filtering with content-based NLP filtering (analyzing book summaries and author styles).

---

## 📄 License

This project is licensed under the [MIT License](LICENSE). Feel free to use, modify, and distribute with attribution.
