from flask import Flask, request, render_template
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import plotly.graph_objs as go
import plotly.offline as pyo
import numpy as np

app = Flask(__name__)

# Load the 20 Newsgroups dataset
newsgroups = fetch_20newsgroups(subset='all')
documents = newsgroups.data

# Prepare TF-IDF matrix
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X = vectorizer.fit_transform(documents)

# Apply LSA using SVD (reduce to 100 components)
svd = TruncatedSVD(n_components=100)
X_reduced = svd.fit_transform(X)

@app.route("/", methods=["GET", "POST"])
def index():
    graph = None
    results = []  # Default empty list for initial load

    if request.method == "POST" and request.form.get("query"):
        query = request.form["query"]
        results, similarities = process_query(query)

        # Generate Plotly graph only if there are results
        graph = create_similarity_plot(results, similarities)

    return render_template("index.html", graph=graph, results=results)

def process_query(query):
    """Process the user query and find top matching documents."""
    query_vec = vectorizer.transform([query])
    query_reduced = svd.transform(query_vec)

    # Calculate cosine similarities
    similarities = cosine_similarity(query_reduced, X_reduced)[0]

    # Get top 5 documents by similarity
    top_indices = np.argsort(similarities)[::-1][:5]
    results = [(f"Document {i}", documents[i], similarities[i]) for i in top_indices]
    return results, similarities[top_indices]

def create_similarity_plot(results, similarities):
    """Generate an interactive Plotly bar chart."""
    doc_ids = [result[0] for result in results]
    sim_values = similarities

    bars = go.Bar(
        x=doc_ids,
        y=sim_values,
        text=[f"{result[1][:100]}..." for result in results],  # Truncated content for tooltips
        hoverinfo="text+y",
        marker=dict(color='lightblue')
    )

    layout = go.Layout(
        title="Cosine Similarity of Top 5 Documents",
        xaxis=dict(title="Document ID"),
        yaxis=dict(title="Cosine Similarity")
    )

    fig = go.Figure(data=[bars], layout=layout)
    graph = pyo.plot(fig, output_type="div")

    return graph

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000)
