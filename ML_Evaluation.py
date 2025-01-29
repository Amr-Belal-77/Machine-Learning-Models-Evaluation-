import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, silhouette_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import seaborn as sns
from IPython.display import HTML

# Function to preprocess dataset
def preprocess_data(data):
    # Handling missing values
    data = data.dropna()
    # Encoding categorical variables
    data = pd.get_dummies(data, drop_first=True)
    return data

# Function to evaluate supervised models
def evaluate_supervised(X, y):
    models = {
        "Random Forest": RandomForestClassifier(),
        "Gradient Boosting": GradientBoostingClassifier(),
        "SVM": SVC(),
        "Logistic Regression": LogisticRegression(),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier()
    }
    results = {}
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = accuracy
    return results

# Function to evaluate unsupervised models
def evaluate_unsupervised(X):
    models = {
        "KMeans": KMeans(n_clusters=3),
        "Agglomerative Clustering": AgglomerativeClustering(n_clusters=3),
        "DBSCAN": DBSCAN()
    }
    results = {}
    for name, model in models.items():
        if name == "KMeans":
            model.fit(X)
            score = silhouette_score(X, model.labels_)
        elif name == "DBSCAN":
            labels = model.fit_predict(X)
            if len(set(labels)) > 1:  # Ensure more than one cluster
                score = silhouette_score(X, labels)
            else:
                score = -1  # Invalid score for single cluster
        else:
            labels = model.fit_predict(X)
            score = silhouette_score(X, labels)
        results[name] = score
    return results

# Function to animate 3D hyperboloid shape
def animate_plot(i, ax):
    ax.clear()

    X, Y = np.meshgrid(np.linspace(-3, 3, 40), np.linspace(-3, 3, 40))
    Z = X*2 - Y*2  # Hyperboloid shape

    mask = np.zeros_like(Z, dtype=bool)
    mask[:i, :i] = True  # Reveal points in a growing square

    colors = np.linspace(0, 1, mask.sum())  # Smooth gradient
    np.random.shuffle(colors)

    ax.scatter(
        X[mask],
        Y[mask],
        Z[mask],
        c=colors,
        cmap='coolwarm',  # Color map
        marker='o'
    )

    ax.set_title('Animated 3D Hyperboloid with Changing Colors')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_zlim(-10, 10)

# Function to visualize results in 3D and animate them
def animate_results_3d(results, task_type):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(121, projection='3d')  # 3D plot on the left

    colors = ['gray'] * len(results)  # Default color for models
    best_model = max(results, key=results.get)
    colors[list(results.keys()).index(best_model)] = 'r'  # Highlight best model in red

    x = np.arange(len(results))  # X-axis: index of algorithms
    y = list(results.values())   # Y-axis: score values (accuracy/silhouette score)
    z = np.zeros_like(x)         # Z-axis: constant for 2D view

    scatter = ax.scatter(x, y, z, color=colors)
    ax.set_xlabel('Models')
    ax.set_ylabel('Score')
    ax.set_zlabel('Z (constant for 2D view)')
    ax.set_xticks(x)
    ax.set_xticklabels(list(results.keys()), rotation=45)
    ax.set_title(f"3D Animation of {task_type.capitalize()} Models")

    def update(frame):
        current_colors = ['gray'] * len(results)
        if frame == len(results):
            current_colors[list(results.keys()).index(best_model)] = 'g'  # Highlight best model at the end
        else:
            current_colors[frame] = 'b'  # Color model blue during animation
        scatter.set_facecolors(current_colors)
        return scatter,

    ani = FuncAnimation(fig, update, frames=len(results), interval=1000, blit=False)

    ax2 = fig.add_subplot(122)  # Heatmap on the right
    data_matrix = np.array(list(results.values())).reshape(1, -1)
    sns.heatmap(data_matrix, annot=True, cmap="YlGnBu", xticklabels=results.keys(), yticklabels=['Score'], ax=ax2)
    ax2.set_title("Model Performance Heatmap")

    plt.tight_layout()
    plt.show()

# Main script
url = input(f'Enter the URL for data : ')
task_type = input(f'Chose Data type :\n1-Supervised.\n2-Unsupervised.\n')

# Load the dataset
data = pd.read_csv(url)
data = preprocess_data(data)

if task_type == "supervised" or task_type == '1':
    target_column = "country"  # Adjust target column
    X = data.drop(columns=[target_column])
    y = data[target_column]
    results = evaluate_supervised(X, y)

elif task_type == "unsupervised" or task_type == '2':
    X = data
    results = evaluate_unsupervised(X)

else:
    raise ValueError("Invalid task type. Please specify 'supervised' or 'unsupervised'.")

# Output results
print("\nModel Evaluation Results:")
for model, score in results.items():
    print(f"{model}: {score:.4f}")

best_model = max(results, key=results.get)
print(f"\nThe best model is {best_model} with a score of {results[best_model]:.4f}.")

# Visualizing the evaluation results in 3D and animate it, and add heatmap
animate_results_3d(results, task_type)