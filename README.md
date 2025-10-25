# ART-1 Spam Detector Visualizer

An interactive web application built with Flask to visualize and understand the ART-1 (Adaptive Resonance Theory) neural network. This project demonstrates how ART-1 performs binary pattern clustering by using it to classify spam and legitimate (ham) emails.

This application is designed as an educational tool to visually explore the effect of the **vigilance parameter ($\rho$)** and the nature of "on-the-fly" cluster creation.

## 🚀 Features

* **Interactive Vigilance ($\rho$) Slider:** Dynamically change the ART-1 network's vigilance parameter.
* **Live Re-Clustering:** A "Recompute" button that retrains the entire model from scratch with the new $\rho$.
* **Interactive Cluster Graph:** An SVG graph that visually maps each training email (vector) to its resulting cluster.
    * **Hover Tooltips:** Hover over any email node in the graph to see its original Subject and Body.
* **Feature Extraction Explained:** A clear, visual breakdown of how a raw email is converted into a 40-bit binary vector.
* **Dynamic UI:** A collapsible navigation sidebar, detailed statistics, and organized results that all update after re-clustering.
* **Detailed Results:** See which specific training emails fall into which cluster and get a clear accuracy score based on test data.

## 🤖 What is ART-1 (Adaptive Resonance Theory)?

**ART-1 (Adaptive Resonance Theory 1)** is an unsupervised neural network model developed by Gail Carpenter and Stephen Grossberg. It is designed to solve the "plasticity-stability dilemma," meaning it can remain **plastic** (able to learn new patterns) without becoming **unstable** (forgetting or "catastrophically overwriting" old, learned patterns).

ART-1 is specifically designed to **cluster binary vectors** (patterns of 0s and 1s).

### How It Works (The Gist)

1.  **Input (F1 Layer):** A binary vector (in our case, a 40-bit email vector) is presented to the input layer.
2.  **Choice (F2 Layer):** The input is compared against the "prototypes" of all existing clusters in the F2 layer. The cluster that is *most similar* to the input "wins."
3.  **Vigilance Test ($\rho$):** This is the most critical step. The network asks: "Is this winning cluster's prototype *similar enough* to the input vector?"
    * The "similarity" is calculated and compared to the **vigilance parameter ($\rho$)**, a value between 0 and 1.
4.  **Resonance (Update Cluster):** If the match is good enough (similarity $\ge \rho$), the input vector is added to that cluster. The cluster's prototype is then updated to become a logical AND (the intersection) of the input and the old prototype.
5.  **Adaptation (New Cluster):** If the match is *not* good enough (similarity $< \rho$), the winning cluster is shut down, and the network searches for the *next* best match. If no existing cluster can satisfy the vigilance test, the network **creates a new cluster** and uses the input vector as its first prototype.

This vigilance-and-search process allows ART-1 to create exactly as many clusters as are needed for the given data and vigilance level, making it a powerful and adaptive clustering tool.

---

## 🔧 How It Works: Project Architecture

This project connects the ART-1 theory to a real-world Flask application.

1.  **Frontend (UI):** `templates/index.html` and `static/style.css` create the user interface. It features a sidebar (built with HTML/CSS) and an interactive graph (built with SVG).
2.  **Backend (Server):** `app.py` is a Flask server that manages the logic.
3.  **Feature Extraction:** The `EmailFeatureExtractor` class converts raw email text into a **40-bit binary vector**. It checks for the presence (1) or absence (0) of 28 keywords (e.g., "free," "winner," "meeting," "report") and 12 patterns (e.g., "ALL CAPS SUBJECT," "multiple exclamation marks").
4.  **ART-1 Model:** The `ART1` class is a pure Python implementation of the algorithm described above.
5.  **The Process:**
    * A user visits `http://127.0.0.1:5000`.
    * They move the $\rho$ slider and click "Recompute Clusters."
    * The browser POSTs the new $\rho$ value to the Flask server.
    * The server **re-initializes** the `SpamDetector` with the new $\rho$.
    * The `SpamDetector` trains on all `spam_emails` and `ham_emails`, creating clusters based on the new vigilance.
    * It then runs the `test_emails` to calculate an accuracy score.
    * It generates all the visualization data (graph nodes, edges, cluster contents).
    * The server renders the `index.html` template, injecting all this new data into the page.

---

## 🏁 Getting Started: Installation & Running

Follow these steps to run the project on your local machine.

### Prerequisites

* [Python 3.7+](https://www.python.org/downloads/)
* [Git](https://git-scm.com/downloads)
* `pip` (Python's package installer)

### Installation Steps

1.  **Clone the repository:**
    ```sh
    git clone https://github.com/real-ds/art1-email-spam-detection.git
    cd art1-email-spam-detection
    ```

2.  **Create a virtual environment:**
    ```sh
    # For Windows
    python -m venv .venv
    
    # For macOS/Linux
    python3 -m venv .venv
    ```

3.  **Activate the virtual environment:**
    ```sh
    # For Windows
    .\.venv\Scripts\activate
    
    # For macOS/Linux
    source .venv/bin/activate
    ```

4.  **Install the required packages:**
    (This project has two main dependencies)
    ```sh
    pip install Flask numpy
    ```

5.  **Run the application:**
    ```sh
    python app.py
    ```

6.  **Open your browser:**
    Navigate to **`http://127.0.0.1:5000`** to see the application live!

---

## 🔬 Findings & Analysis: Why is the Accuracy Low?

During testing (as seen in the screenshot), the best-achieved accuracy was around **58.3%**. This is not a bug; it is the most important **finding** of the project.

The low accuracy demonstrates the limitations of this specific model for this task:

1.  **The Feature Problem:** Our `EmailFeatureExtractor` is too simple. It relies on keywords that can appear in *both* spam and ham emails. For example, a legitimate email ("Hi team, feel **free** to review this **investment** report, **click here** to download") can trigger multiple "spam" features, confusing the model. The model cannot understand **context**.

2.  **The Clustering vs. Classification Problem:** ART-1 is a **clustering** algorithm, not a classifier. Its job is to find *similarity*, not a *decision boundary*. Because of the feature problem, many ham and spam vectors "look" very similar, so ART-1 groups them into the same "mixed cluster," leading to incorrect predictions.

3.  **The Vigilance Paradox:**
    * **High $\rho$ (e.g., 0.95):** The model is too strict. It creates dozens of tiny, "pure" clusters. New test emails don't match any of them, resulting in many "unknown" predictions.
    * **Low $\rho$ (e.g., 0.35):** The model is too lenient. It creates a few large, "mixed" clusters (containing both spam and ham), which leads to a high number of incorrect guesses.

**Conclusion:** The 58.3% accuracy is the "sweet spot" where the model is just lenient enough to make guesses, but those guesses are highly flawed due to the simplistic features. This demonstrates that while ART-1 is a powerful clustering tool, it is not well-suited for a semantically complex task like spam detection *without* a much more sophisticated feature extraction model.

## 🛠️ Built With

* **Python:** Core logic.
* **Flask:** Web server and backend.
* **Numpy:** Vector and matrix operations for ART-1.
* **HTML5 / CSS3:** Frontend structure and styling.
* **SVG (Scalable Vector Graphics):** For the dynamic, interactive clustering graph.
* **JavaScript:** For sidebar toggling, slider updates, and graph tooltips.
