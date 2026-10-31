# Lab 8 Theory — K-Means Clustering

## 1. Unsupervised Learning

So far (Labs 1–6, 10) the target was known — **supervised** learning. Here there are **no labels**: the goal is to discover structure. Iris has true species available, but the algorithm never uses them — they're only revealed *after* clustering to interpret results.

**Clustering** = partitioning samples into groups where intra-cluster similarity is high and inter-cluster similarity is low.

## 2. The K-Means Algorithm

Inputs: data X, number of clusters **k**. Iterate until convergence:

1. **Initialize** — pick k centroids (sklearn default: `k-means++`, which seeds centroids far apart to avoid bad local minima).
2. **Assignment step** — assign each point to its nearest centroid, usually by squared Euclidean distance:

   \( d(x, c) = \sum_{i=1}^{m}(x_i - c_i)^2 \)

3. **Update step** — recompute each centroid as the **mean of its assigned points**.
4. Repeat 2–3 until assignments stop changing (or max iterations reached).

Objective (minimized implicitly):

\( J = \sum_{j=1}^{k} \sum_{x \in C_j} \|x - \mu_j\|^2 \quad \text{(WCSS / inertia)} \)

## 3. Choosing k

K-means needs k in advance. Two standard methods (both used in the lab):

### Elbow Method
Run k-means for k = 1…K_max and plot **inertia (WCSS)**. Inertia always decreases as k grows; the "elbow" — where the drop suddenly flattens — marks diminishing returns. For Iris the elbow is near **k = 3**.

### Silhouette Method
For each point with its cluster a and nearest other cluster b:

\( s(i) = \dfrac{b - a}{\max(a, b)}, \in [-1, 1] \)

- s ≈ 1 → well inside its cluster; s ≈ 0 → on a boundary; s < 0 → probably misassigned.
- Average silhouette per k; **choose k maximizing it**. Iris peaks at k = 2 or 3.

## 4. Clustering Evaluation (no labels)

| Metric | Idea | Best value |
|--------|------|-----------|
| **Inertia** | within-cluster variance | lower (but always falls with k → elbow) |
| **Silhouette score** | separation vs cohesion per point | ↑ toward 1 |
| **Davies-Bouldin index** | average cluster-pair similarity (spread/distance) | ↓ toward 0 |

If ground truth exists (Iris), a **cluster-vs-species table** interprets the result: with k = 3, K-means on Iris typically matches *setosa* perfectly and splits *versicolor/virginica* partially.

## 5. Why Scale First?

K-means works on Euclidean distance, so a feature with a large range dominates every cluster decision. **StandardScaler** (z-scores) puts features on equal footing before clustering.

## 6. PCA for Visualization

Clusters live in 4-D (Iris) — impossible to plot directly. **Principal Component Analysis (PCA)** finds orthogonal directions of maximum variance and projects points onto the top-2:

- The 2 components are linear combinations of the original features (PC1 usually captures most variance).
- PCA here is **visualization only** — clustering itself runs on all 4 (scaled) features.
- The lab plots points colored by predicted cluster with **centroids** projected into the same space (centroid markers via `pca.transform(cluster_centers_)`).

## 7. Limitations of K-Means

- Must fix k in advance.
- Assumes roughly **spherical, equal-size** clusters — fails on elongated/irregular shapes.
- Sensitive to initialization (mitigated by k-means++) and to outliers (centroids are means).
- May converge to a local optimum → results vary with `random_state`.

## 8. Workflow Recap

Load → scale → sweep k with Elbow (inertia) and Silhouette → fit final KMeans with optimal k → evaluate (silhouette, Davies-Bouldin, cluster means) → PCA 2-D scatter with centroids → compare clusters against the true species.
