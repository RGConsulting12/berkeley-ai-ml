# Funk SVD: Step-by-Step Guide (Novice-Friendly)

This guide walks through **Funk SVD** (also called *matrix factorization with gradient descent*) as in Colab Activity 19.3, then shows real-world uses and a mental recap.

---

## Part 1: What Problem Does Funk SVD Solve?

**The situation:** You have a **ratings matrix**: rows = users, columns = items (songs, movies, products), and many cells are **empty** (users didn’t rate those items).

**The goal:** Fill in those empty cells with **predicted ratings** so you can recommend items a user is likely to enjoy.

**The idea:** Assume each rating can be approximated by a small number of hidden “factors.” Funk SVD learns:
- **User factors (P):** one short vector per user (how much they care about each factor).
- **Item factors (Q):** one short vector per item (how much the item has of each factor).

Predicted rating for user \(a\) on item \(j\) = dot product of user \(a\)’s row of **P** with item \(j\)’s column of **Q** (or row of **Q.T**).  
So: **Prediction = P × Q^T** (matrix multiply).

---

## Part 2: The Two Main Matrices (P and Q)

From the notebook:

| Matrix | Shape | Meaning |
|--------|--------|--------|
| **P** (user factors) | users × factors (e.g. 5×2) | Each row = one user’s “profile” in latent space (e.g. F1, F2). |
| **Q** (item factors) | items × factors (e.g. 5×2) | Each row = one item’s “profile” in the same latent space. |

- **Factors (F1, F2, …)** are not given names; the algorithm learns numbers that minimize prediction error. They might loosely correspond to things like “mainstream vs. niche” or “genre,” but we don’t label them.
- **Prediction for user \(a\) on item \(j\):**  
  \(\hat{r}_{a,j} = \sum_k P_{a,k} \, Q_{j,k}\)  
  i.e. the dot product of user \(a\)’s row in P and item \(j\)’s row in Q.

So the notebook’s **Problem 1** is: form the full matrix of predictions **pred_df = P @ Q.T** (rows = users, columns = items).

---

## Part 3: Step-by-Step Through the Notebook

### Step 1: Load data and understand the tables

- **reviews:** real ratings (users × items); many NaNs (no rating).
- **P:** user factor matrix (users × F1, F2).
- **Q:** item factor matrix (items × F1, F2).  
  In code you use **Q.T** when thinking “columns = items”; for **P @ Q.T** you use `Q.T.values` so that (user row) · (item column) gives one predicted rating.

**Mental check:**  
- One row of P × one column of Q.T = one number = predicted rating for that user–item pair.

---

### Step 2: Problem 1 — Make predictions for everyone

**Task:** Fill a matrix where each cell is the predicted rating for that user–item pair.

**Formula:**  
\(\hat{R} = P \times Q^T\)

**Code:**  
`pred_df = pd.DataFrame(P.values @ Q.T.values, index=P.index, columns=Q.index)`

**Why this works:**  
- (P @ Q.T)[a, j] = dot product of user \(a\)’s row of P and item \(j\)’s row of Q = predicted rating for (user \(a\), item \(j\)).

You now have a **dense** matrix of predictions (no NaNs), even though the original ratings were sparse.

---

### Step 3: Problem 2 — Measure error for one prediction

**Task:** Look at one cell: Mandy’s prediction for “Clint Black.”

- **Actual rating (Mandy, Clint Black):** 9  
- **Predicted:** e.g. ~9.46 (from pred_df)

The notebook asks for the prediction there (stored as `ans2`). For **error** we usually use:
- **Error:** \(e = \text{predicted} - \text{actual}\) (or the other way around, depending on how the update rule is written).
- **Squared error:** \((e)^2\).

So “measuring error” here means: take that one prediction (and optionally subtract actual to get error, then square it if we want squared error). The notebook uses the prediction value for `ans2`; the *concept* is “we will improve by reducing how far predictions are from actuals.”

---

### Step 4: Problem 3 — Error for all of Mandy’s rated items

**Task:** For Mandy, only consider items she actually rated: Clint Black, Anti-Cimex, Cardi B.  
For each, compute **squared error**: (actual − predicted)².

**Code idea:**
- Get `actual` = reviews.loc["Mandy", items]
- Get `pred` = pred_df.loc["Mandy", items]
- `ans3 = (actual - pred) ** 2`

So you get one squared error per rated item. **Total loss** for Mandy could be the sum of these (and we’d sum over all users and their rated items for the full loss).

---

### Step 5: Problem 4 — Update one user factor with gradient descent

**Task:** Improve Mandy’s first factor (F1) using one step of gradient descent.

**Update rule (from the notebook):**
\[
P_{a,b} := P_{a,b} - \alpha \sum_{j \in R_a} e_{a,j} Q_{b,j}
\]
where:
- \(R_a\) = set of items that user \(a\) (Mandy) rated.
- \(e_{a,j}\) = error for user \(a\), item \(j\) (often **predicted − actual** so that we *reduce* P when we over-predicted).
- \(Q_{b,j}\) = factor \(b\) for item \(j\) (here \(b=0\) for F1; \(j\) runs over Clint Black, Anti-Cimex, Cardi B).
- \(\alpha\) = learning rate (e.g. 0.1).

**Concrete for Mandy’s F1 (\(P_{1,0}\)):**
- Current value: \(-9.019710\).
- \(e_{1,1} Q_{0,1} + e_{1,3} Q_{0,3} + e_{1,4} Q_{0,4}\) for items Clint Black (1), Anti-Cimex (3), Cardi B (4).
- Using \(e = \text{pred} - \text{actual}\) and \(\alpha = 0.1\) gives the notebook’s result: **P_new ≈ -8.799**.

**Intuition:**  
- If we over-predicted (e > 0), we subtract a positive amount from P (maybe reduce that factor).  
- If we under-predicted (e < 0), we add to P.  
- The same idea is used to update **Q** (item factors); then we repeat over many iterations until error is small.

---

## Part 4: End-to-End Process (What the Algorithm Does)

1. **Initialize** P and Q (e.g. small random numbers).
2. **Predict:** \(\hat{R} = P \times Q^T\).
3. **Compute errors** \(e_{a,j}\) only where we have real ratings (e.g. \(e = \hat{r} - r\)).
4. **Update P:** for each user \(a\) and factor \(b\),  
   \(P_{a,b} := P_{a,b} - \alpha \sum_{j \in R_a} e_{a,j} Q_{b,j}\).
5. **Update Q:** analogous rule for each item and factor (using user errors and P).
6. **Repeat** 2–5 for many epochs (and optionally over random subsets of ratings — SGD).

Result: P and Q get better so that **P @ Q.T** is close to the known ratings and generalizes to unrated user–item pairs.

---

## Part 5: Four Real-World Scenarios to Implement Funk SVD

### 1. **Streaming music (e.g. “songs you might like”)**
- **Users:** listeners; **Items:** songs or albums.
- **Ratings:** explicit (1–5 stars) or implicit (skip vs. play to end, repeat plays).
- **Use:** Predict affinity for unplayed songs; recommend top-K by predicted score.  
- **Implementation:** Build user–item matrix from play/rating logs; run Funk SVD; recommend items with highest \(\hat{r}_{u,j}\) that the user hasn’t consumed.

### 2. **E-commerce (“customers who bought this also…”)**
- **Users:** shoppers; **Items:** products.
- **Ratings:** binary (bought/not) or strength (quantity, repeat purchase).
- **Use:** Predict likelihood to buy; recommend products with highest predicted score.  
- **Implementation:** User–item matrix from purchase history; Funk SVD to get P and Q; for a user, rank items by \(\hat{r}\) and filter to in-stock, not-yet-purchased.

### 3. **Job boards (“jobs matching your profile”)**
- **Users:** job seekers; **Items:** job postings.
- **Ratings:** implicit (clicks, applications, time on listing) or explicit (saved, “not interested”).
- **Use:** Predict interest in unviewed jobs; show “recommended for you.”  
- **Implementation:** User–job matrix from clicks/applications; Funk SVD; recommend jobs with highest predicted score that the user hasn’t applied to.

### 4. **Online courses / learning platforms**
- **Users:** learners; **Items:** courses, videos, or skills.
- **Ratings:** completion, quiz score, or time spent.
- **Use:** Recommend next course or module.  
- **Implementation:** User–course matrix from completions/scores; Funk SVD; recommend courses with highest predicted score that the user hasn’t taken.

In all four, the **same steps** apply: form a user–item matrix (possibly with derived “ratings”), run Funk SVD (or a library like Surprise that implements it), then use **P @ Q.T** to score and rank items for each user.

---

## Part 6: Mental Recap — How to Work Through a Funk SVD Problem

1. **Identify users and items**  
   Who are the “rows” and what are the “columns”? What does a “rating” mean (explicit or implicit)?

2. **Build the ratings matrix**  
   Fill in known values; leave unknown as missing (not used in loss, or masked).

3. **Choose latent dimension**  
   Number of factors (columns of P and Q). Start small (e.g. 10–50); tune later.

4. **Predict**  
   \(\hat{R} = P \times Q^T\). Only the **structure** matters (user vector · item vector); you don’t need to name the factors.

5. **Define error**  
   \(e_{a,j} = \hat{r}_{a,j} - r_{a,j}\) (or the opposite) only where \(r_{a,j}\) is observed. Loss = sum of \(e^2\) (or similar) over observed pairs.

6. **Update P and Q**  
   Gradient descent: move each factor in the direction that reduces error (notebook’s P update; Q update is symmetric). Use a learning rate \(\alpha\) and many iterations.

7. **Recommend**  
   For each user, sort items by \(\hat{r}_{u,j}\), filter (e.g. already consumed, ineligible), take top-K.

8. **Evaluate**  
   Compare predicted vs. actual on held-out ratings (RMSE, MAE) or use ranking metrics (e.g. NDCG, hit rate) for recommendations.

Keeping this sequence in mind (data → P,Q → predict → error → update → recommend → evaluate) is enough to implement and adapt Funk SVD in practice.
