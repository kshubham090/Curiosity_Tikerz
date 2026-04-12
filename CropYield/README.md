# CropYield — Linear Regression from Scratch

> Day 1 of 90 | No sklearn models. Pure numpy. Gradient descent written by hand.

---

## The Problem

India's agricultural output varies wildly across states, crops, and seasons. A district officer sitting in Punjab has historical records — what crop was grown, how much land was used, which season — but no reliable way to estimate expected yield before the harvest. Right now that estimate is gut feeling and experience.

That gut feeling is what this project replaces.

---

## The Goal

Given historical farming records, build a system that predicts **crop yield in Tonnes per Hectare** for any combination of state, crop, season, and cultivated area.

One number in. One number out. How much will this crop yield?

---

## The Dataset

**Source:** Kaggle — India Crop Production Statistics

| Column | Type | Description |
|---|---|---|
| State | categorical | Indian state name |
| District | categorical | District within state (dropped — too granular) |
| Crop | categorical | Crop variety (50+ unique crops) |
| Year | string | Crop year e.g. `2001-02` |
| Season | categorical | Kharif / Rabi / Whole Year / Zaid etc. |
| Area | numerical | Cultivated area in Hectares |
| Area Units | string | Always "Hectare" (dropped) |
| Production | numerical | Total production in Tonnes |
| Production Units | string | Always "Tonnes" (dropped) |
| **Yield** | **numerical** | **Target — Tonnes per Hectare** |

Yield is already computed in the dataset as `Production / Area`. This is what the model learns to predict.

---

## Why Linear Regression

The relationship between cultivated area, season, and yield has a roughly linear signal — more area doesn't always mean more yield, but patterns exist across seasons and crop types that a weighted line can approximate.

More importantly: linear regression is the foundation. Before you touch neural networks, you need to feel gradient descent working on a real problem with real messy data. No abstraction. No `.fit()`. Just math running in a loop.

---

## The Approach

### 1. Clean
- Drop rows where Yield is null or zero (bad records)
- Drop District and unit columns
- Parse Year into a usable integer
- Label encode State, Crop, Season (convert categories to numbers)
- Normalize Area and Year to 0–1 range manually

### 2. Build the Engine (pure numpy)
Three functions written from scratch:

- **Forward pass** — multiply inputs by weights, add bias, output a prediction
- **Loss function** — Mean Squared Error between prediction and actual yield
- **Gradient computation** — how much to adjust each weight to reduce loss

One training loop that runs these three functions thousands of times, nudging weights slightly each iteration. That nudge is the learning rate. That loop is gradient descent.

### 3. Experiment
- Run training at normal learning rate — watch loss drop
- Run at learning rate 10x too high — watch loss explode
- Run at learning rate 100x too low — watch it barely move

These three runs teach you more about optimization than any lecture.

### 4. Visualize
- **Loss curve** — MSE vs epochs. Should drop fast then flatten.
- **Predicted vs Actual scatter** — x-axis is real yield, y-axis is predicted. A perfect model is a diagonal line. See how far off yours is.
- **Residuals by Season** — where does the model get it most wrong? This tells you where linear regression hits its limit.

---

## Results

| Metric | Value |
|---|---|
| Final Training MSE | `[fill after run]` |
| Final Training RMSE | `[fill after run]` |
| Learning Rate Used | `[fill after run]` |
| Epochs | `[fill after run]` |
| Crops model struggled most on | `[fill after run]` |
| Season with highest residuals | `[fill after run]` |

---

## What the Model Gets Wrong (and Why That's the Point)

Linear regression draws a straight line through data. Crop yield doesn't follow a straight line — it depends on rainfall, soil quality, regional farming practices, and dozens of other things not in this dataset.

When you plot residuals grouped by crop type, you'll see certain crops where the model is consistently far off. That's not a bug. That's the model showing you it needs more complexity — which is exactly why decision trees exist, and why you're building one tomorrow.

---

## What I Learned

> *[fill this after you finish — one paragraph, honest, what actually clicked and what broke]*

---

## Stack

- Python 3.x
- numpy — model, loss, gradients, training loop
- pandas — data loading and cleaning only
- matplotlib — visualization only
- sklearn — `LabelEncoder` for categorical encoding only (lookup table, not a model)

---

## Run It

```bash
pip install numpy pandas matplotlib scikit-learn
jupyter notebook cropyield.ipynb
```

---

*Part of a 90-day public ML build. Follow along on X.*