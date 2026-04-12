# AirDeath — Logistic Regression from Scratch

> Day 1 of 90 | Project 2 | Binary classifier. No sklearn models. Pure numpy.

---

## The Problem

India has 800+ air quality monitoring stations running 24/7. CPCB publishes
the data. Nobody acts on it fast enough.

A severe AQI day (> 300) means schools should close, construction should
stop, vulnerable people should stay indoors. But that decision needs to be
made *before* the day gets bad — not after.

This project answers one binary question from pollutant readings:
**is this going to be a severe air quality day or not?**

---

## The Goal

Build a logistic regression classifier from scratch in numpy that takes
real pollutant readings and outputs a 0 or 1 — safe day vs severe day.

Then go deeper:
- which pollutant does the model think matters most?
- where does the model get it wrong, and why?
- what happens when you change the decision threshold from 0.5 to 0.3?

---

## The Dataset

**Source:** IndiaAI / CPCB — Real Time Air Quality Index from various locations  
**Link:** https://aikosh.indiaai.gov.in/home/datasets/details/real_time_air_quality_index_from_various_locations.html  
**License:** Open Government License, India  
**Frequency:** Hourly, Real-Time  
**Coverage:** All major Indian cities, 800+ monitoring stations

| Column | Description |
|---|---|
| country | always India |
| state | state name |
| city | city name |
| station | monitoring station name |
| last_update | timestamp of reading |
| latitude / longitude | station coordinates |
| pollutant_id | which pollutant (PM2.5, PM10, NO2, SO2, CO, O3, NH3) |
| pollutant_min | minimum reading in the hour |
| pollutant_max | maximum reading in the hour |
| pollutant_avg | average reading — this is the main feature |

**The target is derived** — if the computed AQI from pollutant readings
crosses 300, label = 1 (severe). Below 300, label = 0 (safe).

---

## The Approach

**Step 1 — Reshape**  
The dataset is in long format — one row per pollutant per station per hour.
Pivot it so each row is one station-hour with all pollutants as columns.

**Step 2 — Label**  
Compute AQI from PM2.5 and PM10 readings. Threshold at 300 to create
binary target. Check class balance — severe days are rare, expect imbalance.

**Step 3 — Build the classifier**  
Three functions from scratch:
- sigmoid — squashes any number into 0-1 probability
- binary cross-entropy loss — measures how wrong the probability is
- backward pass — gradient of loss w.r.t. each weight, derived by hand

**Step 4 — Train**  
Same gradient descent loop as day 1. Watch loss drop. Hit > 90% accuracy.

**Step 5 — Go deeper**  
- confusion matrix (TP, TN, FP, FN) written manually — no sklearn
- which pollutant has the highest absolute weight? that's what the model thinks matters most
- tune the threshold: lower it from 0.5 → 0.3, watch false negatives drop
- break down accuracy by city — find where the model fails and why

---

## Results

| Metric | Value |
|---|---|
| Accuracy | `[fill after run]` |
| False Negatives (missed severe days) | `[fill after run]` |
| Most influential pollutant | `[fill after run]` |
| Best threshold | `[fill after run]` |

---

## Key Insight

> *[fill this after you finish — where did the model fail, and what does that tell you about the data?]*

---

## Stack

- Python 3.x
- numpy — model, loss, gradients, training loop
- pandas — data reshaping and cleaning only
- matplotlib — confusion matrix, threshold curve, weight visualization

---

## Run It

```bash
pip install numpy pandas matplotlib
jupyter notebook main.ipynb
```

---

*Part of a 90-day public ML build · [@skg_curious](https://x.com/skg_curious) · [LinkedIn](https://www.linkedin.com/in/shubhamgupta04907/)*