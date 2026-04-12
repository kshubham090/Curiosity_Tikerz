# AirDeath — Logistic Regression from Scratch

> Day 1 of 90 | Project 2 | Binary classifier. No sklearn models. Pure numpy.

---

## The Problem

India has 800+ air quality monitoring stations running 24/7. CPCB publishes the data. Nobody acts on it fast enough.

A severe AQI reading means schools should close, construction should stop, vulnerable people should stay indoors. This project looks at a real snapshot of India's air right now and answers one question per station: **severe or not?**

---

## The Goal

Build a logistic regression classifier from scratch in numpy that takes real pollutant readings from Indian monitoring stations and outputs a 0 or 1 — safe vs severe.

Then go deeper:
- which pollutant does the model think matters most?
- where does the model get it wrong, and why?
- what happens when you change the decision threshold from 0.5 to 0.3?
- which cities are severe right now?

---

## The Dataset

**Source:** IndiaAI / CPCB — Real Time Air Quality Index from various locations  
**Link:** https://aikosh.indiaai.gov.in/home/datasets/details/real_time_air_quality_index_from_various_locations.html  
**License:** Open Government License, India  
**Snapshot:** 12 April 2026, 8:00 PM IST  
**Rows:** 3,433 — one per station per pollutant across India's monitoring network

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
| pollutant_avg | average reading — main feature used |

**Important:** this is a single-moment snapshot, not historical data. every row is from the same timestamp. the model learns what a severe station looks like *right now* — not over time.

**The target is derived** — PM2.5 > 250 or PM10 > 350 → label 1 (severe). everything else → label 0 (safe).

---

## The Approach

**Step 1 — understand the shape**  
count unique cities, stations, pollutants. get familiar before touching anything.

**Step 2 — pivot**  
data is long format — one row per station per pollutant. reshape so each row is one station with all pollutants as columns.

**Step 3 — label**  
derive binary target from PM2.5 and PM10 thresholds. check class balance — severe stations will be rare.

**Step 4 — build the classifier from scratch**  
- sigmoid — squashes any number into a 0-1 probability  
- binary cross-entropy loss — measures how wrong the probability is  
- backward pass — gradient of loss w.r.t. each weight, derived by hand first  
- training loop — forward → loss → gradients → update → repeat  

**Step 5 — evaluate**  
- confusion matrix (TP, TN, FP, FN) written manually  
- precision, recall, accuracy computed from those four numbers  

**Step 6 — go deeper**  
- which pollutant has the highest absolute weight? that's what the model thinks matters most  
- tune threshold from 0.5 → 0.3, watch false negatives (missed severe stations) drop  
- plot accuracy vs threshold across 0.1 to 0.9  
- break results down by city — which cities are severe right now?

---

## The Honest Limitation

this model was trained on a single snapshot. it learned what severe looks like at one moment in time — not across seasons, not across years. to actually *predict* tomorrow's air quality you'd need months of daily data and a time-series model. that's a future project.

that limitation is more interesting to think about than the accuracy number.

---

## Results

| Metric | Value |
|---|---|
| Accuracy | `[fill after run]` |
| False Negatives (missed severe stations) | `[fill after run]` |
| Most influential pollutant | `[fill after run]` |
| Best threshold | `[fill after run]` |
| Cities with most severe stations | `[fill after run]` |

---

## Key Insight

> *[fill this after you finish]*

---

## Stack

- Python 3.x
- numpy — model, loss, gradients, training loop
- pandas — pivot and cleaning only
- matplotlib — confusion matrix, threshold curve, weight visualization

---

## Run It

```bash
pip install numpy pandas matplotlib
jupyter notebook main.ipynb
```

---

*Part of a 90-day public ML build · [@skg_curious](https://x.com/skg_curious) · [LinkedIn](https://www.linkedin.com/in/shubhamgupta04907/)*