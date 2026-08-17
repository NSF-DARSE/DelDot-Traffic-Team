# Traffic Forecasting Model


## The Problem

Delaware DOT has traffic counters at 118 locations across the state's road
network. They need to predict **how many cars will pass each counter, every
hour, for the next several months** — even for 43 brand-new counters that
have zero history.

They also want to know: "How confident are you in this number?" and "What
happens if a road closes or an event occurs?"

---

## Our Solution — "Multiply the Patterns"

We think of traffic like a recipe. The final traffic count at any location and
time is a combination of several **layered patterns**, all multiplied together:

```
Traffic = (How busy is this road?) 
        × (What time of day is it?) 
        × (What day of the week?) 
        × (What month?) 
        × (Is traffic growing or shrinking year-over-year?)
        × (Fine-tuning adjustment)
```

### Layer 1: "How busy is this road?" (Station Baseline)

Every road has a typical daily volume. An interstate might see 5,000 cars/hour
at peak; a rural collector might see 90. We learn each station's baseline
from historical data.

**For the 43 new stations with no history:** We use their road characteristics
— number of lanes, speed limit, road type — plus what we know about nearby
stations to *estimate* their typical volume. It's like saying: "This is a
2-lane, 55mph arterial. We know 20 roads like that, and they average about
800 cars/hour."

### Layer 2: "What time patterns does traffic follow?"

Traffic seems to have incredibly regular rhythms:
- **Hourly:** Very low at 2am, builds to a morning rush (7-9am), lunch bump,
  then a bigger evening peak (4-6pm)
- **Day of week:** Friday is 12% busier than average; Sunday is 14% quieter
- **Monthly/seasonal:** July is the busiest month (+13%); November is quietest (-28%)
- **Weekend shape:** Weekends don't have the commuter spikes — traffic builds
  gradually and peaks around midday

Each of these is a multiplier. A Friday in July at 5pm might be:
1.12 (Friday) × 1.13 (July) × 1.76 (5pm weekday) = a very busy period.

### Layer 3: "Is traffic trending up or down?"

We discovered that 2025 traffic is about **10% lower** than 2024 across most
stations. We apply this correction so we don't systematically over-predict.

### Layer 4: "Fine-tuning with Machine Learning" (LightGBM)

After the basic recipe does its job, there are still small patterns it misses
— maybe a specific road type behaves differently at certain times of year, or
certain week numbers are unusual. A machine learning model (LightGBM) learns
these leftover patterns and makes small corrections. This improved accuracy
by another **11%**.

---

## Key Discovery: The Clock Was Off

The single biggest insight was discovering that the **training data timestamps
were shifted by 2 hours** compared to the submission timestamps. Without
fixing this, the model thought rush hour was at 2pm instead of 4pm, and all
predictions were systematically wrong. After correcting the alignment,
accuracy jumped from R²=0.76 to **R²=0.94**.

This is the kind of data quality issue that can only be found by careful
exploratory analysis — and it made all the difference.

---

## How Do We Handle Uncertainty?

For each prediction, we also provide:

1. **A confidence range (90% interval):** "We think 1,200 cars/hour, and
   we're 90% sure it'll be between 950 and 1,450." These ranges are wider
   when we're less sure (new stations, nighttime, far in the future) and
   tighter when the pattern is well-established.

2. **A reliability score (0 to 1):** A quick summary of how much to trust
   this particular prediction. Known stations in regular hours get 0.75–0.82.
   New stations with no history get 0.32–0.45.

---

## Why This Works

| Strength | Reason |
|----------|--------|
| **Interpretable** | Each layer has a clear meaning. You can explain *why* the model predicts 1,200 at a given time. |
| **Handles new stations** | Uses road characteristics and nearby stations — doesn't require years of data. |
| **Robust** | The multiplicative structure means a single bad estimate doesn't destroy everything. |
| **Calibrated uncertainty** | Confidence intervals tested against real data: 89.2% coverage (target: 90%). |
| **Accurate** | Average error of ~105 vehicles/hour with R²=0.94 on validation data. |
| **Cost-effective** | Runs on AWS serverless for ~$210/month. No expensive always-on infrastructure. |

---

## Results at a Glance

- **170,956 predictions** generated across 118 stations
- **93.8% of traffic variation explained** (R² = 0.938)
- **Average error: ~105 vehicles/hour** (with machine learning refinement)
- **GEH statistic: 3.86 mean** (industry uses < 5 as "good")
- **93.1% of predictions** are within acceptable accuracy (GEH < 10)
- **43 new stations** predicted using synthetic history from neighbors
- **Submission validates: PASS** ✓

---

## What's Running Right Now in AWS

This isn't a plan — it's deployed and live:

1. **Every morning at 5am**, the system automatically generates fresh 30-day
   predictions for all 118 stations (257,760 forecasts in ~13 seconds)
2. **By 6am**, the QuickSight dashboard refreshes with the latest data
3. **Staff open the dashboard** and see this week's and next month's
   predicted traffic — always forward-looking, never stale
4. **On-demand API** available for any app to query a specific station/date/hour

**Dashboard URL:** Ask your admin for QuickSight access  
**API endpoint:** `https://94d3hvwu93.execute-api.us-east-1.amazonaws.com/prod/forecast` (API key required)  
**Manual refresh:** requires AWS credentials (no public endpoint, by design)

**Current cost:** ~$25/month (mostly QuickSight license). Scales to 500+ stations
with no architecture changes.

---

## What Could Make It Even Better?

- Adding weather data (rain/snow reduce traffic 10-20%)
- Holiday calendars (Thanksgiving week is very different from normal)
- Real-time updates (use yesterday's actuals to refine today's forecast)
- Event data (concert at a venue nearby → temporary spike)

---

## Questions?

The model code is fully reproducible. Run:
```
python3 src/run_pipeline.py
```
and the entire pipeline regenerates from raw data in under 10 minutes.

All documentation lives in `docs/`.
