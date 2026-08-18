# 2026 Hennovate the State — Hackathon Day 1 Feedback

**Project Name:** Traffic Pattern Analysis  
**Date:** 8/17/2026  
**Team members:** [Your team names here]

---

## 1. Our Understanding of the Problem

**a. What is the current process?**

Delaware DOT collects hourly traffic counts from automatic traffic recorders (ATRs) across the state's road network. Currently, traffic volume estimation for planning (road closures, maintenance scheduling, capacity analysis) relies on historical averages, manual lookups, and engineer experience. There's no automated forecasting system that accounts for time-of-day, seasonality, day-of-week, or year-over-year trends — and no way to estimate traffic at new sensor locations with no historical data.

**b. What is the key pain point?**

Two main pain points: (1) Planning road work or closures without knowing the specific traffic impact — leading to either overly conservative timing or unexpected congestion. (2) New monitoring stations come online with zero history, and there's no systematic way to forecast their traffic until months of data accumulate.

**c. What would a useful solution look like?**

A system that provides hourly traffic volume forecasts for any station (including new ones), any date up to 30 days out, with confidence intervals — refreshed daily and accessible through an interactive dashboard. Staff can look up a station, see when traffic is lightest, and plan maintenance accordingly.

---

## 2. What We Explored Today

**a. Approach or solution attempted**

Built a multiplicative hierarchical forecasting model: Station Baseline × Hourly Profile × Day-of-Week × Monthly Seasonality × Year-over-Year Trend, refined with LightGBM machine learning. For 43 new stations with no history, we generated synthetic historical profiles from neighboring stations using network graph relationships. Also built 5 scenario analyses (road closures, weather events, construction, etc.).

**b. Data/case-study materials explored**

- 902K hourly traffic observations across 75 stations (2024)
- 557K validation observations (Jan–Jun 2025)
- Network metadata: 127 stations with road characteristics (lanes, speed, functional class)
- 186 spatial relationships between stations (same-corridor and proximity edges)
- Discovered a critical 2-hour timestamp offset between training and submission data

**c. Platforms/tools tested**

- Python (pandas, numpy, scikit-learn, LightGBM) for model development
- AWS Lambda for serverless forecast serving (on-demand API + daily batch)
- AWS S3 for data lake and model artifact storage
- AWS API Gateway for REST API endpoint
- AWS EventBridge for daily automated pipeline scheduling
- AWS QuickSight for interactive dashboards (3 sheets, 12 visuals)
- AWS DynamoDB for forecast caching (provisioned for future use)

---

## 3. What Worked

**a. Case study: What was clear or promising?**

The traffic data has extremely strong and stable patterns — station rankings are 99.4% correlated between years, hourly/weekly/seasonal profiles are highly regular. This makes the problem very well-suited to a multiplicative decomposition approach. The network graph (edges between stations) proved valuable for cold-start prediction. The expert clarification that we could reconstruct synthetic history for new stations unlocked a significant improvement.

**b. Platform: What capabilities worked well?**

The AI-assisted development workflow (Kiro CLI) was highly effective — went from raw data exploration to a fully deployed AWS system with live API, automated daily pipeline, and interactive dashboard in a single day. LightGBM provided a clean 11% accuracy improvement as a residual correction layer. QuickSight SPICE ingestion was fast (257K rows in seconds) and the daily refresh schedule means zero manual maintenance.

---

## 4. What Didn't Work

**a. Case Study: Missing data, unclear requirements, assumptions, etc.**

- The 2-hour timestamp offset between training and validation was undocumented and nearly derailed the model (R² went from 0.76 to 0.94 after discovery). This required careful detective work. Stakeholders subsequently supplied a corrected file (2026-08-18) with proper timestamps and the two previously-missing hours, confirming the diagnosis and lifting R² to 0.95.
- Scenario descriptions were not provided — we had to define our own 5 scenarios.
- No weather or event data was available, limiting scenario analysis to literature-based estimates.
- The GEH < 5 industry standard (85%) is designed for calibrated microsimulations, not batch forecasts. Our 77.2% is strong for the task but doesn't hit that benchmark.
- Cold-start stations (43 of 118) cannot be empirically validated — we have zero ground truth for them.

**b. Platform: Technical limitations, integration issues, learning curve, etc.**

- QuickSight dashboard creation via API is complex (JSON definitions are verbose and error-prone with strict type requirements for columns).
- QuickSight S3 data source permissions required multiple attempts to configure correctly (bucket policy + role-based access).
- Lambda cold starts add ~3 seconds on first invocation (acceptable but noticeable).
- The batch Lambda generates 258K rows in 13 seconds — approaching the point where we'd need to optimize or split if station count grows significantly.

---

## 5. Tomorrow's Focus

**a. What are the next steps?**

1. Finalize the presentation with live dashboard demo
2. Consider building the "road closure impact simulator" — let users pick a station + time and see estimated impact on the network
3. Explore whether we can improve GEH < 5 rate (currently 77.2%) by adding per-station error correction or additional features
4. Prepare scenario analysis talking points for audience Q&A
5. Test the dashboard with realistic stakeholder questions ("When should I close I-95 for bridge work?")
