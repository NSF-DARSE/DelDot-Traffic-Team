# AWS Deployment — Delaware DOT Traffic Forecasting (Live System)

## What's Deployed

This is a **live, operational traffic forecasting system** running in AWS account
`062905933333` (us-east-1). It generates rolling 30-day predictions for 118
traffic monitoring stations and displays them in an interactive dashboard.

---

## Architecture

```
         ┌─────────────────────────────────────────────────────────────┐
         │                  Daily Automated Pipeline                     │
         │                                                              │
         │   5:00 AM ET          5:00 AM ET           6:00 AM ET        │
         │   EventBridge    →    Batch Lambda    →    QuickSight         │
         │   (trigger)           (predict 30d)        (SPICE refresh)   │
         │                            │                                  │
         │                            ▼                                  │
         │                    S3 forecast_latest.csv                     │
         └─────────────────────────────────────────────────────────────┘

         ┌─────────────────────────────────────────────────────────────┐
         │                  On-Demand Forecast API                       │
         │                                                              │
         │   User/App   →   API Gateway   →   Lambda   →   JSON        │
         │                  (REST API)        (forecast)    response     │
         └─────────────────────────────────────────────────────────────┘

         ┌─────────────────────────────────────────────────────────────┐
         │                  Dashboard (QuickSight)                       │
         │                                                              │
         │   DOT Staff   →   QuickSight Dashboard   ←   SPICE (in-mem) │
         │   (browser)       12 operational visuals      257K rows      │
         └─────────────────────────────────────────────────────────────┘
```

---

## Live Resources

### Storage (S3)

**Bucket:** `deldot-traffic-forecasting-062905933333`  
**Versioning:** Enabled  
**Total size:** ~129 MB

| Path | Content | Size |
|------|---------|------|
| `models/v2.2/model_artifact.json` | Serialised model + enhanced cold-start | 976 KB |
| `forecasts/submission_2025-08-17.csv` | Competition submission | 10.5 MB |
| `quicksight/rolling/forecast_latest.csv` | **Live 30-day rolling forecast** | 38.9 MB |
| `quicksight/rolling/history/` | Daily forecast archives | 38.9 MB each |
| `quicksight/detailed/quicksight_dataset.csv` | Historical analysis data | 37.3 MB |
| `quicksight/manifest_rolling.json` | QuickSight data source manifest | 264 B |

### Compute (Lambda)

| Function | Purpose | Memory | Timeout | Trigger |
|----------|---------|--------|---------|---------|
| `deldot-traffic-forecast` | On-demand API (single predictions) | 512 MB | 30s | API Gateway |
| `deldot-batch-forecast` | Daily batch (30-day rolling for all stations) | 1024 MB | 300s | EventBridge daily + manual URL |

### API Gateway

| API | Endpoint | Stage |
|-----|----------|-------|
| `deldot-traffic-api` (ID: `94d3hvwu93`) | `https://94d3hvwu93.execute-api.us-east-1.amazonaws.com/prod/` | prod |

**Endpoints:**
**Authentication:** API key required. Send it in the `x-api-key` header.
Requests without a valid key receive `HTTP 403`.

- `GET /health` — Health check (reports model version, whether enhanced cold-start is loaded)
- `GET /forecast?station=STN_0067&direction=1&date=2026-09-15&hour=16` — Single forecast
- `GET /forecast?station=STN_0067&direction=1&date=2026-09-15` — Full 24-hour profile
- `GET /explain?station=STN_0067&direction=1&date=2026-09-15&hour=16` — Factor-by-factor
  decomposition of the prediction (see `security_privacy_explainability.md`)
- `GET /explain?...&narrate=true` — adds a Bedrock-generated plain-English narrative,
  guarded by numeric containment (see `bedrock_narrative_layer.md`)

- `GET /best-hours?station=STN_0067&date=2026-08-19` — quietest hours (deterministic)
- `GET /best-window?station=STN_0067&start=...&end=...` — lowest-impact work window (deterministic)
- `GET /simulate-closure?station=...&start=...&duration_hours=...&lanes_closed=...` —
  closure impact on the station and its neighbours (see `closure_simulator.md`)
- `GET /ask?q=...` — natural-language planning questions (see `traffic_assistant.md`)

**Rate limits:** 25 req/s, burst 50, 50,000 requests/month quota.

### DynamoDB

| Table | Purpose | Billing |
|-------|---------|---------|
| `deldot-traffic-forecasts` | Cache for batch forecasts (available for future use) | PAY_PER_REQUEST |

### EventBridge

| Rule | Schedule | Target | Status |
|------|----------|--------|--------|
| `deldot-daily-forecast` | `cron(0 9 * * ? *)` (5am ET / 9am UTC) | `deldot-batch-forecast` Lambda | ENABLED |

### QuickSight

| Resource | Name | Details |
|----------|------|---------|
| Account | `deldot-traffic` | Enterprise edition |
| Data Source | `deldot-rolling-forecast-s3` | Points to S3 rolling manifest |
| Dataset | `DelDOT Live Forecasts (Next 30 Days)` | 257,760 rows in SPICE, daily refresh at 10:00 UTC |
| Dataset | `DelDOT Traffic Forecast Data` | Historical competition data (170,956 rows) |
| Dashboard | `DelDOT Traffic Intelligence — Operations Dashboard` | 3 sheets, 12 operational visuals |

### IAM

| Role | Used By | Permissions |
|------|---------|-------------|
| `deldot-traffic-lambda-role` | Both Lambda functions | S3 read/write, DynamoDB CRUD, CloudWatch Logs |

---

## Access Points

| What | URL |
|------|-----|
| **Dashboard** | `https://us-east-1.quicksight.aws.amazon.com/sn/dashboards/deldot-traffic-dashboard-v3` |
| **Forecast API** | `https://94d3hvwu93.execute-api.us-east-1.amazonaws.com/prod/forecast?station=STN_0067&direction=1&date=2026-09-15&hour=16` |
| **Health Check** | `https://94d3hvwu93.execute-api.us-east-1.amazonaws.com/prod/health` |
| **Manual Refresh** | IAM-authenticated only — `aws lambda invoke --function-name deldot-batch-forecast` (no public URL) |

---

## How to Refresh Data

### Automatic (daily)
- 5:00 AM ET: EventBridge triggers batch Lambda → writes new forecasts to S3
- 6:00 AM ET: QuickSight SPICE auto-refreshes from S3

### Manual
1. **Regenerate forecasts** (requires IAM credentials):
   ```bash
   aws lambda invoke --region us-east-1 --function-name deldot-batch-forecast \
     --cli-binary-format raw-in-base64-out --payload '{}' /tmp/out.json
   ```
2. **Refresh dashboard data:** In QuickSight → Datasets → `DelDOT Live Forecasts` → Refresh Now

---

## Dashboard Visuals

The dashboard (`DelDOT Traffic Intelligence — Operations Dashboard`) has 3 sheets
with 12 total visuals:

### Sheet 1: 30-Day Forecast Overview
1. **Forecast Summary** (table) — Count, avg volume, max volume, avg reliability by station type
2. **Daily Traffic Pattern** (line) — Hourly profile split by weekday/weekend
3. **30-Day Total Volume** (bar) — Daily total across all stations
4. **Volume by Direction** (bar) — NB/SB/EB/WB comparison

### Sheet 2: Maintenance Planning
5. **Peak Congestion Windows** (grouped bar) — Traffic by time-of-day, weekday vs weekend
6. **Best Days for Road Work** (bar) — Day-of-week ranking (lowest = best for closures)
7. **Busiest Stations** (ranked bar) — Highest-impact stations sorted descending
8. **Lowest Traffic Dates** (sorted bar) — Next 30 days sorted ascending (best dates for work)

### Sheet 3: Forecast Confidence
9. **Confidence Bounds** (multi-line) — Forecast + upper/lower 90% bounds by hour
10. **Reliability by Time & Type** (grouped bar) — When/where forecasts are most trusted
11. **Prediction Uncertainty** (bar) — Interval width by hour (wider = less certain)
12. **Station Type Split** (pie) — New vs established stations in the prediction set

---

## Cost Estimate (Current Monthly)

| Service | Usage | Est. Cost |
|---------|-------|-----------|
| S3 | ~200 MB + requests | $1 |
| Lambda (API) | ~1K invocations/month | $0.01 |
| Lambda (batch) | 30 invocations × 13s × 1GB | $0.10 |
| API Gateway | ~1K requests/month | $0.01 |
| DynamoDB | Idle (no reads/writes yet) | $0 |
| EventBridge | 30 invocations/month | $0 |
| QuickSight | 1 Enterprise author | $24 |
| **Total** | | **~$25/month** |

---

## Future Enhancements (Not Yet Built)

1. Road closure impact simulator (what-if analysis)
2. Anomaly alerting (SNS notifications when actuals deviate from forecast)
3. Maintenance scheduling optimizer
4. Weather data integration
5. Kinesis data ingestion from live ATR sensors
6. Additional QuickSight readers for DOT staff ($5/user/month each)
