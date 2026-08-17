# Security, Privacy & Explainability

This document records the security, privacy, and explainability posture of the
traffic forecasting system, including issues found during review and how they
were resolved.

---

## 1. Privacy

### What data the system handles

The system processes **aggregate hourly vehicle counts only**. Specifically:

| Field | Content | Privacy implication |
|-------|---------|---------------------|
| `station_key` | Opaque identifier (`STN_0001`–`STN_0127`) | No location disclosed |
| `timestamp` | Hour-level granularity | No sub-hour behaviour |
| `direction_code` | Travel direction (N/S/E/W) | Aggregate only |
| `volume` | Count of vehicles in that hour | Aggregate only |
| Road context | Lanes, speed limit, surface type, AADT band | Infrastructure, not people |

### Privacy properties

- **No personally identifiable information.** No plate numbers, no vehicle
  identifiers, no trip origins or destinations, no individual trajectories.
- **No individual tracking is possible.** The finest resolution is
  "N vehicles passed this station during this hour." A single vehicle cannot be
  isolated or followed between stations.
- **Station identifiers remain anonymous.** We did not attempt to
  reverse-map `station_key` values to physical roads. No external traffic
  datasets, map APIs, or public archives were used for de-anonymisation. This
  complies with the stated data-use rules.
- **Aggregation is inherent to the model.** The multiplicative model consumes
  group means (per station, hour, day-of-week, month). Individual observations
  only ever influence an average.
- **Synthetic cold-start history is derived, not observed.** For stations with
  no history, we reconstruct estimated profiles from *neighbouring stations'
  aggregate patterns*. No withheld target values are used, and the result is
  labelled an estimate rather than observed truth (`prediction_path` field).

### Data residency and retention

- All data stays in `us-east-1`.
- S3 versioning is enabled, so forecast history is auditable.
- Daily forecast archives are written to `quicksight/rolling/history/`.

---

## 2. Security

### Issues found during review and resolved

Two genuine gaps were identified in the deployed system and fixed.

#### Issue 1 — Unauthenticated public Lambda function URL (resolved)

**What it was:** The batch forecast Lambda had a function URL with
`AuthType: NONE` and a resource policy granting `Principal: "*"`.

**Why it mattered:** This was not merely a read exposure. Anyone on the
internet who discovered the URL could:
- trigger a compute job on demand (cost / denial-of-wallet), and
- cause the dashboard's underlying S3 data to be **overwritten**.

**Resolution:**
- Deleted the function URL configuration.
- Removed the `Principal: "*"` permission statement.
- The only remaining invoke permission is the EventBridge daily schedule.

**Verified:** `get-function-url-config` now returns not-found; the resource
policy contains a single statement (`Sid=eventbridge-daily`,
`Principal={"Service":"events.amazonaws.com"}`).

**Trade-off accepted:** The convenience of a browser-clickable "refresh"
link is gone. Manual refresh now requires IAM credentials (see below).

#### Issue 2 — Unauthenticated REST API (resolved)

**What it was:** API Gateway methods had `authorizationType: NONE` and
`apiKeyRequired: false`.

**Resolution:**
- `apiKeyRequired: true` on both the root and `{proxy+}` methods.
- Created an API key and a usage plan with throttling and a quota.
- Redeployed the `prod` stage.

**Verified:**

| Request | Result |
|---------|--------|
| `GET /health` with no key | `HTTP 403` |
| `GET /health` with `x-api-key` | `HTTP 200` |

### Current controls

| Control | Status | Detail |
|---------|--------|--------|
| API authentication | Enforced | API key via `x-api-key` header |
| Rate limiting | Enforced | 25 req/s, burst 50 |
| Quota | Enforced | 50,000 requests/month |
| S3 public access | Blocked | All four block-public-access flags true |
| S3 encryption at rest | Enabled | SSE-S3 (AES256) |
| S3 bucket policy | Scoped | QuickSight service principal only, conditioned on `aws:SourceAccount` |
| Encryption in transit | Enforced | HTTPS/TLS on API Gateway and S3 |
| IAM | Least privilege | Lambda role limited to this bucket and this DynamoDB table |
| Batch job invocation | IAM only | EventBridge schedule; no public entry point |
| Audit trail | Enabled | CloudWatch Logs per function; S3 versioning |

### How to refresh forecasts now (authenticated)

```bash
# Regenerate the 30-day rolling forecast (requires IAM credentials)
aws lambda invoke --region us-east-1 \
  --function-name deldot-batch-forecast \
  --cli-binary-format raw-in-base64-out \
  --payload '{}' /tmp/out.json

# Reload the dashboard data into SPICE
aws quicksight create-ingestion --region us-east-1 \
  --aws-account-id 062905933333 \
  --data-set-id deldot-rolling-forecast-dataset \
  --ingestion-id "manual-$(date +%s)"
```

### Recommended before a real production deployment

These are appropriate for a state agency but out of scope for the hackathon build:

1. Replace the static API key with IAM/SigV4 or Amazon Cognito. API keys
   identify a caller but are not a strong authentication mechanism.
2. Store the API key in AWS Secrets Manager rather than a local file.
3. Place Lambdas in a private VPC subnet and restrict the API to the DOT network
   (or an AWS PrivateLink endpoint).
4. Enable AWS CloudTrail data events and AWS Config rules.
5. Add SSE-KMS with a customer-managed key for key rotation and separation of duties.
6. Add WAF in front of API Gateway for IP allow-listing and request inspection.

---

## 3. Explainability

Explainability was a deliberate architecture decision, not an add-on.

### Why the model is inherently interpretable

The core forecast is a **product of named factors**, each with a direct
real-world meaning:

```
forecast = station_baseline
         x hour_of_day_factor
         x day_of_week_factor
         x month_of_year_factor
         x station_hour_adjustment
         x station_month_adjustment
         x year_over_year_trend
```

Consequences:

- Every prediction can be reproduced by hand with a calculator.
- Every factor is auditable and can be challenged independently
  ("is the November seasonal factor right?").
- There is no opaque embedding or hidden state.
- Machine learning (LightGBM) is used **only as a residual correction** on top
  of the transparent base model, contributing roughly an 11% MAE improvement.
  The interpretable structure carries the prediction; the ML layer refines it.

This contrasts with a sequence model (LSTM/Transformer) which would likely
score similarly but could not answer "why 1,193?".

### The `/explain` endpoint

A dedicated endpoint returns the full decomposition of any single prediction.

```bash
curl -H "x-api-key: $KEY" \
  "$API/explain?station=STN_0067&direction=1&date=2026-09-15&hour=16"
```

Verified output for a **known station** (`STN_0067`, 4pm Tue 15 Sep 2026):

| Factor | Value | Running total | Meaning |
|--------|------:|--------------:|---------|
| station_baseline | 548.02 | 548.0 | Observed average volume for this station/direction |
| hour_of_day | 1.7632 | 966.3 | Hour 16 on a weekday vs network average |
| day_of_week | 0.9976 | 964.0 | Tuesday vs weekly average |
| month_of_year | 0.9723 | 937.3 | September seasonal effect |
| station_hour_adjustment | 1.0456 | 980.0 | This station's own hourly shape |
| station_month_adjustment | 1.0463 | 1025.4 | This station's own seasonal shape |
| year_over_year_trend | 0.9643 | **988.7** | Observed trend applied to 2025+ |

Result: **988.7 veh/hr**, 90% interval [630.9, 1346.6], reliability 0.75.

Verified output for a **cold-start station** (`STN_0001`, same time):

| Factor | Value | Running total | Meaning |
|--------|------:|--------------:|---------|
| station_baseline | 528.79 | 528.8 | Estimated from synthetic history reconstructed from neighbours |
| hour_of_day | 1.8175 | 961.1 | Station-specific profile (not the network average) |
| day_of_week | 0.9881 | 949.7 | Tuesday effect |
| month_of_year | 1.0711 | 1017.2 | September seasonal effect |
| year_over_year_trend | 0.8972 | **912.7** | Trend inherited from neighbouring stations |

Result: **912.7 veh/hr**, 90% interval [450.3, 1375.1], reliability 0.53.

Note the honesty of the contrast: the cold-start station gets a **wider interval**
(CV inflated 1.4x) and a **lower reliability score**, and the response states
plainly that its baseline is an estimate.

### What `/explain` returns

- `prediction_path` — one of `known_station`, `cold_start_enhanced`,
  `cold_start_fallback`. Makes the provenance of every number explicit.
- `decomposition` — each factor, its value, the running product, and a
  plain-English meaning.
- `uncertainty` — the coefficient of variation and the exact interval formula.
- `reliability` — the base score plus every named adjustment and its delta,
  so the confidence number is itself explainable.
- `privacy_note` — a reminder that the data is aggregate and anonymous.

### Provenance in the dashboard data

The rolling forecast CSV now carries `prediction_path` and `model_version`
columns, so any dashboard figure can be traced to the method and model
version that produced it.

Current 30-day batch composition:

| Path | Rows | Meaning |
|------|-----:|---------|
| `cold_start_enhanced` | 123,840 | Station-specific profiles from synthetic neighbour history |
| `known_station` | 108,000 | Fully observed history |
| `cold_start_fallback` | 25,920 | Road-feature regression (no connected neighbours) |

### Limits of explainability we state openly

- The LightGBM residual layer is applied in the offline submission pipeline and
  is less directly interpretable than the base factors. It is bounded to a small
  correction, and the base model remains fully transparent.
- Cold-start predictions are **structurally** explainable (we can name the
  neighbours and weights used) but **not empirically validated** — no ground
  truth exists for those stations. Reliability scores of roughly 0.53 encode
  this uncertainty rather than hiding it.
