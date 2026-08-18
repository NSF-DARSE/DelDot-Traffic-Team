# Demo Runbook — 2026-08-19, ~17:00 ET

## Open this

**https://d3u4765on3vdko.cloudfront.net/**

**API key:** `fy7olg5ctf73ztzlcFWyW58SQJfEEpU62qxeQM55`

Paste the key into the field at the top and press Save. It is held in
`sessionStorage` for that browser tab only, so re-paste it if you reopen the tab
or use a second machine.

**Dashboard:** https://us-east-1.quicksight.aws.amazon.com/sn/dashboards/deldot-traffic-dashboard-v3
(requires being signed in to QuickSight; the page links to it)

---

## Suggested run of play

The page has four quick-action buttons, so nothing needs typing.

1. **"Best time to travel tomorrow"** — simplest case. Shows the answer, the
   ranked hours, and the peak for contrast.
2. **"Best closure day next week"** — the operational payoff. Recommends Sunday
   and quantifies the vehicles avoided versus the worst day.
3. **Expand "Show computation"** — this is the point worth making out loud: the
   ranking is computed deterministically and the numbers are auditable. The model
   only phrased the result.
4. **"Closure impact: 2 of 3 lanes"** — the strongest technical moment. Closing
   two of three lanes at STN_0053 turns away 36,880 vehicles over 72 hours and
   runs over capacity for 38 of them. The table names **STN_0066** as the binding
   constraint, rising 76% to 86% of capacity, while the other four neighbours
   stay under 56%. The sensitivity line underneath shows how much that depends on
   the diversion assumption (50% -> 83%, 90% -> 89%).
5. **"Forecast for a new station"** (STN_0001) — a station with *no* observed
   history. The answer says so unprompted and carries a wider interval and a
   lower reliability score. Good honesty story.
6. Switch to the QuickSight dashboard for the 30-day view.

## Lines worth having ready

- **Primary metric is WAPE**, 13.43% on warm validation. GEH<5 is 77.2% and mean
  GEH 3.64, but WAPE is what the organisers score.
- **The model never produces a number.** `planning.py` computes and ranks; the
  language model converts a sentence into parameters and then describes the
  result. A numeric-containment guardrail discards any narrative that introduces
  a figure not present in the computation.
- **43 of 118 stations have no history.** Their profiles are reconstructed from
  neighbouring stations via the network graph, and they are labelled
  `cold_start_enhanced` with lower reliability rather than presented as equally
  certain.
- **Capacity is calibrated from the data**, not textbook numbers: observed
  per-lane throughput varies about threefold by road class (freeway ~1,400,
  signalised arterial ~480). That also proved `through_lanes` is per direction.
- **Cost:** roughly $25/month for the stack, and under a cent per question.

## If something misbehaves

| Symptom | Cause | Fix |
|---|---|---|
| "Enter the API key above first" | key not saved in this tab | paste it again |
| 403 in the chat panel | wrong or truncated key | re-copy, no trailing space |
| Narrative missing, table still shows | Bedrock hiccup or guardrail rejection | the computation is still correct; expand "Show computation". Re-ask — it is usually transient |
| Page will not load | CloudFront edge issue | fall back to calling the API directly (below) |

### Calling the API from a browser

The API needs an `x-api-key` **header**, which a URL bar cannot send. Three ways:

1. **API explorer on the page** (easiest). Pick an endpoint from the dropdown,
   adjust the query string, press Send. Shows the raw JSON, the HTTP status and
   the response time. Good answer if a judge asks to see the API directly.
2. **"Show computation"** under any chat answer — the same JSON the endpoint
   returns.
3. **Devtools console** (F12), on the page so the origin is allowed:

```javascript
k = sessionStorage.getItem('deldot_key');
A = 'https://94d3hvwu93.execute-api.us-east-1.amazonaws.com/prod';
f = (p) => fetch(A+p, {headers:{'x-api-key':k}}).then(r=>r.json()).then(d=>console.log(d));

f('/simulate-closure?station=STN_0053&start=2026-08-30&duration_hours=72&lanes_closed=2')
```

**Direct API fallback**, works in any terminal:

```bash
KEY=fy7olg5ctf73ztzlcFWyW58SQJfEEpU62qxeQM55
API=https://94d3hvwu93.execute-api.us-east-1.amazonaws.com/prod

# Deterministic, no language model involved — cannot fail on Bedrock
curl -H "x-api-key: $KEY" "$API/best-window?station=STN_0067&start=2026-08-24&end=2026-08-30"
curl -H "x-api-key: $KEY" "$API/best-hours?station=STN_0067&date=2026-08-20"
curl -H "x-api-key: $KEY" "$API/simulate-closure?station=STN_0053&start=2026-08-30&duration_hours=72&lanes_closed=2"
```

The deterministic endpoints are the safest thing to demo if Bedrock is unhealthy:
they return the same recommendation without any model call.

## Verified durable for the demo window

Checked 2026-08-18 16:0x ET. Nothing in the demo path depends on a temporary
credential.

| Component | State |
|---|---|
| CloudFront page | serving, HTTPS enforced |
| API Gateway + API key | enabled; 25 req/s, 50,000/month quota |
| Both Lambdas | Active, on `models/v2.2/model_artifact.json` |
| Bedrock agreements | `AVAILABLE` for Haiku and the Sonnet fallback |
| EventBridge daily batch | `ENABLED`, `cron(0 9 * * ? *)` = 05:00 ET |
| SPICE refresh | daily 10:00 UTC = 06:00 ET |
| S3 bucket | all four public-access blocks still enabled |

**Tomorrow's automated cycle finishes ~11 hours before you present:**
05:00 ET batch regenerates the rolling 30-day forecast, 06:00 ET SPICE reloads
it, so the dashboard will show 2026-08-19 onward.

The one risk outside our control is the workshop **account** itself expiring. If
that happens nothing in this runbook works, so it is worth loading the page once
in the morning as a smoke test.

## Note on relative dates

Questions resolve against the day they are asked. During the demo on 19 August,
"tomorrow" means the 20th and "next week" means Mon 24 – Sun 30 August. The
assistant states the range it used in its answer.
