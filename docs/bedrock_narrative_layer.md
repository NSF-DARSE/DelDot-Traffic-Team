# Bedrock Narrative Layer

Turns the deterministic factor decomposition into a plain-English paragraph using
Amazon Bedrock (Claude). Built 2026-08-18.

**Status: live.** Model access was enabled via the CLI on 2026-08-18 (see
[Enabling model access](#enabling-model-access)). Serving on
`claude-haiku-4-5` at roughly **$0.0017 per call**.

---

## Design contract

**The language model never produces a number.** It only rephrases values the
multiplicative model already computed. This preserves the auditability that makes
the forecast defensible: every figure in the narrative traces to a factor in the
decomposition.

This is the whole reason the feature is safe to put in front of a DOT audience.
An LLM that *generated* traffic figures would produce plausible, unverifiable
numbers and destroy the explainability property the model was built for.

## Usage

```bash
# Deterministic only (default, unchanged, no Bedrock call)
curl -H "x-api-key: $KEY" \
  "$API/explain?station=STN_0067&direction=1&date=2026-09-15&hour=16"

# With narrative (opt-in)
curl -H "x-api-key: $KEY" \
  "$API/explain?station=STN_0067&direction=1&date=2026-09-15&hour=16&narrate=true"
```

Opt-in by design: the default path stays deterministic and free.

## Guardrail: numeric containment

After generation, every number in the narrative is extracted and matched against
the values supplied in the prompt. If the model introduces an unsupported figure,
**the narrative is discarded** and only the decomposition is returned.

Permitted values are the forecast, interval bounds, reliability score, every
factor value and running total, percentage restatements of a factor
(`1.8957` → "89.6% higher"), reliability deltas, the coefficient of variation,
date/time components of the requested timestamp, and station-identifier digits.
Small integers (0–100) are allowed as ordinary prose. Tolerance is 2%, plus
rounding to the nearest ten for volumes.

### Verified behaviour

| Case | Expected | Result |
|---|---|---|
| Only supplied numbers | accept | accept |
| Fabricated volume "2500" | reject | reject |
| Invented "47.3% above regional mean" | reject | reject |
| Factor restated as "89.6%" | accept | accept |
| Volume rounded to "990" | accept | accept |
| "9 times out of 10" prose | accept | accept |
| Year "2026" from the timestamp | accept | accept |

The last row was a real false positive caught in live testing: the first
production call produced a good narrative that was rejected because it mentioned
the year. Date components are now permitted.

## Graceful degradation

Any failure — access denial, throttling, timeout, or a failed containment check —
returns the decomposition with `narrative: null` and a `narrative_meta.status`
explaining why. Verified with Bedrock unavailable:

| Endpoint | Result |
|---|---|
| `/health` | OK, v2.2 |
| `/forecast` single hour | OK, 986.5 veh/hr |
| `/forecast` full day | OK, 24 hours |
| `/explain` without `narrate` | OK, 7 factors, no `narrative_meta` |
| `/explain` with `narrate=true` | OK, decomposition intact, reason reported |
| Auth without API key | 403 |

The layer is strictly additive. Nothing that worked before can break.

## Model selection and fallback

| Setting | Value |
|---|---|
| `NARRATIVE_MODEL_ID` | `us.anthropic.claude-haiku-4-5-20251001-v1:0` |
| `NARRATIVE_FALLBACK_MODELS` | `us.anthropic.claude-sonnet-4-6` |
| `BEDROCK_REGION` | `us-east-1` (same region as the stack) |
| `max_tokens` | 320 |
| `temperature` | 0.2 |

Haiku is the default for cost. On `AccessDenied`, `Validation` or
`ResourceNotFound` the next model is tried, so a single unavailable model does not
disable the feature. `claude-haiku-4-5` requires a cross-region **inference
profile** (`us.` prefix); the bare model ID is not invokable.

## Example output

Known station STN_0067, 2026-09-15 16:00, via `claude-sonnet-4-6`,
**569 input / 223 output tokens**:

> The central forecast for STN_0067 direction 1 at 4 pm on Tuesday 15 September
> 2026 is 987 vehicles per hour, built on observed historical volumes at this
> station rather than estimated data. The afternoon peak hour is by far the
> dominant driver, nearly doubling the station baseline, while the seasonal and
> day-of-week effects are modest and partially offset by a slight downward
> year-over-year trend. […] The 90 percent interval runs from 630 to 1344
> vehicles per hour, which is a wide spread and the more meaningful guide to
> uncertainty here. The reliability score of 0.75 is a relative indicator,
> lifted slightly by the recognisable regular peak pattern but pulled down
> because this forecast sits beyond the near-term horizon.

Every figure traces to the decomposition, it states the basis is observed rather
than estimated history, and it correctly describes reliability as a *relative*
indicator with the interval carrying the statistical meaning — all three of the
properties the prompt requires.

Prompt size is ~213 tokens of structure plus the decomposition; ~570 input tokens
in practice. See [Measured cost](#measured-cost).

## Enabling model access

Bedrock gates each foundation model behind an AWS Marketplace agreement. This is
separate from IAM: the role can hold `bedrock:InvokeModel` and still be refused
with

```
AccessDeniedException: not authorized to perform the required AWS Marketplace
actions (aws-marketplace:ViewSubscriptions, aws-marketplace:Subscribe)
```

Granting those Marketplace actions in the Lambda's IAM policy does **not** fix
it — the agreement itself must exist. It can be created entirely from the CLI,
no console required:

```bash
REGION=us-east-1
MODEL=anthropic.claude-haiku-4-5-20251001-v1:0

# 1. Diagnose. agreementAvailability.status is the field that matters.
aws bedrock get-foundation-model-availability --region $REGION --model-id $MODEL

# 2. Retrieve the offer token.
TOKEN=$(aws bedrock list-foundation-model-agreement-offers \
  --region $REGION --model-id $MODEL \
  --query 'offers[0].offerToken' --output text)

# 3. Accept the agreement (performs the Marketplace subscription).
aws bedrock create-foundation-model-agreement \
  --region $REGION --model-id $MODEL --offer-token "$TOKEN"

# 4. Confirm. PENDING becomes AVAILABLE within a minute or so.
aws bedrock get-foundation-model-availability --region $REGION --model-id $MODEL \
  --query 'agreementAvailability.status'
```

Related commands: `put-use-case-for-model-access` submits the Anthropic use-case
form when required (it was already on file in this account),
`get-use-case-for-model-access` reads it back, and
`delete-foundation-model-agreement` revokes access.

Note the model ID here is the **bare** foundation-model ID, not the `us.` prefixed
inference profile used for invocation.

### Observed sequence, 2026-08-18

| Step | Result |
|---|---|
| Before | `agreement NOT_AVAILABLE`, auth `AUTHORIZED`, entitlement `AVAILABLE`, region `AVAILABLE` |
| After `create-foundation-model-agreement` | `agreement PENDING` |
| ~1 minute later | `agreement AVAILABLE` for both Haiku and the Sonnet fallback |
| Live call | Haiku serving, guardrail `numeric_containment_passed` |

While Haiku was `PENDING`, the fallback chain served Sonnet and the endpoint kept
working — the degradation design behaving as intended.

## Measured cost

Rate card for `claude-haiku-4-5` in us-east-1, read from the agreement offer:

| Dimension | Price per 1M tokens |
|---|---:|
| Input, standard | $1.10 |
| Output, standard | $5.50 |
| Cache read | $0.11 |
| Cache write | $1.375 |

Observed calls:

| Case | Tokens in/out | Cost |
|---|---|---:|
| Known station (STN_0067) | 568 / 202 | $0.001736 |
| Cold-start station (STN_0001) | 539 / 231 | $0.001863 |

So roughly **$1.80 per 1,000 narrated explanations**. Because the narrative is
opt-in and never runs in batch, this is negligible against the ~$25/month stack.

## Scope

**Covered:** single-prediction explanation, on demand, for both known and
cold-start stations, including why the interval is wide and what the reliability
score reflects.

**Not covered:** generating any number; the "when should I close this road"
planning agent (needs multi-step querying and ranking); batch narration of the
170,956 submission rows; any graded deliverable. The submission and dashboard are
untouched.

## Privacy

Data sent to Bedrock is anonymous aggregate hourly counts and derived factor
values — no PII, no station de-anonymisation, consistent with the posture in
`security_privacy_explainability.md`. The station key is an opaque identifier.
