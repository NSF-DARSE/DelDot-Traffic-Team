# Lessons Learned — Building on AWS

What we ran, and what actually surprised us. Every item here cost us real
debugging time, so each is written as the thing we wish we had known.

---

## What we ran

| Service | Use | Verdict |
|---|---|---|
| **S3** | Data lake, model artifacts, dashboard feed, static page | Versioning saved us when a merge reverted outputs |
| **Lambda** ×2 | On-demand forecast API; daily 30-day batch | 257,760 forecasts in ~13 s on 1 GB. No servers to manage |
| **API Gateway** | REST API, API-key auth, throttling, quota | Auth was one flag; the CORS interaction was not (see below) |
| **EventBridge** | Daily 05:00 ET batch trigger | One rule, zero maintenance |
| **QuickSight** | 3-sheet dashboard, SPICE, scheduled refresh | 257k rows ingest in ~14 s. Dashboard-as-code via API is painful |
| **DynamoDB** | Forecast cache (provisioned, ultimately unused) | On-demand billing meant an unused table cost nothing |
| **CloudFront** | Stable HTTPS entry point with Origin Access Control | The correct answer for serving a private-bucket page |
| **Bedrock** | Claude Haiku for language only | Model *access* is the hard part, not the API |
| **IAM** | Least-privilege roles per function | Where most of our lost time went |

Total running cost: roughly **$25/month**, dominated by the QuickSight licence.
Lambda, EventBridge, and S3 together were under $2.

---

## The five things that cost us the most time

### 1. Bedrock model access is a Marketplace agreement, not an IAM permission

We had `bedrock:InvokeModel` on the Lambda role and still got:

```
AccessDeniedException: not authorized to perform the required AWS Marketplace
actions (aws-marketplace:ViewSubscriptions, aws-marketplace:Subscribe)
```

The error names IAM actions, so we spent three attempts adding those permissions
to the role — conditioned, then unconditioned. **None of it helped**, because the
missing thing was the *agreement*, not the permission. `get-foundation-model-availability`
is the diagnostic that actually tells you:

```
agreementAvailability.status : NOT_AVAILABLE   <- the real gap
authorizationStatus          : AUTHORIZED
entitlementAvailability      : AVAILABLE
regionAvailability            : AVAILABLE
```

The fix is two CLI calls, no console needed:

```bash
TOKEN=$(aws bedrock list-foundation-model-agreement-offers \
  --model-id anthropic.claude-haiku-4-5-20251001-v1:0 \
  --query 'offers[0].offerToken' --output text)

aws bedrock create-foundation-model-agreement \
  --model-id anthropic.claude-haiku-4-5-20251001-v1:0 --offer-token "$TOKEN"
```

`PENDING` becomes `AVAILABLE` in about a minute. Two further gotchas: use the
**bare** model ID here, not the `us.`-prefixed inference profile you invoke with;
and newer models such as `claude-haiku-4-5` are `INFERENCE_PROFILE`-only, so the
bare ID is not invokable.

**Lesson:** read the availability API, not the error message. And a model listed
by `list-foundation-models` is not necessarily a model you can call.

### 2. On a greedy proxy resource, `ANY` shadows an explicit `OPTIONS`

Our browser page failed CORS preflight with 403 while `curl` worked perfectly.
The API Gateway `OPTIONS` method was configured correctly — `authorizationType:
NONE`, `apiKeyRequired: false` — and still returned `{"message":"Forbidden"}`.

The tell: `OPTIONS` **with** an API key returned 200. The request was matching the
`ANY` method on `/{proxy+}`, which required a key. A browser cannot attach a
custom header to a preflight, so the page could never authenticate.

Fix: replace `ANY` with explicit `GET` (key required) and `OPTIONS` (no key).

**Lesson:** `ANY` is convenient until you need one method to behave differently.
If CORS matters, declare methods explicitly from the start.

### 3. Presigned URLs inherit the signer's credential lifetime

We generated a presigned S3 URL with `--expires-in 604800` (7 days) and assumed
we had a week. We did not: the URL was signed with temporary workshop
credentials, so it would have died with them — potentially before the demo.

**Lesson:** `--expires-in` is an upper bound, not a guarantee. For anything that
must outlive your session, use CloudFront with Origin Access Control. It also let
us keep all four S3 public-access blocks enabled and grant read only to the
CloudFront service principal, conditioned on the distribution ARN.

### 4. An explicit `Deny` binds the principal it is attached to, not the resource

A workshop guardrail policy denied `bedrock:InvokeModel` on 58 models. We assumed
that blocked the feature outright. It did not: the policy was attached to
`WSParticipantRole`, and our Lambda ran as `deldot-traffic-lambda-role`.

The practical consequence was subtler than a simple pass/fail — we could not
*test* Bedrock with our own credentials even though the Lambda could call it.
That made the first failure look like a code bug.

**Lesson:** when debugging access, confirm *which principal* is being evaluated.
"I can't do it" and "my Lambda can't do it" are different questions.

### 5. Infrastructure code has to run on the machine you have

`scripts/deploy.sh` failed on macOS with an unhelpful error. Two causes:

- macOS ships **bash 3.2**, where `"${arr[@]}"` on an *empty* array under
  `set -u` raises `unbound variable`. We replaced array-based dispatch with
  function redefinition.
- A regex that rewrote `aws ` calls to a wrapper function also rewrote the
  *credential-detection* lines, which run before the wrapper is defined —
  producing "command not found" that the script reported as expired credentials.

**Lesson:** test deployment scripts on the target shell, and be wary of
mechanically rewriting code you are about to execute.

---

## Smaller things worth knowing

- **QuickSight S3 datasets need all-`STRING` input columns.** Casting happens in
  the logical table via `CastColumnTypeOperation`. Declaring a `DATETIME` input
  column fails outright.
- **QuickSight `IngestionId` allows only alphanumerics and hyphens.** Our
  `deploy-v2.2-<epoch>` was rejected for the dot.
- **A QuickSight visual's `CategoricalDimensionField` rejects integer columns.**
  Use `NumericalDimensionField`, or the analysis silently reaches
  `CREATION_FAILED` and you have to read `Analysis.Errors` to find out why.
- **Dashboard "last updated" refers to the definition, not the data.** Ours read
  a day old while SPICE was refreshing hourly — easy to misread as staleness.
- **Lambda cold start with a 976 KB JSON artifact from S3** is about 3 s; warm
  calls are under 200 ms. Caching the artifact in a module-level global made this
  a non-issue.
- **JSON beat pickle for the model artifact.** No library-version coupling
  between our laptop and the Lambda runtime, no arbitrary-code-execution risk on
  load, and the artifact stays human-inspectable. Tuple keys had to be
  stringified, which cost a few lines and was worth it.

---

### QuickSight embed URLs are single use, and region-pinned

Two traps in one feature. The dashboard exists only in `us-east-1`, so a console
session defaulting elsewhere shows an empty list that reads as a broken
dashboard. And `generate-embed-url-for-registered-user` returns a URL carrying a
one-time auth code: the first request consumes it, every later request returns
403.

We wrote a helper that verified the URL with `curl` before printing it, which
consumed the code and handed over a dead link. Measured: request 1 -> 200,
requests 2 and 3 -> 403. The check had to be removed entirely -- some things
cannot be pre-flighted without destroying them.

**Lesson:** `--session-lifetime-in-minutes` governs the session *after* the first
open, not how long the link stays clickable.

## What we would do differently

1. **Declare API Gateway methods explicitly from the beginning.** The `ANY`
   shortcut cost more than it saved once a browser was involved.
2. **Check `get-foundation-model-availability` before writing any Bedrock code.**
   We wrote and deployed the whole narration layer before discovering access was
   gated.
3. **Commit and push far more often.** A collaborator's merge, based on an older
   `src/`, reverted our corrected-data fixes — including a `HOUR_OFFSET` value
   that was wrong for the reissued data and a path bug that crashed the pipeline
   on startup. Our work survived only because GitHub Desktop stashed it. Frequent
   small commits would have turned a recovery exercise into a routine merge.
4. **Pin the Python environment inside the repo from day one.** Ours lived
   outside the project directory, which quietly undermined the reproducibility we
   were claiming until we added `requirements.txt`.

---

## The architectural decision we would keep

Keeping the language model off the numbers. Every forecast and every ranking is
computed deterministically; Bedrock converts a question into parameters and a
result into prose, and a numeric-containment guardrail rejects any generated text
containing a figure absent from the computation.

That guardrail caught a real error on its first production call — a narrative
mentioning the year "2026", which was legitimate context we had not allow-listed.
The same mechanism rejects fabricated volumes and invented percentages, verified
by unit tests.

It means we can hand a traffic engineer a recommendation and, one click later,
the arithmetic behind it.
