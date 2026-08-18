"""
Bedrock Narrative Layer
=======================
Turns the deterministic factor decomposition produced by the forecasting model
into a plain-English paragraph, using Amazon Bedrock (Claude).

DESIGN CONTRACT
---------------
The language model NEVER produces a number. It only rephrases values that the
multiplicative model already computed. This preserves the auditability that makes
the forecast defensible: every figure in the narrative traces to a factor in the
decomposition.

Two guardrails enforce that:

1. Numeric containment. After generation, every number in the narrative is
   extracted and matched against the values supplied in the prompt (allowing for
   rounding and for percentage restatements of a multiplicative factor). If the
   model introduces an unsupported figure, the narrative is REJECTED.

2. Graceful degradation. Any failure — throttling, access denial, timeout, a
   failed containment check — returns the decomposition without a narrative.
   `/explain` can therefore never be broken by this layer.

Model selection: Haiku by default for cost. NARRATIVE_MODEL_ID overrides it, and
a fallback chain is attempted so a single denied model does not disable the
feature.
"""

import json
import os
import re
from datetime import datetime

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

# Default to Haiku (cheapest adequate model for constrained summarisation).
# Cross-region inference profile, required for claude-haiku-4-5.
PRIMARY_MODEL = os.environ.get(
    'NARRATIVE_MODEL_ID', 'us.anthropic.claude-haiku-4-5-20251001-v1:0')
FALLBACK_MODELS = [
    m for m in os.environ.get('NARRATIVE_FALLBACK_MODELS', '').split(',') if m
] or ['us.anthropic.claude-sonnet-4-6']

MAX_TOKENS = int(os.environ.get('NARRATIVE_MAX_TOKENS', '320'))

_client = None


def _bedrock():
    """Lazily create the Bedrock runtime client with tight timeouts."""
    global _client
    if _client is None:
        _client = boto3.client(
            'bedrock-runtime',
            region_name=os.environ.get('BEDROCK_REGION', 'us-east-1'),
            config=Config(
                connect_timeout=3,
                read_timeout=12,
                retries={'max_attempts': 2, 'mode': 'standard'},
            ),
        )
    return _client


# ----------------------------------------------------------------------
# Prompt construction
# ----------------------------------------------------------------------
SYSTEM_PROMPT = """You explain traffic volume forecasts to transport agency staff.

ABSOLUTE RULES:
1. Use ONLY the numbers given in the input. Never calculate, infer, or invent a
   number. If a value is not supplied, do not mention it.
2. Do not restate every factor mechanically. Explain what drives the result.
3. State plainly when a forecast rests on estimated rather than observed history.
4. The reliability score is a RELATIVE confidence indicator, not a probability.
   The 90% interval is what carries the statistical meaning.
5. Plain prose. No markdown, no bullet points, no headings. 3 to 5 sentences.
6. Round volumes to whole vehicles when you mention them.

Write for a traffic engineer who wants to know what to expect and how much to
trust it."""


def _build_prompt(payload):
    """Render the decomposition as a compact, unambiguous prompt."""
    p = payload['prediction']
    e = payload['explanation']

    lines = [
        f"Station {p['station']}, direction {p['direction']}, {p['timestamp']}.",
        f"Forecast: {p['forecast_volume']} vehicles/hour.",
        f"90% interval: {p['lower_90']} to {p['upper_90']}.",
        f"Reliability score: {p['reliability_score']} (relative indicator, 0 to 1).",
        f"Prediction basis: {e['prediction_path']}.",
        "",
        "How the forecast was built (each factor multiplies the running total):",
    ]
    for s in e['decomposition']:
        lines.append(
            f"  - {s['factor']} = {s['value']} -> running total {s['running_total']}"
            f"   [{s['meaning']}]")

    rel = e.get('reliability', {})
    if rel.get('adjustments'):
        lines.append("")
        lines.append("Reliability adjustments applied:")
        for a in rel['adjustments']:
            lines.append(f"  - {a['reason']}: {a['delta']:+}")

    if e.get('is_cold_start'):
        lines.append("")
        lines.append("NOTE: this station has no observed history. Its baseline and "
                     "profiles are estimated from neighbouring stations, so the "
                     "interval is deliberately wider and reliability lower.")

    return "\n".join(lines)


# ----------------------------------------------------------------------
# Guardrail: numeric containment
# ----------------------------------------------------------------------
# A leading minus only counts as a sign when it is not preceded by a digit or
# word character. Otherwise a range such as "389-1188" or "08:00-20:00" is
# misread as containing -1188 / -20, causing spurious guardrail rejections.
_NUM_RE = re.compile(r'(?<![\w\d])-?\d[\d,]*\.?\d*')

# Small integers are allowed as ordinary prose ("three factors", "9 times out of
# 10", clock hours). Restricting the check to meaningful magnitudes avoids
# rejecting good narratives over incidental digits.
_ALLOWED_SMALL_INTS = set(range(0, 101))


def _numbers_in(text):
    out = []
    for raw in _NUM_RE.findall(text):
        try:
            out.append(float(raw.replace(',', '')))
        except ValueError:
            continue
    return out


def _supported_values(payload):
    """Every numeric value the model is permitted to reproduce."""
    p = payload['prediction']
    e = payload['explanation']
    vals = {
        float(p['forecast_volume']), float(p['lower_90']),
        float(p['upper_90']), float(p['reliability_score']),
    }

    # Date/time components of the requested timestamp are legitimate context:
    # the model may naturally write "September 2026" or "16:00".
    ts = str(p.get('timestamp', ''))
    for part in re.findall(r'\d+', ts):
        try:
            vals.add(float(part))
        except ValueError:
            pass
    # Station identifiers such as STN_0067 may be quoted verbatim.
    for part in re.findall(r'\d+', str(p.get('station', ''))):
        try:
            vals.add(float(part))
            vals.add(float(int(part)))     # 0067 -> 67
        except ValueError:
            pass

    for s in e['decomposition']:
        vals.add(float(s['value']))
        vals.add(float(s['running_total']))
        # a factor of 1.8957 may legitimately be described as "90%" higher, or
        # "1.9 times"; permit the percentage restatement of any factor
        v = float(s['value'])
        vals.add(round(abs(v - 1.0) * 100, 1))
        vals.add(round(v * 100, 1))
    for a in e.get('reliability', {}).get('adjustments', []) or []:
        vals.add(abs(float(a['delta'])))
        vals.add(round(abs(float(a['delta'])) * 100, 1))
    if 'base_score' in e.get('reliability', {}):
        vals.add(float(e['reliability']['base_score']))
    if 'coefficient_of_variation' in e.get('uncertainty', {}):
        vals.add(float(e['uncertainty']['coefficient_of_variation']))
        vals.add(round(float(e['uncertainty']['coefficient_of_variation']) * 100, 1))
    return vals


def check_containment(text, payload, tolerance=0.02):
    """
    Verify every number in `text` traces to a supplied value.

    Returns (ok: bool, unsupported: list[float]).
    """
    return check_numbers(text, _supported_values(payload), tolerance)


def collect_numbers(obj, acc=None):
    """
    Recursively gather every numeric value in an arbitrary JSON-like structure.

    Used to build the allow-list when narrating a planning result, whose shape
    differs from a forecast decomposition.
    """
    if acc is None:
        acc = set()
    if isinstance(obj, bool):
        return acc
    if isinstance(obj, (int, float)):
        acc.add(float(obj))
        # A ratio is naturally read aloud as a percentage: a utilisation of
        # 0.862 becomes "86.2%", and an over-capacity 1.018 becomes "101.8%".
        # Permit that restatement for plausible ratios. This only ever licences
        # x100 of a value that genuinely appears in the result, so it cannot
        # licence an arbitrary figure.
        if 0.0 <= float(obj) <= 10.0:
            acc.add(round(float(obj) * 100, 1))
        return acc
    if isinstance(obj, str):
        for part in re.findall(r'\d+\.?\d*', obj):
            try:
                acc.add(float(part))
                if '.' not in part:
                    acc.add(float(int(part)))     # "07" -> 7
            except ValueError:
                pass
        return acc
    if isinstance(obj, dict):
        for k, v in obj.items():
            collect_numbers(k, acc)
            collect_numbers(v, acc)
        return acc
    if isinstance(obj, (list, tuple, set)):
        for v in obj:
            collect_numbers(v, acc)
        return acc
    return acc


def check_numbers(text, supported, tolerance=0.02):
    """
    Core containment check against an explicit set of permitted values.

    Returns (ok: bool, unsupported: list[float]).
    """
    unsupported = []
    for n in _numbers_in(text):
        if n.is_integer() and int(n) in _ALLOWED_SMALL_INTS:
            continue
        if any(abs(n - s) <= max(tolerance, abs(s) * tolerance) for s in supported):
            continue
        # tolerate rounding of large values to the nearest ten or hundred
        if any(abs(n - round(s, -1)) < 1e-6 or abs(n - round(s, -2)) < 1e-6
               for s in supported):
            continue
        unsupported.append(n)
    return (not unsupported), unsupported


# ----------------------------------------------------------------------
# Public entry point
# ----------------------------------------------------------------------
def generate_narrative(payload):
    """
    Produce a plain-English narrative for a decomposition payload.

    Returns a dict describing the outcome. Never raises: a failure yields
    {'status': ..., 'narrative': None} so the caller can degrade gracefully.
    """
    prompt = _build_prompt(payload)
    body = {
        'anthropic_version': 'bedrock-2023-05-31',
        'max_tokens': MAX_TOKENS,
        'temperature': 0.2,
        'system': SYSTEM_PROMPT,
        'messages': [{'role': 'user', 'content': prompt}],
    }

    attempted = []
    for model_id in [PRIMARY_MODEL] + FALLBACK_MODELS:
        attempted.append(model_id)
        try:
            resp = _bedrock().invoke_model(
                modelId=model_id, body=json.dumps(body))
            data = json.loads(resp['body'].read())
            text = ''.join(
                c.get('text', '') for c in data.get('content', [])).strip()
            usage = data.get('usage', {}) or {}

            ok, unsupported = check_containment(text, payload)
            if not ok:
                print(f"NARRATIVE REJECTED (unsupported numbers {unsupported}) "
                      f"model={model_id}")
                return {
                    'status': 'rejected_unsupported_numbers',
                    'narrative': None,
                    'model_id': model_id,
                    'unsupported_numbers': unsupported,
                    'input_tokens': usage.get('input_tokens'),
                    'output_tokens': usage.get('output_tokens'),
                    'guardrail': 'numeric_containment',
                }

            return {
                'status': 'ok',
                'narrative': text,
                'model_id': model_id,
                'input_tokens': usage.get('input_tokens'),
                'output_tokens': usage.get('output_tokens'),
                'guardrail': 'numeric_containment_passed',
            }

        except ClientError as err:
            code = err.response.get('Error', {}).get('Code', 'Unknown')
            print(f"Bedrock {code} for {model_id}: {err}")
            if code in ('AccessDeniedException', 'ValidationException',
                        'ResourceNotFoundException'):
                continue          # try the next model
            return {'status': f'error_{code}', 'narrative': None,
                    'models_attempted': attempted}
        except Exception as err:                     # noqa: BLE001
            print(f"Narrative generation failed for {model_id}: {err}")
            return {'status': 'error_unexpected', 'narrative': None,
                    'models_attempted': attempted}

    return {'status': 'error_all_models_unavailable', 'narrative': None,
            'models_attempted': attempted}


# ======================================================================
# Conversational layer
# ======================================================================
# Two separate model calls, deliberately:
#   1. parse_question  -> structured intent (validated before use)
#   2. narrate_answer  -> prose for a result computed deterministically
#
# The model never selects the answer. It converts a sentence into parameters,
# a deterministic tool computes the result, and the model then describes it.

INTENT_SYSTEM = """You convert a traffic-planning question into JSON parameters.

Return ONLY a JSON object, no prose, no code fences.

Schema:
{
  "intent": "quietest_hours" | "lowest_impact_window" | "closure_impact" | "single_forecast" | "unsupported",
  "station": "STN_0067" or null,
  "direction": "1" | "3" | "5" | "7" or null,
  "date": "YYYY-MM-DD" or null,
  "start": "YYYY-MM-DD" or null,
  "end": "YYYY-MM-DD" or null,
  "hour": 0-23 or null,
  "duration_hours": integer or null,
  "lanes_closed": integer or null,
  "align": "day" | "hour" or null,
  "earliest": 0-23 or null,
  "latest": 0-23 or null,
  "note": "short clarification if you had to assume something, else null"
}

Rules:
- "best/quietest time to travel" on one date  -> quietest_hours, set date.
- "best day/window for a closure, roadworks, maintenance" -> lowest_impact_window,
  set start and end. Use align "day" for "which day", "hour" for "which window".
  Default duration_hours to 24 for a day, 72 for "3 days", and so on.
- "what happens if / impact of closing X", "effect on the network", or a closure
  with a stated start -> closure_impact. Set start and duration_hours, and
  lanes_closed if a partial closure is described ("two of three lanes").
  Use lowest_impact_window instead when the question asks WHICH day or window.
- A specific station and hour with no ranking implied -> single_forecast.
- Resolve relative dates ("tomorrow", "next week") against TODAY given below.
  A week runs Monday to Sunday.
- If no station is named, leave station null.
- If the question is not about traffic forecasting, use intent "unsupported".
- Never invent a station id that was not mentioned."""

ANSWER_SYSTEM = """You report the result of a traffic-planning calculation to
agency staff.

ABSOLUTE RULES:
1. Use ONLY numbers present in the RESULT. Never calculate or invent a number.
2. The recommendation has already been computed. Report it; do not re-rank or
   second-guess it.
3. Lead with the answer, then the supporting figures.
4. For closure or roadworks questions, state that this is an analytical estimate
   for planning comparison, not an engineering-grade operational recommendation.
   If the result names a binding_constraint, say which neighbouring station it is
   and what utilisation it reaches. If capacity_caveats lists floored stations,
   note that their utilisation is a lower bound.
5. Lower forecast volume means lighter traffic; it is not a guarantee of a
   shorter journey.
6. Plain prose. No markdown, no bullets, no headings. 2 to 5 sentences."""


def parse_question(question, today_iso, known_stations=None):
    """
    Turn a natural-language question into validated parameters.

    Returns (params: dict | None, error: str | None). Validation is applied to
    the model's output, so a malformed or hallucinated field is rejected here
    rather than reaching the planning tools.
    """
    prompt = (f"TODAY is {today_iso}.\n\nQuestion: {question.strip()}\n\n"
              "Return the JSON object only.")
    body = {
        'anthropic_version': 'bedrock-2023-05-31',
        'max_tokens': 400,
        'temperature': 0.0,
        'system': INTENT_SYSTEM,
        'messages': [{'role': 'user', 'content': prompt}],
    }

    raw = None
    for model_id in [PRIMARY_MODEL] + FALLBACK_MODELS:
        try:
            resp = _bedrock().invoke_model(modelId=model_id, body=json.dumps(body))
            data = json.loads(resp['body'].read())
            raw = ''.join(c.get('text', '') for c in data.get('content', [])).strip()
            break
        except ClientError as err:
            code = err.response.get('Error', {}).get('Code', 'Unknown')
            print(f"Intent parse {code} for {model_id}: {err}")
            if code in ('AccessDeniedException', 'ValidationException',
                        'ResourceNotFoundException'):
                continue
            return None, f'intent_parse_error_{code}'
        except Exception as err:                          # noqa: BLE001
            print(f"Intent parse failed for {model_id}: {err}")
            return None, 'intent_parse_error'

    if raw is None:
        return None, 'intent_parse_no_model_available'

    # Strip code fences if the model added them despite instructions
    raw = re.sub(r'^```(?:json)?|```$', '', raw.strip(), flags=re.M).strip()
    try:
        params = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r'\{.*\}', raw, re.S)
        if not m:
            return None, 'intent_not_json'
        try:
            params = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None, 'intent_not_json'

    if not isinstance(params, dict):
        return None, 'intent_not_object'

    intent = params.get('intent')
    if intent not in ('quietest_hours', 'lowest_impact_window', 'closure_impact',
                      'single_forecast', 'unsupported'):
        return None, 'intent_unrecognised'

    # ---- validate every field the model may have supplied ----
    def _as_int(key, lo, hi):
        v = params.get(key)
        if v is None:
            return None
        try:
            iv = int(v)
        except (TypeError, ValueError):
            raise ValueError(f"{key} must be an integer")
        if not (lo <= iv <= hi):
            raise ValueError(f"{key} must be between {lo} and {hi}")
        return iv

    try:
        for key in ('date', 'start', 'end'):
            v = params.get(key)
            if v is not None:
                datetime.strptime(str(v)[:10], '%Y-%m-%d')
        params['hour'] = _as_int('hour', 0, 23)
        params['earliest'] = _as_int('earliest', 0, 23)
        params['latest'] = _as_int('latest', 0, 23)
        params['duration_hours'] = _as_int('duration_hours', 1, 24 * 14)
        params['lanes_closed'] = _as_int('lanes_closed', 1, 12)
    except ValueError as err:
        return None, f'intent_invalid_field: {err}'

    if params.get('align') not in (None, 'day', 'hour'):
        return None, 'intent_invalid_align'
    if params.get('direction') is not None:
        params['direction'] = str(params['direction'])
        if params['direction'] not in ('1', '3', '5', '7'):
            return None, 'intent_invalid_direction'

    station = params.get('station')
    if station is not None:
        station = str(station).strip().upper()
        if not re.fullmatch(r'STN_\d{4}', station):
            return None, f'intent_invalid_station_format: {station}'
        if known_stations and station not in known_stations:
            return None, f'unknown_station: {station}'
        params['station'] = station

    return params, None


def narrate_answer(question, result):
    """
    Describe a deterministically-computed planning result.

    The allow-list is every number appearing anywhere in `result`, so the model
    can quote any computed figure but cannot introduce a new one.
    """
    prompt = (f"Question: {question.strip()}\n\n"
              f"RESULT (already computed, authoritative):\n"
              f"{json.dumps(result, indent=2, default=str)}")
    body = {
        'anthropic_version': 'bedrock-2023-05-31',
        'max_tokens': MAX_TOKENS,
        'temperature': 0.2,
        'system': ANSWER_SYSTEM,
        'messages': [{'role': 'user', 'content': prompt}],
    }

    allowed = collect_numbers(result)
    attempted = []
    for model_id in [PRIMARY_MODEL] + FALLBACK_MODELS:
        attempted.append(model_id)
        try:
            resp = _bedrock().invoke_model(modelId=model_id, body=json.dumps(body))
            data = json.loads(resp['body'].read())
            text = ''.join(c.get('text', '') for c in data.get('content', [])).strip()
            usage = data.get('usage', {}) or {}

            ok, unsupported = check_numbers(text, allowed)
            if not ok:
                print(f"ANSWER REJECTED (unsupported numbers {unsupported}) "
                      f"model={model_id}")
                return {'status': 'rejected_unsupported_numbers', 'narrative': None,
                        'model_id': model_id, 'unsupported_numbers': unsupported,
                        'input_tokens': usage.get('input_tokens'),
                        'output_tokens': usage.get('output_tokens'),
                        'guardrail': 'numeric_containment'}

            return {'status': 'ok', 'narrative': text, 'model_id': model_id,
                    'input_tokens': usage.get('input_tokens'),
                    'output_tokens': usage.get('output_tokens'),
                    'guardrail': 'numeric_containment_passed'}

        except ClientError as err:
            code = err.response.get('Error', {}).get('Code', 'Unknown')
            print(f"Answer narration {code} for {model_id}: {err}")
            if code in ('AccessDeniedException', 'ValidationException',
                        'ResourceNotFoundException'):
                continue
            return {'status': f'error_{code}', 'narrative': None,
                    'models_attempted': attempted}
        except Exception as err:                          # noqa: BLE001
            print(f"Answer narration failed for {model_id}: {err}")
            return {'status': 'error_unexpected', 'narrative': None,
                    'models_attempted': attempted}

    return {'status': 'error_all_models_unavailable', 'narrative': None,
            'models_attempted': attempted}
