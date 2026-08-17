# Submission Schema

## Forecast file

Start from `submission_template.csv`.

Required columns:

| Field | Requirement |
|---|---|
| `target_id` | Preserve exactly as supplied |
| `station_key` | Preserve exactly as supplied |
| `direction_code` | Preserve exactly as supplied |
| `timestamp` | Preserve exactly as supplied |
| `forecast_volume` | Numeric point forecast |
| `lower_90` | Lower bound of claimed 90% interval |
| `upper_90` | Upper bound of claimed 90% interval |
| `reliability_score` | Numeric value from 0 through 1 |

Rules:

- Do not add, remove, duplicate, or reorder target rows unless instructed.
- `lower_90 <= forecast_volume <= upper_90`.
- `0 <= reliability_score <= 1`.
- Forecast fields must be populated for all scored rows.
- Do not use information that would only become available after the forecast timestamp.

## Scenario file

Use `scenario_template.csv`.

For each supplied scenario, report:
- estimated traffic impact;
- lower and upper uncertainty bounds;
- recovery time where relevant;
- reliability;
- concise assumptions.

Scenario outputs are analytical estimates and must not be represented as
engineering-grade operational predictions.
