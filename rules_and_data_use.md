# Rules and Data Use

These rules apply to the supplied traffic forecasting materials.

## Prediction-time information

Use only information that would reasonably be available at or before the time
being predicted.

Do not use future observations, target values, or event outcomes that would only
be known after the prediction time.

## Anonymous station identifiers

Use the supplied `station_key` values as the authoritative station identifiers.

Do not use external sources, public archives, APIs, source-system identifiers,
or pattern matching to identify the underlying stations or recover withheld
traffic observations.

## Internet and external references

General technical references, software documentation, transportation research,
open-source libraries, and approved AI tools may be used.

External traffic datasets may not be used to reconstruct anonymous station
identities, retrieve withheld target values, or recreate restricted evaluation
material.

If use of an external dataset is unclear, request a ruling before incorporating it.

## AI-assisted work

Approved AI coding tools and assistants may be used. Users remain responsible
for understanding the submitted work, checking calculations, preventing target
leakage, and ensuring reproducibility.

## Synthetic or augmented data

Synthetic or augmented data may be created from the supplied data for model
development. The method should be reproducible and must not incorporate
externally reconstructed ground truth.

## Forecast file requirements

Start from `submission_template.csv`.

Do not alter:
- `target_id`
- `station_key`
- `direction_code`
- `timestamp`

Populate:
- `forecast_volume`
- `lower_90`
- `upper_90`
- `reliability_score`

Requirements:
- every target row must be present;
- target rows must not be duplicated;
- output fields must contain finite numeric values;
- `forecast_volume >= 0`;
- `lower_90 >= 0`;
- `lower_90 <= forecast_volume <= upper_90`;
- `0 <= reliability_score <= 1`.

Run `validate_submission.py` before delivery.

## Scenario analysis

Scenario results should include estimated impact, uncertainty, assumptions,
recovery time where relevant, and a reliability assessment.

Scenario results are analytical estimates and must not be represented as
engineering-grade operational, safety, capacity, or design recommendations.

## Reproducibility

Document the major dependencies, preprocessing steps, model-development
procedure, random seeds where practical, and any external general-purpose
resources used.

## Restricted evaluation material

If restricted evaluation files, hidden target values, private station mappings,
or other non-public evaluation information are received accidentally, stop
using the material and notify the designated contact.

## Clarifications

Questions about file structure, terminology, modeling concepts, and submission
format may be clarified by mentors.

Questions that could affect permitted data use, scoring, or fairness should be
referred to the designated evaluation lead so the same interpretation can be
applied consistently.
