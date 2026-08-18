#!/usr/bin/env bash
#
# Deploy the model artifact and Lambda code to AWS, then refresh the dashboard.
#
# Prerequisites:
#   - Valid AWS credentials (workshop tokens expire; re-authenticate first)
#   - python src/cold_start_enhance.py && python src/build_artifact.py <version>
#
# Usage:
#   bash scripts/deploy.sh 2.2
#
set -euo pipefail

VERSION="${1:-2.2}"
REGION="us-east-1"
ACCOUNT="062905933333"
BUCKET="deldot-traffic-forecasting-${ACCOUNT}"
DATASET="deldot-rolling-forecast-dataset"
API_LAMBDA="deldot-traffic-forecast"
BATCH_LAMBDA="deldot-batch-forecast"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARTIFACT="${REPO_ROOT}/output/model_artifact_v${VERSION}.json"
KEY="models/v${VERSION}/model_artifact.json"

echo "=============================================="
echo " Deploying model v${VERSION}"
echo "=============================================="

# 0. Resolve working credentials.
#
# Trap this guards against: expired AWS_* environment variables take precedence
# over a freshly refreshed `aws login` session or a --profile credentials file,
# so the CLI keeps failing with ExpiredToken even after you re-authenticate.
# We try the default chain first, then retry with the env vars stripped.
#
# Implementation note: dispatch via a redefined function rather than an array.
# macOS ships bash 3.2, where "${arr[@]}" on an EMPTY array under `set -u`
# raises "unbound variable". Functions avoid that entirely.
# Detection below must call the real `aws`, not the wrapper, which is why the
# wrapper is only defined afterwards.
if aws sts get-caller-identity --region "$REGION" >/dev/null 2>&1; then
  aws_() { command aws "$@"; }
  echo "Credentials OK (default chain)"
elif env -u AWS_ACCESS_KEY_ID -u AWS_SECRET_ACCESS_KEY -u AWS_SESSION_TOKEN \
     aws sts get-caller-identity --region "$REGION" >/dev/null 2>&1; then
  aws_() {
    env -u AWS_ACCESS_KEY_ID -u AWS_SECRET_ACCESS_KEY -u AWS_SESSION_TOKEN \
        aws "$@"
  }
  echo "Credentials OK (stale AWS_* env vars ignored)"
else
  cat >&2 <<'MSG'
ERROR: no usable AWS credentials.

Both paths failed:
  - AWS_* environment variables      -> expired
  - credentials file / login session -> expired or absent

Fix with ONE of:
  aws login                                        # refreshes ~/.aws/login cache
  aws configure set aws_access_key_id     <key>    --profile deldot
  aws configure set aws_secret_access_key <secret> --profile deldot
  aws configure set aws_session_token     <token>  --profile deldot
  aws configure set region us-east-1               --profile deldot
  # then re-run with: AWS_PROFILE=deldot bash scripts/deploy.sh <version>
MSG
  exit 1
fi

echo "Identity: $(aws_ sts get-caller-identity --region "$REGION" --query Arn --output text)"

[ -f "$ARTIFACT" ] || { echo "ERROR: missing $ARTIFACT (run build_artifact.py)" >&2; exit 1; }

# 1. Upload the model artifact
echo ""
echo "--- Uploading artifact ---"
aws_ s3 cp "$ARTIFACT" "s3://${BUCKET}/${KEY}" --region "$REGION"

# 2. Package and deploy Lambda code
echo ""
echo "--- Deploying Lambda code ---"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
( cd "${REPO_ROOT}/src" && zip -q "${TMP}/api.zip" lambda_function.py )
( cd "${REPO_ROOT}/src" && zip -q "${TMP}/batch.zip" lambda_batch_forecast.py )

aws_ lambda update-function-code --region "$REGION" \
  --function-name "$API_LAMBDA" --zip-file "fileb://${TMP}/api.zip" \
  --query 'FunctionName' --output text
aws_ lambda update-function-code --region "$REGION" \
  --function-name "$BATCH_LAMBDA" --zip-file "fileb://${TMP}/batch.zip" \
  --query 'FunctionName' --output text

aws_ lambda wait function-updated --region "$REGION" --function-name "$API_LAMBDA"
aws_ lambda wait function-updated --region "$REGION" --function-name "$BATCH_LAMBDA"

# 3. Point both functions at the new artifact (preserving other env vars)
echo ""
echo "--- Updating MODEL_KEY -> ${KEY} ---"
for FN in "$API_LAMBDA" "$BATCH_LAMBDA"; do
  ENV_JSON="$(aws_ lambda get-function-configuration --region "$REGION" \
      --function-name "$FN" --query 'Environment.Variables' --output json)"
  NEW_ENV="$(python3 -c "
import json, sys
env = json.loads(sys.argv[1])
env['MODEL_KEY'] = sys.argv[2]
print(json.dumps({'Variables': env}))
" "$ENV_JSON" "$KEY")"
  echo "$NEW_ENV" > "${TMP}/env_${FN}.json"
  aws_ lambda update-function-configuration --region "$REGION" \
    --function-name "$FN" --environment "file://${TMP}/env_${FN}.json" \
    --query '{Function:FunctionName,ModelKey:Environment.Variables.MODEL_KEY}'
  aws_ lambda wait function-updated --region "$REGION" --function-name "$FN"
done

# 4. Regenerate the rolling forecast that feeds the dashboard
echo ""
echo "--- Regenerating 30-day rolling forecast ---"
aws_ lambda invoke --region "$REGION" --function-name "$BATCH_LAMBDA" \
  --cli-binary-format raw-in-base64-out --payload '{}' "${TMP}/batch_out.json" \
  --query 'StatusCode' --output text
python3 -c "
import json, sys
print(json.loads(json.load(open(sys.argv[1]))['body'])) " "${TMP}/batch_out.json"

# 5. Refresh QuickSight SPICE
echo ""
echo "--- Refreshing QuickSight SPICE ---"
# QuickSight IngestionId allows only alphanumerics and hyphens, so "2.2" -> "2-2".
VERSION_SAFE="$(printf '%s' "$VERSION" | tr -c '[:alnum:]-' '-')"
INGESTION="deploy-v${VERSION_SAFE}-$(date +%s)"
aws_ quicksight create-ingestion --region "$REGION" \
  --aws-account-id "$ACCOUNT" --data-set-id "$DATASET" \
  --ingestion-id "$INGESTION" --query 'IngestionStatus' --output text

echo "Waiting for ingestion..."
for _ in $(seq 1 30); do
  STATUS="$(aws_ quicksight describe-ingestion --region "$REGION" \
      --aws-account-id "$ACCOUNT" --data-set-id "$DATASET" \
      --ingestion-id "$INGESTION" --query 'Ingestion.IngestionStatus' --output text)"
  [ "$STATUS" = "COMPLETED" ] && break
  [ "$STATUS" = "FAILED" ] && { echo "Ingestion FAILED" >&2; exit 1; }
  sleep 5
done
aws_ quicksight describe-ingestion --region "$REGION" \
  --aws-account-id "$ACCOUNT" --data-set-id "$DATASET" \
  --ingestion-id "$INGESTION" \
  --query 'Ingestion.{Status:IngestionStatus,Rows:RowInfo.RowsIngested,Dropped:RowInfo.RowsDropped}'

# 6. Verify the API reports the new version and that auth is enforced
echo ""
echo "--- Verifying deployment ---"
API_URL="https://94d3hvwu93.execute-api.${REGION}.amazonaws.com/prod"
NOKEY="$(curl -s -o /dev/null -w '%{http_code}' "${API_URL}/health")"
echo "  /health without API key -> HTTP ${NOKEY} (expect 403)"
if [ -n "${DELDOT_API_KEY:-}" ]; then
  curl -s -H "x-api-key: ${DELDOT_API_KEY}" "${API_URL}/health"
  echo ""
else
  echo "  Set DELDOT_API_KEY to verify the authenticated path."
fi

echo ""
echo "Deploy of v${VERSION} complete."
