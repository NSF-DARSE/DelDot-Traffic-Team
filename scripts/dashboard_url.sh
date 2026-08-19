#!/usr/bin/env bash
#
# Print a ready-to-open QuickSight dashboard URL.
#
# Why this exists: the dashboard lives in us-east-1, but a console session
# defaulting to another region shows an empty dashboard list, which looks like a
# broken dashboard and is not. This generates a signed embed URL that opens the
# dashboard directly, independent of whatever region the console is pointing at.
#
# Usage:
#   bash scripts/dashboard_url.sh              # 600-minute session (max)
#   bash scripts/dashboard_url.sh 120          # shorter session
#   AWS_PROFILE=deldot bash scripts/dashboard_url.sh
#
set -euo pipefail

MINUTES="${1:-600}"
REGION="us-east-1"
ACCOUNT="062905933333"
DASHBOARD="deldot-traffic-dashboard-v3"
USER_ARN="arn:aws:quicksight:${REGION}:${ACCOUNT}:user/default/WSParticipantRole/Participant"

# Expired AWS_* environment variables take precedence over a valid profile or
# login session, so fall back to stripping them if the default chain fails.
if aws sts get-caller-identity --region "$REGION" >/dev/null 2>&1; then
  aws_() { command aws "$@"; }
elif env -u AWS_ACCESS_KEY_ID -u AWS_SECRET_ACCESS_KEY -u AWS_SESSION_TOKEN \
     aws sts get-caller-identity --region "$REGION" >/dev/null 2>&1; then
  aws_() { env -u AWS_ACCESS_KEY_ID -u AWS_SECRET_ACCESS_KEY -u AWS_SESSION_TOKEN aws "$@"; }
else
  cat >&2 <<'MSG'
ERROR: no usable AWS credentials.
  aws login
  # or: AWS_PROFILE=<profile> bash scripts/dashboard_url.sh
MSG
  exit 1
fi

echo "Identity: $(aws_ sts get-caller-identity --region "$REGION" --query Arn --output text)" >&2

# Confirm the dashboard is actually there before minting a URL for it.
if ! aws_ quicksight describe-dashboard --region "$REGION" \
      --aws-account-id "$ACCOUNT" --dashboard-id "$DASHBOARD" >/dev/null 2>&1; then
  echo "ERROR: dashboard $DASHBOARD not found in $REGION." >&2
  echo "       Dashboards in $REGION:" >&2
  aws_ quicksight list-dashboards --region "$REGION" --aws-account-id "$ACCOUNT" \
    --query 'DashboardSummaryList[].DashboardId' --output text >&2
  exit 1
fi

URL="$(aws_ quicksight generate-embed-url-for-registered-user \
  --region "$REGION" --aws-account-id "$ACCOUNT" --user-arn "$USER_ARN" \
  --experience-configuration "{\"Dashboard\":{\"InitialDashboardId\":\"${DASHBOARD}\"}}" \
  --session-lifetime-in-minutes "$MINUTES" \
  --query EmbedUrl --output text)"

# Deliberately NOT verified with curl. The URL carries a ONE-TIME auth code
# (isauthcode=true): the first request consumes it and every later request gets
# 403. A pre-flight check would therefore hand over a dead link. Verified
# empirically: request 1 -> 200, requests 2 and 3 -> 403.
echo "Session: ${MINUTES} minutes from first open." >&2
echo "NOTE: single use. Opening it consumes the auth code -- rerun this script" >&2
echo "      for another link. Do not 'test' it in one tab and demo in another." >&2
echo "" >&2
echo "$URL"
