#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# deploy_to_hfspace.sh
# Push the Gradio app to HuggingFace Spaces.
#
# Usage:
#   ./deploy_to_hfspace.sh              # push deploy/ as-is
#   ./deploy_to_hfspace.sh --sync       # also sync helper scripts from scripts/
#   ./deploy_to_hfspace.sh -m "msg"     # custom commit message
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_DIR="$ROOT/deploy"
SCRIPTS_DIR="$ROOT/scripts"
HF_REPO="kvsnoufal/gemma-abliteration-study"
PYTHON="${PYTHON:-python}"  # Use 'python' by default or override with PYTHON env var

# ── Parse args ────────────────────────────────────────────────────────────────
SYNC=false
COMMIT_MSG="Update Gradio app"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --sync|-s)   SYNC=true; shift ;;
        -m)          COMMIT_MSG="$2"; shift 2 ;;
        *)           echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ── Load HF_TOKEN ─────────────────────────────────────────────────────────────
if [[ -f "$SCRIPTS_DIR/.env" ]]; then
    # shellcheck disable=SC2046
    export $(grep -v '^#' "$SCRIPTS_DIR/.env" | grep HF_TOKEN | xargs)
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "❌  HF_TOKEN not found. Add it to scripts/.env or export it first."
    exit 1
fi

# ── Optionally sync helper scripts ────────────────────────────────────────────
# These files are identical between local and deploy.
# deploy/app.py is NOT synced automatically because it has Space-specific
# changes (no local model path, no .env loading, no server_name/port in launch).
# Edit deploy/app.py directly when changing UI/CSS/logic.
if [[ "$SYNC" == true ]]; then
    echo "🔄  Syncing helper scripts from scripts/ → deploy/"
    cp "$SCRIPTS_DIR/config.py"    "$DEPLOY_DIR/config.py"
    cp "$SCRIPTS_DIR/evaluate.py"  "$DEPLOY_DIR/evaluate.py"
    cp "$SCRIPTS_DIR/abliterate.py" "$DEPLOY_DIR/abliterate.py"
    echo "    config.py, evaluate.py, abliterate.py updated"
fi

# ── Show what will be pushed ──────────────────────────────────────────────────
echo ""
echo "📦  Files to push from: $DEPLOY_DIR"
ls -lh "$DEPLOY_DIR" "$DEPLOY_DIR/cache" 2>/dev/null | grep -v '^total' | awk '{print "    " $NF, $5}'
echo ""
echo "🚀  Target: https://huggingface.co/spaces/$HF_REPO"
echo "📝  Commit: $COMMIT_MSG"
echo ""

# ── Confirm ───────────────────────────────────────────────────────────────────
read -r -p "Push now? [y/N] " confirm
if [ "$(echo "$confirm" | tr '[:upper:]' '[:lower:]')" != "y" ]; then
    echo "Aborted."
    exit 0
fi

# ── Push ──────────────────────────────────────────────────────────────────────
echo ""
echo "⬆️   Uploading..."

HF_TOKEN="$HF_TOKEN" "$PYTHON" - <<PYEOF
import os, sys
from huggingface_hub import HfApi

api = HfApi(token=os.environ["HF_TOKEN"])
url = api.upload_folder(
    folder_path="$DEPLOY_DIR",
    repo_id="$HF_REPO",
    repo_type="space",
    commit_message="$COMMIT_MSG",
)
print(f"\n✅  Pushed: {url}")
print(f"🌐  Space:  https://huggingface.co/spaces/$HF_REPO")
PYEOF


# # Push deploy/ as-is
# ./deploy_to_hfspace.sh

# # Sync config.py / evaluate.py / abliterate.py from scripts/ first, then push
# ./deploy_to_hfspace.sh --sync

# # Custom commit message
# ./deploy_to_hfspace.sh -m "Fix panel alignment"

# # Both
# ./deploy_to_hfspace.sh --sync -m "Update CSS and helpers"