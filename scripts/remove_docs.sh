#!/bin/bash
# Remove docs folder from git tracking but keep locally

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# Get the project root (parent of scripts)
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Change to project root
cd "$PROJECT_ROOT"

# Remove docs folder from git tracking but keep locally
git rm --cached -r docs/

# Remove other personal files from git tracking
git rm --cached tests/manual_test_gradio.py 2>/dev/null
git rm --cached main_legacy.py 2>/dev/null

echo "✅ Files removed from git tracking (but kept locally)"
echo "Next steps:"
echo "1. Review changes: git status"
echo "2. Commit: git commit -m 'Remove personal docs and files from tracking'"
echo "3. Push: git push origin main"
