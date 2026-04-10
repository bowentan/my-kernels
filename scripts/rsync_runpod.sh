#!/usr/bin/env bash
# rsync_mirror.sh
# Mirror SRC -> DEST, excluding Git’s internal tracking dir (.git/).
# Optional: also exclude *Git-tracked files* (everything in `git ls-files`).

set -euo pipefail

SRC="${1:-.}"
DEST="${2:?Usage: $0 <src_dir> <user@host:/remote/dir>}"

SSH_PORT="${SSH_PORT:-22}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"

# Set to 1 if you meant “exclude files tracked by git” (not just .git/):
EXCLUDE_GIT_TRACKED="${EXCLUDE_GIT_TRACKED:-0}"

RSYNC_OPTS=(
  -a
  --delete
  --compress
  --partial
  --human-readable
  --info=progress2
  --no-owner
  --no-group
)

# Always exclude Git’s internal metadata directory
EXCLUDES=(--exclude='.git/')

SSH_CMD="ssh -p ${SSH_PORT} -i ${SSH_KEY} -o StrictHostKeyChecking=accept-new"

rsync "${RSYNC_OPTS[@]}" "${EXCLUDES[@]}" -e "$SSH_CMD" "${SRC%/}" "$DEST"
