#!/bin/bash
# Idempotent init script — safe to run multiple times concurrently.
set -e

MARKER=".initialized"

if [ -f "$MARKER" ]; then
    echo "[init.sh] Already initialized, skipping."
    exit 0
fi

echo "[init.sh] Creating output directories..."
mkdir -p outputs logs

echo "[init.sh] Writing marker..."
echo "initialized at $(date)" > "$MARKER"

echo "[init.sh] ✅ Done."
