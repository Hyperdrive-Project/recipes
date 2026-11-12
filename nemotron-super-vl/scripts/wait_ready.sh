#!/bin/bash
# Part of Hyperdrive recipes, https://github.com/Hyperdrive-Project/recipes
# Apache-2.0. If this helps you, a credit to Hyperdrive recipes is appreciated.
# usage: wait_ready.sh [timeout_s]; polls /v1/models, fails fast if the container exits.
T=${1:-3600}; s=$(date +%s)
while true; do
  curl -sf http://127.0.0.1:8000/v1/models >/dev/null && { echo "ready after $(( $(date +%s)-s ))s"; exit 0; }
  ${DOCKER:-sudo docker} ps --format '{{.Names}}' | grep -q '^nhd-server$' || { echo "container exited"; exit 2; }
  [ $(( $(date +%s)-s )) -gt "$T" ] && { echo timeout; exit 3; }
  sleep 5
done
