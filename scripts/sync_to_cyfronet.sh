#!/usr/bin/env bash
#
# Sync code + image to Cyfronet, and pull results back. Code goes to $SCRATCH
# (fast, large) not $HOME. Edit LOGIN/CLUSTER, then:
#   ./scripts/sync_to_cyfronet.sh push        # code -> cluster
#   ./scripts/sync_to_cyfronet.sh push-sif     # tsc-marl.sif -> cluster
#   ./scripts/sync_to_cyfronet.sh pull-results # results -> ./results/
set -euo pipefail

LOGIN="${PLG_LOGIN:-plgYOURLOGIN}"
CLUSTER="${PLG_CLUSTER:-ares.cyfronet.pl}"   # or athena.cyfronet.pl / helios.cyfronet.pl
# $SCRATCH is a per-cluster env var; hardcode the resolved path if you know it.
REMOTE_SCRATCH="${PLG_SCRATCH:-\$SCRATCH}"
REMOTE_DIR="${REMOTE_SCRATCH}/tsc-marl"

RSYNC="rsync -avz --progress"
# Never sync heavyweight/local-only stuff to the cluster repo.
EXCLUDES=(--exclude '.git' --exclude 'results' --exclude 'logs'
          --exclude '__pycache__' --exclude '*.sif' --exclude '*.tar'
          --exclude '.venv')

cmd="${1:-help}"
case "$cmd" in
  push)
    $RSYNC "${EXCLUDES[@]}" ./ "${LOGIN}@${CLUSTER}:${REMOTE_DIR}/"
    ;;
  push-sif)
    scp tsc-marl.sif "${LOGIN}@${CLUSTER}:${REMOTE_SCRATCH}/tsc-marl.sif"
    ;;
  pull-results)
    mkdir -p ./results
    $RSYNC "${LOGIN}@${CLUSTER}:${REMOTE_SCRATCH}/results/" ./results/
    ;;
  *)
    echo "usage: $0 {push|push-sif|pull-results}"
    echo "  set PLG_LOGIN / PLG_CLUSTER env vars, or edit the defaults in this file."
    ;;
esac
