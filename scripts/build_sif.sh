#!/usr/bin/env bash
#
# Convert the Docker image `tsc-marl:latest` into an Apptainer image `tsc-marl.sif`.
# HPC clusters (Cyfronet Ares/Athena/Helios) forbid Docker and run Apptainer.
#
# Pick ONE of the paths below. The registry path (A) is recommended for
# reproducibility; the offline path (B) needs no registry account.
#
# Requires Apptainer on the build machine. Build the .sif where you have
# privileges (your GPU desktop, or a compute node with --fakeroot) — NOT on a
# cluster login node.
set -euo pipefail

IMAGE="${IMAGE:-tsc-marl:latest}"
SIF="${SIF:-tsc-marl.sif}"
REGISTRY_REF="${REGISTRY_REF:-}"   # e.g. ghcr.io/youruser/tsc-marl:v1  (path A)

if ! command -v apptainer >/dev/null 2>&1; then
    echo "ERROR: apptainer not found. Install Apptainer, or build on a machine that has it."
    echo "  Ubuntu: sudo add-apt-repository -y ppa:apptainer/ppa && sudo apt install -y apptainer"
    exit 1
fi

if [[ -n "${REGISTRY_REF}" ]]; then
    # ---- Path A: pull directly from a container registry (recommended) --------
    # First, on the machine that built the image:
    #   docker tag ${IMAGE} ${REGISTRY_REF} && docker push ${REGISTRY_REF}
    # Then on ANY machine (incl. the cluster login node — pulling is allowed):
    echo ">> Building ${SIF} from registry ${REGISTRY_REF}"
    apptainer build "${SIF}" "docker://${REGISTRY_REF}"
else
    # ---- Path B: offline via a docker-save archive (no registry) --------------
    if command -v docker >/dev/null 2>&1 && docker image inspect "${IMAGE}" >/dev/null 2>&1; then
        echo ">> Saving ${IMAGE} to docker-archive and building ${SIF}"
        TAR="$(mktemp --suffix=.tar)"
        docker save "${IMAGE}" -o "${TAR}"
        apptainer build "${SIF}" "docker-archive://${TAR}"
        rm -f "${TAR}"
    else
        echo "ERROR: no REGISTRY_REF set and Docker image '${IMAGE}' not found locally."
        echo "  Either: build the image first (make build), or"
        echo "  set REGISTRY_REF=ghcr.io/youruser/tsc-marl:tag after pushing it."
        exit 1
    fi
fi

echo ">> Done: ${SIF}"
echo "   Transfer it to the cluster's \$SCRATCH, e.g.:"
echo "     scp ${SIF} plgYOURLOGIN@ares.cyfronet.pl:\$SCRATCH/"
echo "   Verify on the cluster:  apptainer exec --nv \$SCRATCH/${SIF} python /opt/smoke_test.py"
