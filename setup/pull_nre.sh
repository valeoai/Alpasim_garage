#!/usr/bin/env bash
# Build the Apptainer image of the NRE renderer (PAI/NuRec track) from NGC.
#   bash setup/pull_nre.sh
# nvcr.io/nvidia/nre/nre-ga is public, so no NGC key is needed (set NGC_API_KEY
# if NGC ever asks for one). It is amd64-only: x86_64 machines only.
# ~14 GB of layers; building the .sif needs about 2x that in $APPTAINER_TMPDIR
# and takes a few CPU-heavy minutes. Needs internet, unless NRE_OCI_LAYOUT points
# to the image's layers already on disk (an OCI layout), e.g. on a cluster whose
# internet-connected nodes can pull but not squash the image (see the README).
set -euo pipefail
HPC_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "$HPC_ROOT/lib/common.sh"

[ "$(uname -m)" = x86_64 ] || { echo "ERROR: $NRE_IMAGE is amd64-only; this is $(uname -m)" >&2; exit 1; }
if [ -f "$NRE_SIF" ]; then echo ">> $NRE_SIF exists; delete it to rebuild"; exit 0; fi

# Apptainer, or SingularityCE where Apptainer isn't installed (e.g. `module load singularity`)
RUNTIME=${CONTAINER_RUNTIME:-$(command -v apptainer || command -v singularity || true)}
[ -n "$RUNTIME" ] || { echo "ERROR: neither apptainer nor singularity is on PATH" >&2; exit 1; }

# keep the layer cache and build scratch off $HOME (quota). A site that sets
# SINGULARITY_CACHEDIR itself (e.g. in its singularity module) keeps its own.
export APPTAINER_CACHEDIR
export APPTAINER_TMPDIR=${APPTAINER_TMPDIR:-$APPTAINER_CACHEDIR/tmp}
export SINGULARITY_CACHEDIR=${SINGULARITY_CACHEDIR:-$APPTAINER_CACHEDIR}
export SINGULARITY_TMPDIR=${SINGULARITY_TMPDIR:-$APPTAINER_TMPDIR}
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR" "$SINGULARITY_CACHEDIR" "$(dirname "$NRE_SIF")"
if [ -n "${NGC_API_KEY:-}" ]; then
  export APPTAINER_DOCKER_USERNAME='$oauthtoken' APPTAINER_DOCKER_PASSWORD=$NGC_API_KEY
  export SINGULARITY_DOCKER_USERNAME='$oauthtoken' SINGULARITY_DOCKER_PASSWORD=$NGC_API_KEY
fi

rm -f "$NRE_SIF.part"   # a leftover from a failed build would stop the next one
if [ -n "${NRE_OCI_LAYOUT:-}" ]; then
  # Offline: build from an OCI image layout already on disk (one image), e.g. the layers a
  # Singularity cache holds after a pull that was killed while squashing.
  echo ">> building $NRE_SIF from oci:$NRE_OCI_LAYOUT with $RUNTIME"
  "$RUNTIME" build "$NRE_SIF.part" "oci:$NRE_OCI_LAYOUT"
else
  echo ">> pulling docker://$NRE_IMAGE -> $NRE_SIF with $RUNTIME"
  "$RUNTIME" pull "$NRE_SIF.part" "docker://$NRE_IMAGE"
fi
mv "$NRE_SIF.part" "$NRE_SIF"
"$RUNTIME" inspect --runscript "$NRE_SIF" | tail -3
echo ">> done. The layer cache can be removed: $(basename "$RUNTIME") cache clean -f"
