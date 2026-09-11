#!/bin/bash
# Build SegmentME inside an Ubuntu 22.04 container (docker/Dockerfile) so the
# tarball and .deb run on Ubuntu 22.04+ / glibc >= 2.35. A bundle built
# directly on a newer host only runs on hosts at least that new.
#
#   docker/build.sh                                # CPU tarball (build.py's arguments)
#   docker/build.sh --deb                          # tarball + .deb
#   docker/build.sh --skip-build --deb --no-archive
#
# Output lands in dist/ as usual. The container's venv is kept in
# build_env/ubuntu22.04-<flavor>/ so later runs are quick, and everything is
# created with your own user id, not root's.
set -e
cd "$(dirname "$0")/.."

IMAGE="segmentme-build:ubuntu22.04"
case " $* " in *" cuda "*) FLAVOR=cuda ;; *) FLAVOR=cpu ;; esac
if [ -t 0 ]; then TTY="-it"; else TTY=""; fi

docker build -t "$IMAGE" docker
mkdir -p build_env build dist
exec docker run --rm $TTY \
    --user "$(id -u):$(id -g)" -e HOME=/tmp -e USER="$(id -un)" \
    -v "$PWD:/src" -w /src \
    "$IMAGE" python3 build.py --env-dir "build_env/ubuntu22.04-$FLAVOR" "$@"
