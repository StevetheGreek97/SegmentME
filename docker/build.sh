#!/bin/bash
# Build SegmentME inside an Ubuntu container (docker/Dockerfile) so the
# tarball and .deb run on hosts at least as new as the container, not just
# the (possibly newer) machine doing the build. On x86_64 that's Ubuntu
# 22.04+ / glibc >= 2.35; on arm64 it's Ubuntu 24.04+ / glibc >= 2.39 --
# forced up from 22.04 because PyQt6 6.9.0's arm64 wheel needs a newer
# glibc than its x86_64 one does (22.04's glibc is too old for it, and
# building PyQt6 from source needs qmake, which isn't worth adding just to
# stay on 22.04 -- see the comment in docker/Dockerfile).
#
#   docker/build.sh                                # CPU tarball (build.py's arguments)
#   docker/build.sh --deb                          # tarball + .deb
#   docker/build.sh --skip-build --deb --no-archive
#
# Output lands in dist/ as usual. The container's venv is kept in
# build_env/ubuntu<version>-<flavor>/ so later runs are quick, and
# everything is created with your own user id, not root's.
set -e
cd "$(dirname "$0")/.."

case "$(uname -m)" in
    aarch64|arm64) UBUNTU_VERSION=24.04; PY_VERSION=3.12 ;;
    *)             UBUNTU_VERSION=22.04; PY_VERSION=3.10 ;;
esac
IMAGE="segmentme-build:ubuntu$UBUNTU_VERSION"
case " $* " in *" cuda "*) FLAVOR=cuda ;; *) FLAVOR=cpu ;; esac
if [ -t 0 ]; then TTY="-it"; else TTY=""; fi

docker build --build-arg "UBUNTU_VERSION=$UBUNTU_VERSION" --build-arg "PY_VERSION=$PY_VERSION" \
    -t "$IMAGE" docker
mkdir -p build_env build dist
exec docker run --rm $TTY \
    --user "$(id -u):$(id -g)" -e HOME=/tmp -e USER="$(id -un)" \
    -v "$PWD:/src" -w /src \
    "$IMAGE" python3 build.py --env-dir "build_env/ubuntu$UBUNTU_VERSION-$FLAVOR" "$@"
