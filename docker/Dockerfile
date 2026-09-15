# Ubuntu build host for SegmentME's Linux packages (used by docker/build.sh).
#
# PyInstaller copies system libraries (libpython, glib, libstdc++, Qt's xcb
# dependencies, ...) from the machine it runs on, so a bundle only runs on
# distributions at least as new as its build host. Built on 22.04 (the
# default -- see docker/build.sh), the tarball and the .deb run on Ubuntu
# 22.04+ and anything with glibc >= 2.35 (Debian 12+).
#
# docker/build.sh overrides UBUNTU_VERSION/PY_VERSION to 24.04/3.12 on
# arm64 only: PyQt6 6.9.0 has no manylinux wheel for Linux arm64 older than
# manylinux_2_39 (glibc >= 2.39, vs. x86_64's manylinux_2_28) -- on 22.04's
# glibc 2.35, pip falls back to building PyQt6 from source, which then
# fails needing qmake (not installed, and not worth installing just to
# avoid a newer base image). Confirmed via a real CI run on
# ubuntu-24.04-arm before this existed. This means the arm64 tarball/.deb
# need glibc >= 2.39 (24.04+) rather than 2.35 -- a real, unavoidable
# asymmetry between the two architectures' minimum supported distro.
ARG UBUNTU_VERSION=22.04
FROM ubuntu:${UBUNTU_VERSION}
ARG PY_VERSION=3.10
ENV DEBIAN_FRONTEND=noninteractive PIP_DISABLE_PIP_VERSION_CHECK=1
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv python3-pip "libpython${PY_VERSION}" binutils ca-certificates \
        libgl1 libegl1 libglx0 libglvnd0 libdrm2 \
        libwayland-client0 libwayland-cursor0 libwayland-egl1 \
        libatk-bridge2.0-0 libatk1.0-0 libatomic1 libatspi2.0-0 libblkid1 \
        libbrotli1 libbsd0 libbz2-1.0 libcairo-gobject2 libcairo2 libcap2 \
        libcom-err2 libdatrie1 libdbus-1-3 libepoxy0 libexpat1 libffi8 \
        libfontconfig1 libfreetype6 libfribidi0 libgcc-s1 libgcrypt20 \
        libgdk-pixbuf-2.0-0 libglib2.0-0 libgpg-error0 libgraphite2-3 \
        libgssapi-krb5-2 libgtk-3-0 libharfbuzz0b libice6 libjpeg-turbo8 \
        libk5crypto3 libkeyutils1 libkrb5-3 libkrb5support0 liblz4-1 liblzma5 \
        libmd0 libmount1 libncursesw6 libpango-1.0-0 libpangocairo-1.0-0 \
        libpangoft2-1.0-0 libpcre2-8-0 libpixman-1-0 libpng16-16 libreadline8 \
        libselinux1 libsm6 libsqlite3-0 libssl3 libstdc++6 libsystemd0 libthai0 \
        libtinfo6 libuuid1 libx11-6 libx11-xcb1 libxau6 libxcb-cursor0 \
        libxcb-glx0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-randr0 \
        libxcb-render-util0 libxcb-render0 libxcb-shape0 libxcb-shm0 \
        libxcb-sync1 libxcb-util1 libxcb-xfixes0 libxcb-xkb1 libxcomposite1 \
        libxcursor1 libxdamage1 libxdmcp6 libxext6 libxfixes3 libxi6 \
        libxinerama1 libxkbcommon-x11-0 libxkbcommon0 libxrandr2 libxrender1 \
        libzstd1 zlib1g \
    && rm -rf /var/lib/apt/lists/*
