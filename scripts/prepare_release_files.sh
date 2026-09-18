#!/usr/bin/env bash
# Gather every installer/archive from the downloaded build artifacts into one
# flat folder, ready to attach to a GitHub Release.
#
# GitHub Releases rejects any single file of 2 GiB or more (the Linux CUDA
# bundle is ~3.5 GB per file), so bigger files are cut into parts under the
# limit -- "<name>.part-00", "<name>.part-01", ... -- which users join with
# `cat`. SHA256SUMS lists the ORIGINAL files, so a joined file can be checked.
#
# Used by .github/workflows/build.yml:
#   bash scripts/prepare_release_files.sh artifacts release
#
# LIMIT_BYTES and PART_SIZE exist so the splitting can be tested with tiny files.
set -euo pipefail

src=${1:?usage: prepare_release_files.sh <artifacts-dir> <output-dir>}
out=${2:?usage: prepare_release_files.sh <artifacts-dir> <output-dir>}
limit=${LIMIT_BYTES:-2147483648}
part=${PART_SIZE:-1900M}

mkdir -p "$out"
: > "$out/SHA256SUMS"

while IFS= read -r -d '' file; do
    name=$(basename "$file")
    size=$(stat -c %s "$file")
    (cd "$(dirname "$file")" && sha256sum "$name") >> "$out/SHA256SUMS"
    if [ "$size" -lt "$limit" ]; then
        cp "$file" "$out/"
    else
        echo "::notice title=Split for upload::$name is $size bytes, over the 2 GiB release limit: publishing it in parts"
        split -b "$part" -d -a 2 "$file" "$out/$name.part-"
    fi
done < <(find "$src" -type f \( -name "*.zip" -o -name "*.tar.gz" -o -name "*.deb" -o -name "*.exe" -o -name "*.dmg" \) -print0 | sort -z)

sort -k2 "$out/SHA256SUMS" -o "$out/SHA256SUMS"
ls -la "$out"
