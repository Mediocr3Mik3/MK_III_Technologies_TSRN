#!/usr/bin/env bash
# Mount an Azure blob container at /mnt/blob using blobfuse2.
#
# Required env vars:
#   STORAGE_ACCOUNT, STORAGE_CONTAINER, STORAGE_KEY
#
# Optional:
#   MOUNT_POINT (default /mnt/blob)
#   CACHE_DIR   (default /mnt/blob_cache; should be on local SSD)
set -euo pipefail

: "${STORAGE_ACCOUNT:?must be set}"
: "${STORAGE_CONTAINER:?must be set}"
: "${STORAGE_KEY:?must be set}"

MOUNT_POINT="${MOUNT_POINT:-/mnt/blob}"
CACHE_DIR="${CACHE_DIR:-/mnt/blob_cache}"

# Install blobfuse2 if missing
if ! command -v blobfuse2 >/dev/null 2>&1; then
    echo "Installing blobfuse2..."
    sudo apt-get update -y
    # Microsoft repo for blobfuse2
    UBUNTU_VER=$(lsb_release -rs)
    wget -q "https://packages.microsoft.com/config/ubuntu/${UBUNTU_VER}/packages-microsoft-prod.deb" -O /tmp/msprod.deb
    sudo dpkg -i /tmp/msprod.deb || sudo apt-get -f install -y
    sudo apt-get update -y
    sudo apt-get install -y blobfuse2 fuse3
fi

sudo mkdir -p "${MOUNT_POINT}" "${CACHE_DIR}"
sudo chown "$(id -u):$(id -g)" "${MOUNT_POINT}" "${CACHE_DIR}"

# Generate config
CFG=/tmp/blobfuse2.yaml
cat >"${CFG}" <<EOF
allow-other: true
logging:
  type: syslog
  level: log_warning
file_cache:
  path: ${CACHE_DIR}
  timeout-sec: 240
  max-size-mb: 0
attr_cache:
  timeout-sec: 7200
azstorage:
  type: block
  account-name: ${STORAGE_ACCOUNT}
  container: ${STORAGE_CONTAINER}
  mode: key
  account-key: ${STORAGE_KEY}
  endpoint: https://${STORAGE_ACCOUNT}.blob.core.windows.net
components:
  - libfuse
  - file_cache
  - attr_cache
  - azstorage
EOF

# Already mounted?
if mountpoint -q "${MOUNT_POINT}"; then
    echo "Already mounted at ${MOUNT_POINT}"
    exit 0
fi

blobfuse2 mount "${MOUNT_POINT}" --config-file="${CFG}"
echo "Mounted ${STORAGE_ACCOUNT}/${STORAGE_CONTAINER} at ${MOUNT_POINT}"
ls -la "${MOUNT_POINT}" || true
