#!/usr/bin/env bash
# Remote paths are fixed constants expanded locally on purpose.
# shellcheck disable=SC2029
# Copy the AEGIS package and this deploy bundle from the laptop to the VM.
#
#   deploy/oracle/push.sh <host> [ssh-user]      # ssh-user defaults to ubuntu
#
# Uses rsync over ssh rather than git clone, so the VM runs exactly the local
# tree, uncommitted work included. Files land root-owned under /opt/aegis/app. If the services
# are already installed, it re-runs setup.sh, which reinstalls the package and
# restarts the console. On a fresh VM it prints the setup command to run.
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 <host> [ssh-user]" >&2
  exit 64
fi

HOST=$1
SSH_USER=${2:-ubuntu}
TARGET="${SSH_USER}@${HOST}"

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
PKG_SRC=${REPO_ROOT}/aegis

REMOTE_APP=/opt/aegis/app
REMOTE_PKG=${REMOTE_APP}/aegis
REMOTE_BUNDLE=${REMOTE_APP}/deploy/oracle

if [[ ! -f ${PKG_SRC}/pyproject.toml ]]; then
  echo "cannot find ${PKG_SRC}/pyproject.toml" >&2
  exit 1
fi

# -rlpt instead of -a: no -o/-g, so files are owned by root on the VM rather
# than by the laptop's numeric uid. tests/fixtures is kept on purpose.
RSYNC_FLAGS=(
  -rlptz --delete
  --rsync-path="sudo rsync"
  --exclude='.venv/'
  --exclude='__pycache__/'
  --exclude='.pytest_cache/'
  --exclude='.ruff_cache/'
  --exclude='/artifacts/'
  --exclude='*.egg-info/'
  --exclude='*.pyc'
  --exclude='.DS_Store'
  --exclude='.env'
  --exclude='.env.*'
)

echo "==> Preparing ${TARGET}"
ssh "${TARGET}" "command -v rsync >/dev/null 2>&1 || { sudo apt-get update -qq && sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq rsync; }
sudo install -d -m 0755 /opt/aegis ${REMOTE_APP} ${REMOTE_PKG} ${REMOTE_APP}/deploy ${REMOTE_BUNDLE}"

echo "==> Syncing aegis/ -> ${REMOTE_PKG}"
rsync "${RSYNC_FLAGS[@]}" "${PKG_SRC}/" "${TARGET}:${REMOTE_PKG}/"

echo "==> Syncing deploy/oracle/ -> ${REMOTE_BUNDLE}"
rsync "${RSYNC_FLAGS[@]}" "${SCRIPT_DIR}/" "${TARGET}:${REMOTE_BUNDLE}/"

if ssh "${TARGET}" "test -f /etc/systemd/system/aegis-api.service"; then
  echo "==> Services installed; re-running setup.sh"
  ssh "${TARGET}" "sudo bash ${REMOTE_BUNDLE}/setup.sh"
else
  cat <<EOF

Code is on the VM. First-time setup:
  ssh ${TARGET} sudo bash ${REMOTE_BUNDLE}/setup.sh
EOF
fi
