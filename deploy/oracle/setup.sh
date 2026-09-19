#!/usr/bin/env bash
# One-time (and re-runnable) VM setup for AEGIS on an Oracle Always Free
# Ampere A1 instance running Ubuntu 24.04. Run on the VM after push.sh:
#
#   sudo bash /opt/aegis/app/deploy/oracle/setup.sh
#
# Safe to run repeatedly: it installs what is missing, reinstalls the package
# and units, restarts the console, and never touches an existing env file.
set -euo pipefail

APP_ROOT=/opt/aegis/app
PKG_DIR=${APP_ROOT}/aegis
VENV=/opt/aegis/venv
ENV_DIR=/etc/aegis
ENV_FILE=${ENV_DIR}/aegis.env
CACHE_DIR=/var/cache/aegis
UNIT_DIR=/etc/systemd/system
BUNDLE_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

log() { printf '\n==> %s\n' "$*"; }

if [[ ${EUID} -ne 0 ]]; then
  echo "setup.sh must run as root: sudo bash $0" >&2
  exit 1
fi
if [[ ! -f ${PKG_DIR}/pyproject.toml ]]; then
  echo "No package at ${PKG_DIR}. Run deploy/oracle/push.sh from the laptop first." >&2
  exit 1
fi

log "System packages"
missing=()
# nano: the Minimal image may ship without an editor, and sudoedit needs one.
# unattended-upgrades: Minimal may also ship without it; configured below.
for pkg in python3-venv python3-pip rsync nano unattended-upgrades; do
  dpkg -s "${pkg}" >/dev/null 2>&1 || missing+=("${pkg}")
done
if ((${#missing[@]})); then
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq
  apt-get install -y -qq "${missing[@]}"
else
  echo "already installed"
fi

log "Automatic security updates"
# Written explicitly rather than trusting the image default: this box sits on
# the internet with SSH open and nobody logs in to patch it.
cat >/etc/apt/apt.conf.d/20auto-upgrades <<'APT'
APT::Periodic::Update-Package-Lists "1";
APT::Periodic::Unattended-Upgrade "1";
APT
# Kernel and libc fixes only apply after a reboot. Take it at a quiet hour
# (UTC, clear of the :17 refresh); both AEGIS units are enabled, so they
# come back on boot.
cat >/etc/apt/apt.conf.d/52aegis-reboot <<'APT'
Unattended-Upgrade::Automatic-Reboot "true";
Unattended-Upgrade::Automatic-Reboot-Time "04:40";
APT
echo "enabled (reboots at 04:40 UTC only when an update requires it)"

log "Service user"
if id -u aegis >/dev/null 2>&1; then
  echo "aegis exists"
else
  useradd --system --user-group --no-create-home \
    --home-dir /nonexistent --shell /usr/sbin/nologin aegis
  echo "created aegis"
fi

log "Python environment (${VENV})"
if [[ ! -x ${VENV}/bin/python ]]; then
  python3 -m venv "${VENV}"
fi
"${VENV}/bin/python" -m pip install --quiet --upgrade pip
# Editable on purpose: ingest/ops.py finds its fallback TLE slices through
# the source tree (tests/fixtures), which a regular wheel install would drop.
"${VENV}/bin/python" -m pip install --quiet -e "${PKG_DIR}"
# The services cannot write bytecode under ProtectSystem=strict; do it now.
"${VENV}/bin/python" -m compileall -q "${PKG_DIR}/src" >/dev/null

log "Credentials file (${ENV_FILE})"
install -d -m 0750 -o root -g aegis "${ENV_DIR}"
if [[ -e ${ENV_FILE} ]]; then
  chown root:aegis "${ENV_FILE}"
  chmod 0640 "${ENV_FILE}"
  echo "kept existing file"
else
  install -m 0640 -o root -g aegis "${BUNDLE_DIR}/aegis.env.example" "${ENV_FILE}"
  echo "installed empty template"
fi

log "Cache directory (${CACHE_DIR})"
install -d -m 0750 -o aegis -g aegis "${CACHE_DIR}"

log "systemd units and helper"
for unit in aegis-api.service aegis-refresh.service aegis-refresh.timer; do
  install -m 0644 "${BUNDLE_DIR}/${unit}" "${UNIT_DIR}/${unit}"
done
install -m 0755 "${BUNDLE_DIR}/aegis-spacetrack.sh" /usr/local/bin/aegis-spacetrack
systemctl daemon-reload
systemctl enable --quiet aegis-api.service aegis-refresh.timer
systemctl restart aegis-api.service
systemctl start aegis-refresh.timer

log "Status"
systemctl --no-pager --lines=0 status aegis-api.service || true
systemctl --no-pager list-timers aegis-refresh.timer || true

creds_set=no
if (
  set -a
  # shellcheck disable=SC1090
  . "${ENV_FILE}"
  [[ -n ${SPACETRACK_USER:-} && -n ${SPACETRACK_PASS:-} ]]
); then
  creds_set=yes
fi

cat <<EOF

Setup complete.
EOF
if [[ ${creds_set} == no ]]; then
  cat <<EOF

Next:
  1. sudoedit ${ENV_FILE}                  # fill in SPACETRACK_USER / SPACETRACK_PASS
  2. sudo systemctl restart aegis-api.service
  3. sudo aegis-spacetrack check           # expect exit 0
EOF
else
  cat <<EOF

Credentials are set. Verify with: sudo aegis-spacetrack check
EOF
fi
cat <<EOF

Console (from the laptop):
  ssh -N -L 8000:127.0.0.1:8000 ubuntu@<vm-ip>    then open http://localhost:8000
EOF
