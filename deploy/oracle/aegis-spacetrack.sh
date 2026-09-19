#!/usr/bin/env bash
# Run `python -m aegis.ingest.spacetrack <args>` on the VM as the aegis user,
# with the same credentials, cache and working directory as the services.
# Installed by setup.sh as /usr/local/bin/aegis-spacetrack.
#
#   sudo aegis-spacetrack check     # 0 ok, 2 credentials missing, 1 failure
#   sudo aegis-spacetrack refresh
set -euo pipefail

if [[ ${EUID} -ne 0 ]]; then
  exec sudo -- "$0" "$@"
fi

# shellcheck disable=SC2016  # expanded by the inner shell, not this one
exec sudo -u aegis -- /bin/bash -c '
  set -euo pipefail
  set -a
  . /etc/aegis/aegis.env
  set +a
  export XDG_CACHE_HOME=/var/cache/aegis
  umask 0027
  cd /opt/aegis/app/aegis
  exec /opt/aegis/venv/bin/python -m aegis.ingest.spacetrack "$@"
' aegis-spacetrack "$@"
