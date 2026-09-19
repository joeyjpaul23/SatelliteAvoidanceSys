#!/usr/bin/env bash
# Create the AEGIS VM on Oracle Always Free (VM.Standard.A1.Flex, 1 OCPU /
# 6 GB), retrying across availability domains until A1 host capacity frees up.
#
#   deploy/oracle/launch.sh          # retry until RUNNING, then print the IP
#   deploy/oracle/launch.sh --plan   # look everything up, launch nothing
#
# Needs the OCI CLI with a working ~/.oci/config [DEFAULT] profile. Nothing
# account-specific lives in this file: the tenancy comes from the config, and
# the subnet, image and availability domains are looked up by name.
#
# Never creates a second instance: before every attempt it checks for a
# non-terminated instance with the same name and, if one exists, waits for
# that one instead. A failed attempt creates nothing and costs nothing.
set -euo pipefail

NAME=${AEGIS_INSTANCE_NAME:-aegis}
SUBNET_NAME=${AEGIS_SUBNET_NAME:-public subnet-aegis-vcn}
SSH_KEY=${AEGIS_SSH_PUBLIC_KEY:-${HOME}/.ssh/id_ed25519.pub}
INTERVAL_S=${AEGIS_RETRY_INTERVAL_S:-60}
SHAPE=VM.Standard.A1.Flex
SHAPE_CONFIG='{"ocpus": 1, "memoryInGBs": 6}'
IMAGE_OS="Canonical Ubuntu"
IMAGE_VERSION="24.04 Minimal aarch64"

log() { printf '%s  %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"; }
die() { log "FATAL: $*" >&2; exit 1; }

command -v oci >/dev/null 2>&1 || die "oci CLI not found (install: uv tool install oci-cli)"
[[ -f ${SSH_KEY} ]] || die "no SSH public key at ${SSH_KEY}"
[[ -f ${HOME}/.oci/config ]] || die "no ~/.oci/config"

TENANCY=$(awk -F= '
  /^\[/ { in_default = ($0 == "[DEFAULT]"); next }
  in_default && $1 ~ /^[[:space:]]*tenancy[[:space:]]*$/ { gsub(/[[:space:]]/, "", $2); print $2 }
' "${HOME}/.oci/config")
[[ ${TENANCY} == ocid1.tenancy.* ]] || die "no tenancy OCID in ~/.oci/config [DEFAULT]"

json_lines() { python3 -c 'import json, sys; print("\n".join(json.load(sys.stdin) or []))'; }

SUBNET=$(oci network subnet list -c "${TENANCY}" --all \
  --query "data[?\"display-name\"=='${SUBNET_NAME}' && \"lifecycle-state\"=='AVAILABLE'].id | [0]" \
  --raw-output)
[[ ${SUBNET} == ocid1.subnet.* ]] || die "no available subnet named '${SUBNET_NAME}'"

IMAGE=$(oci compute image list -c "${TENANCY}" --all \
  --operating-system "${IMAGE_OS}" --operating-system-version "${IMAGE_VERSION}" \
  --shape "${SHAPE}" --sort-by TIMECREATED --sort-order DESC \
  --query 'data[0].id' --raw-output)
IMAGE_NAME=$(oci compute image get --image-id "${IMAGE}" --query 'data."display-name"' --raw-output)
[[ ${IMAGE} == ocid1.image.* ]] || die "no ${IMAGE_OS} ${IMAGE_VERSION} image for ${SHAPE}"

ADS=()
while IFS= read -r ad; do
  [[ -n ${ad} ]] && ADS+=("${ad}")
done < <(oci iam availability-domain list -c "${TENANCY}" --query 'data[].name' | json_lines)
((${#ADS[@]})) || die "no availability domains listed"

existing_instance() {
  oci compute instance list -c "${TENANCY}" --display-name "${NAME}" \
    --query "data[?\"lifecycle-state\"!='TERMINATED' && \"lifecycle-state\"!='TERMINATING'].id | [0]" \
    --raw-output
}

log "instance ${NAME}: ${SHAPE} ${SHAPE_CONFIG}"
log "image     ${IMAGE_NAME}"
log "subnet    ${SUBNET_NAME}"
log "ADs       ${ADS[*]}"
log "ssh key   ${SSH_KEY}"

if [[ ${1:-} == --plan ]]; then
  current=$(existing_instance)
  [[ ${current} == ocid1.instance.* ]] && log "an instance named ${NAME} already exists: ${current}"
  log "plan only; nothing launched"
  exit 0
fi

wait_and_report() {
  local id=$1
  log "waiting for ${id} to reach RUNNING"
  oci compute instance get --instance-id "${id}" --wait-for-state RUNNING \
    --max-wait-seconds 1200 --wait-interval-seconds 15 >/dev/null
  local ip
  ip=$(oci compute instance list-vnics --instance-id "${id}" --query 'data[0]."public-ip"' --raw-output)
  log "RUNNING. public IP: ${ip}   (ssh ubuntu@${ip})"
}

err_file=$(mktemp)
trap 'rm -f "${err_file}"' EXIT
attempt=0
while :; do
  if ! current=$(existing_instance 2>"${err_file}"); then
    # The listing call can time out too (laptop asleep, Wi-Fi drop): retry, never exit.
    log "could not list instances (network); retrying in 2 min"
    sleep 120
    continue
  fi
  if [[ ${current} == ocid1.instance.* ]]; then
    log "instance ${NAME} already exists; not launching another"
    wait_and_report "${current}"
    exit 0
  fi

  ad=${ADS[$((attempt % ${#ADS[@]}))]}
  attempt=$((attempt + 1))
  if out=$(oci compute instance launch \
    --compartment-id "${TENANCY}" \
    --availability-domain "${ad}" \
    --display-name "${NAME}" \
    --hostname-label "${NAME}" \
    --shape "${SHAPE}" \
    --shape-config "${SHAPE_CONFIG}" \
    --image-id "${IMAGE}" \
    --subnet-id "${SUBNET}" \
    --assign-public-ip true \
    --ssh-authorized-keys-file "${SSH_KEY}" \
    --instance-options '{"areLegacyImdsEndpointsDisabled": true}' \
    --availability-config '{"recoveryAction": "RESTORE_INSTANCE"}' \
    --is-pv-encryption-in-transit-enabled true \
    --query 'data.id' --raw-output 2>"${err_file}"); then
    log "attempt ${attempt} (${ad}): LAUNCHED ${out}"
    wait_and_report "${out}"
    exit 0
  fi

  if grep -q "Out of host capacity" "${err_file}"; then
    log "attempt ${attempt} (${ad}): out of host capacity; next try in ${INTERVAL_S}s"
    sleep "${INTERVAL_S}"
  elif grep -qE 'TooManyRequests|"status": 429' "${err_file}"; then
    log "attempt ${attempt} (${ad}): throttled by OCI; backing off 5 min"
    sleep 300
  elif grep -qiE 'timed out|RequestException|ConnectionError|Max retries exceeded|"status": 50[234]' "${err_file}"; then
    # Laptop slept, Wi-Fi dropped, or OCI hiccuped: transient, so keep going.
    log "attempt ${attempt} (${ad}): network error; retrying in 2 min"
    sleep 120
  else
    log "attempt ${attempt} (${ad}): unexpected error, stopping:"
    cat "${err_file}" >&2
    exit 1
  fi
done
