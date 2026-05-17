#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT_FORWARD_PORT="${KFP_UI_PORT:-8080}"
PORT_FORWARD_PID_FILE="${KFP_PORT_FORWARD_PID_FILE:-/tmp/kfp-ui-port-forward.pid}"
PORT_FORWARD_LOG_FILE="${KFP_PORT_FORWARD_LOG_FILE:-/tmp/kfp-ui-port-forward.log}"

log() {
  echo "[$(date +"%H:%M:%S")] $*"
}

die() {
  log "❌ $*"
  exit 1
}

usage() {
  cat <<'USAGE'
Usage: scripts/kfp_operator_workflow.sh <command>

Commands:
  all               Deploy/update KFP, start port-forward, then verify status
  deploy            Deploy/update KFP and pipeline image
  port-forward      Start KFP UI port-forward in the background
  stop-port-forward Stop running KFP UI port-forward process
  verify            Verify Minikube, Kubeflow pods, UI access, and recent workflow status
  status            Print quick operator status summary

Environment variables:
  KFP_UI_PORT                Local port for KFP UI (default: 8080)
  KFP_PORT_FORWARD_PID_FILE  PID file location (default: /tmp/kfp-ui-port-forward.pid)
  KFP_PORT_FORWARD_LOG_FILE  Log file location (default: /tmp/kfp-ui-port-forward.log)
USAGE
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "Required command not found: $1"
}

check_prereqs() {
  require_cmd minikube
  require_cmd kubectl
  require_cmd curl
}

check_minikube_running() {
  minikube status >/dev/null 2>&1 || die "Minikube is not running. Start it with: minikube start --cpus 4 --memory 8192"
}

run_deploy() {
  log "🚀 Deploying/updating Kubeflow operator workflow"
  check_prereqs
  check_minikube_running

  [ -x "${ROOT_DIR}/scripts/deploy_kfp_minikube.sh" ] || die "Missing executable deploy script at scripts/deploy_kfp_minikube.sh"

  (
    cd "${ROOT_DIR}"
    ./scripts/deploy_kfp_minikube.sh
  )

  log "✅ Deployment/update complete"
}

is_pid_running() {
  local pid="$1"
  kill -0 "$pid" >/dev/null 2>&1
}

start_port_forward() {
  log "🔌 Ensuring KFP UI port-forward on localhost:${PORT_FORWARD_PORT}"
  check_prereqs
  check_minikube_running

  if [ -f "${PORT_FORWARD_PID_FILE}" ]; then
    local existing_pid
    existing_pid="$(cat "${PORT_FORWARD_PID_FILE}" 2>/dev/null || true)"
    if [ -n "${existing_pid}" ] && is_pid_running "${existing_pid}"; then
      log "ℹ️ Port-forward already running (pid=${existing_pid})"
      return 0
    fi
    rm -f "${PORT_FORWARD_PID_FILE}"
  fi

  nohup kubectl port-forward -n kubeflow svc/ml-pipeline-ui "${PORT_FORWARD_PORT}":80 \
    >"${PORT_FORWARD_LOG_FILE}" 2>&1 &
  local pf_pid=$!
  echo "${pf_pid}" >"${PORT_FORWARD_PID_FILE}"

  sleep 2
  if ! is_pid_running "${pf_pid}"; then
    log "----- port-forward logs -----"
    tail -n 40 "${PORT_FORWARD_LOG_FILE}" || true
    die "Port-forward failed to start"
  fi

  log "✅ Port-forward running (pid=${pf_pid}, log=${PORT_FORWARD_LOG_FILE})"
}

stop_port_forward() {
  if [ ! -f "${PORT_FORWARD_PID_FILE}" ]; then
    log "ℹ️ No port-forward PID file found"
    return 0
  fi

  local pid
  pid="$(cat "${PORT_FORWARD_PID_FILE}" 2>/dev/null || true)"
  if [ -n "${pid}" ] && is_pid_running "${pid}"; then
    kill "${pid}" >/dev/null 2>&1 || true
    log "🛑 Stopped port-forward (pid=${pid})"
  else
    log "ℹ️ Port-forward process is not running"
  fi
  rm -f "${PORT_FORWARD_PID_FILE}"
}

verify_kfp() {
  log "🔍 Running Kubeflow operator verification"
  check_prereqs
  check_minikube_running

  kubectl get ns kubeflow >/dev/null 2>&1 || die "Namespace 'kubeflow' not found"

  log "• Pod health (kubeflow namespace):"
  kubectl get pods -n kubeflow

  local unhealthy
  unhealthy="$(kubectl get pods -n kubeflow --no-headers | awk '$3!="Running" && $3!="Completed" {print $1":"$3}')"
  if [ -n "${unhealthy}" ]; then
    log "⚠️ Unhealthy pods detected:"
    echo "${unhealthy}"
    die "Verification failed due to unhealthy pods"
  fi

  kubectl get svc -n kubeflow ml-pipeline-ui >/dev/null 2>&1 || die "Service ml-pipeline-ui not found in kubeflow namespace"

  log "• Checking UI endpoint: http://localhost:${PORT_FORWARD_PORT}/"
  local http_code
  http_code="$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:${PORT_FORWARD_PORT}/" || true)"
  if [ "${http_code}" != "200" ]; then
    log "⚠️ UI check returned HTTP ${http_code}"
    die "KFP UI health check failed; ensure port-forward is running"
  fi

  if kubectl get wf -n kubeflow >/dev/null 2>&1; then
    log "• Recent workflows (kubeflow):"
    kubectl get wf -n kubeflow --sort-by=.metadata.creationTimestamp | tail -n 5
  else
    log "ℹ️ Workflow CRD not currently available; skipping workflow summary"
  fi

  log "✅ Verification passed: cluster, pods, UI, and workflow visibility are healthy"
}

status_summary() {
  check_prereqs
  check_minikube_running

  log "📋 Operator status summary"
  kubectl get pods -n kubeflow
  if [ -f "${PORT_FORWARD_PID_FILE}" ]; then
    local pid
    pid="$(cat "${PORT_FORWARD_PID_FILE}" 2>/dev/null || true)"
    if [ -n "${pid}" ] && is_pid_running "${pid}"; then
      log "• Port-forward: running (pid=${pid}) on localhost:${PORT_FORWARD_PORT}"
    else
      log "• Port-forward: stale PID file"
    fi
  else
    log "• Port-forward: not started"
  fi
}

main() {
  local cmd="${1:-}"
  case "${cmd}" in
    all)
      run_deploy
      start_port_forward
      verify_kfp
      ;;
    deploy)
      run_deploy
      ;;
    port-forward)
      start_port_forward
      ;;
    stop-port-forward)
      stop_port_forward
      ;;
    verify)
      verify_kfp
      ;;
    status)
      status_summary
      ;;
    -h|--help|help|"")
      usage
      ;;
    *)
      die "Unknown command: ${cmd}. Run with --help for usage."
      ;;
  esac
}

main "$@"
