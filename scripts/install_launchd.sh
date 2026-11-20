#!/usr/bin/env bash
# Install the Telegram bot launchd agent from the template in launchd/ (paths filled in
# here, so the repo holds no machine-specific paths). Pipeline runs are started from
# Telegram (PLAN.md §17); there is no nightly schedule any more.
#   scripts/install_launchd.sh install     render into ~/Library/LaunchAgents and load
#   scripts/install_launchd.sh uninstall   unload and remove
#   scripts/install_launchd.sh status      show state, last exit code, pid
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$PWD"
AGENTS="$HOME/Library/LaunchAgents"
DOMAIN="gui/$(id -u)"
LABELS=(com.sanyam.clipper.telegram)
RETIRED=(com.sanyam.clipper) # the old 01:30 nightly run
LOGS="$("$ROOT/.venv/bin/python" -c 'from clipper.config import load_config; print(load_config().dir("logs"))')"

retire() {
  for label in "${RETIRED[@]}"; do
    if launchctl print "$DOMAIN/$label" >/dev/null 2>&1 || [[ -f "$AGENTS/$label.plist" ]]; then
      launchctl bootout "$DOMAIN/$label" 2>/dev/null || true
      rm -f "$AGENTS/$label.plist"
      echo "removed retired agent $label"
    fi
  done
}

case "${1:-status}" in
install)
  mkdir -p "$AGENTS"
  chmod +x scripts/run_pipeline.sh scripts/run_bot.sh
  retire
  for label in "${LABELS[@]}"; do
    dst="$AGENTS/$label.plist"
    sed -e "s|__REPO__|$ROOT|g" -e "s|__LOGS__|$LOGS|g" "launchd/$label.plist" >"$dst"
    plutil -lint -s "$dst"
    launchctl bootout "$DOMAIN/$label" 2>/dev/null || true
    # right after a bootout, bootstrap can fail (error 5) until the old job is gone
    for try in 1 2 3 4 5; do
      launchctl bootstrap "$DOMAIN" "$dst" 2>/dev/null && break
      [[ $try -eq 5 ]] && launchctl bootstrap "$DOMAIN" "$dst" # last try: show the error
      sleep 2
    done
    launchctl enable "$DOMAIN/$label"
    echo "loaded $label"
  done
  ;;
uninstall)
  retire
  for label in "${LABELS[@]}"; do
    launchctl bootout "$DOMAIN/$label" 2>/dev/null || true
    rm -f "$AGENTS/$label.plist"
    echo "removed $label"
  done
  ;;
status)
  for label in "${LABELS[@]}" "${RETIRED[@]}"; do
    echo "== $label"
    launchctl print "$DOMAIN/$label" 2>/dev/null |
      grep -E '^\s*(state|pid|last exit code|runs) =' || echo "   not loaded"
  done
  ;;
*)
  echo "usage: $0 install|uninstall|status" >&2
  exit 2
  ;;
esac
