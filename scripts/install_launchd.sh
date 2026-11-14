#!/usr/bin/env bash
# Install the launchd agents from the templates in launchd/ (paths filled in here, so the
# repo holds no machine-specific paths).
#   scripts/install_launchd.sh install     render into ~/Library/LaunchAgents and load
#   scripts/install_launchd.sh uninstall   unload and remove
#   scripts/install_launchd.sh status      show state, last exit code, pid
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$PWD"
AGENTS="$HOME/Library/LaunchAgents"
DOMAIN="gui/$(id -u)"
LABELS=(com.sanyam.clipper com.sanyam.clipper.telegram)
LOGS="$("$ROOT/.venv/bin/python" -c 'from clipper.config import load_config; print(load_config().dir("logs"))')"

case "${1:-status}" in
install)
  mkdir -p "$AGENTS"
  chmod +x scripts/run_nightly.sh scripts/run_bot.sh
  for label in "${LABELS[@]}"; do
    dst="$AGENTS/$label.plist"
    sed -e "s|__REPO__|$ROOT|g" -e "s|__LOGS__|$LOGS|g" "launchd/$label.plist" >"$dst"
    plutil -lint -s "$dst"
    launchctl bootout "$DOMAIN/$label" 2>/dev/null || true
    launchctl bootstrap "$DOMAIN" "$dst"
    launchctl enable "$DOMAIN/$label"
    echo "loaded $label"
  done
  ;;
uninstall)
  for label in "${LABELS[@]}"; do
    launchctl bootout "$DOMAIN/$label" 2>/dev/null || true
    rm -f "$AGENTS/$label.plist"
    echo "removed $label"
  done
  ;;
status)
  for label in "${LABELS[@]}"; do
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
