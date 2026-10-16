#!/usr/bin/env bash
# docker_smoke_test.sh - builds the image and checks the whole container workflow on your machine.
#   ./docker_smoke_test.sh            (run from the project folder; takes a few minutes the first time)
# Steps: build, train both models, run the test suite inside the container, start the service,
# wait until it reports healthy, check the interface and a replay, then stop it.
set -euo pipefail

cd "$(dirname "$0")"
export HOST_UID="${HOST_UID:-$(id -u)}" HOST_GID="${HOST_GID:-$(id -g)}"
PORT=8000
step() { printf '\n== %s\n' "$*"; }
fail() { printf '\nFAILED: %s\n' "$*" >&2; docker compose down >/dev/null 2>&1 || true; exit 1; }

step "Checking prerequisites"
command -v docker >/dev/null || fail "docker is not installed"
docker compose version >/dev/null 2>&1 || fail "the docker compose plugin is not installed"
if ! docker info >/dev/null 2>&1; then
  # (|| true: docker info is expected to fail here, and set -euo pipefail must not stop the script before the advice)
  err=$(docker info 2>&1 | grep -iE "cannot connect|permission denied|failed to connect|error" | head -1 || true)
  ctx=$(docker context show 2>/dev/null || true); ctx=${ctx:-default}
  echo "Docker is installed, but this terminal cannot reach the Docker engine (context: $ctx)."
  echo "  $err"
  echo
  if printf '%s' "$err" | grep -qi "permission denied"; then
    echo "The engine is running but your user may not use it. Add yourself to the docker group, then log out and in again:"
    echo "  sudo usermod -aG docker \$USER"
    echo "(or, for this terminal only: newgrp docker)"
  elif { docker context ls 2>/dev/null || true; } | grep -q "desktop-linux"; then
    echo "Docker Desktop is installed. Start the Docker Desktop app and wait until it says it is running, then select its context:"
    echo "  docker context use desktop-linux"
  elif [ -S "${XDG_RUNTIME_DIR:-/run/user/$(id -u)}/docker.sock" ] || command -v dockerd-rootless.sh >/dev/null 2>&1; then
    echo "This looks like rootless Docker. Start it and point the CLI at it:"
    echo "  systemctl --user start docker"
    echo "  export DOCKER_HOST=unix://${XDG_RUNTIME_DIR:-/run/user/$(id -u)}/docker.sock"
  else
    echo "Start the Docker engine (and make it start at boot):"
    echo "  sudo systemctl enable --now docker"
    echo "If your system has no systemd (for example WSL), use: sudo service docker start"
    echo "Then, if you get 'permission denied', add yourself to the docker group:"
    echo "  sudo usermod -aG docker \$USER      (log out and in again afterwards)"
  fi
  fail "the Docker engine is not reachable; nothing was built"
fi
[ -f data/1_blast_furnace_data_first_dataset.xlsx ] || fail "data/1_blast_furnace_data_first_dataset.xlsx is missing"
[ -f static/index.html ] || fail "static/index.html is missing (the interface files belong in static/)"
[ -f examples/sample_request.json ] || fail "examples/sample_request.json is missing (create it with: python make_sample_request.py)"
mkdir -p model_docker      # the container's own models (./model stays for python app.py on this computer)
echo "ok (running as uid $HOST_UID)"

step "Building the image"
docker compose build || fail "the image did not build"

step "Training both models inside the container"
docker compose run --rm si train || fail "training failed"
[ -f model_docker/si_pipeline_hybrid.pkl ] && [ -f model_docker/si_pipeline_demo.pkl ] || fail "the model files were not written to ./model_docker"

step "Running the test suite inside the container"
docker compose run --rm si test || fail "the container test suite failed"

step "Starting the service"
docker compose up -d || fail "the service did not start"
ok=""
for _ in $(seq 1 45); do
  if curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then ok=1; break; fi
  sleep 2
done
[ -n "$ok" ] || { docker compose logs --tail 40; fail "the service did not answer /health within 90 s"; }
echo "healthy"

step "Checking the interface and a replay"
code=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/")
[ "$code" = "200" ] || fail "GET / returned $code"
summary=$(curl -fsS -X POST -H 'Content-Type: application/json' \
  -d '{"start":"2016-03-10T00:00","casts":48}' "http://127.0.0.1:$PORT/api/demo/replay") || fail "the replay request failed"
case "$summary" in *'"mae"'*) ;; *) fail "unexpected replay response: $summary" ;; esac
echo "interface: HTTP 200; replay answered"

step "Stopping the service"
docker compose down

printf '\nALL DOCKER CHECKS PASSED. Start the demo with: docker compose up   (then open http://localhost:%s)\n' "$PORT"