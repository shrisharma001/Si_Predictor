#!/bin/sh
# Start-up script for the si-predictor container.
#   serve  (default)  run the API and interface with gunicorn on port $PORT (8000)
#   train             train the live and demo models into /app/model from the dataset in /app/data
#   test              run the API test suite (needs the dataset and both models)
#   anything else     run that command inside the container
set -eu
cmd="${1:-serve}"
DATA="${SI_DEMO_DATA:-data/1_blast_furnace_data_first_dataset.xlsx}"
MODEL="${SI_MODEL_PATH:-model/si_pipeline_hybrid.pkl}"

need_data() {
  if [ ! -f "$DATA" ]; then
    echo "No dataset at /app/$DATA." >&2
    echo "Mount your data folder, for example: -v \"\$(pwd)/data:/app/data:ro\" (docker compose does this for you)." >&2
    exit 2
  fi
}

case "$cmd" in
  train)
    need_data
    if [ ! -w model ]; then
      echo "/app/model is not writable by this container (it runs as user $(id -u))." >&2
      echo "Mount a model folder you own and run with your user id, for example HOST_UID=\$(id -u) HOST_GID=\$(id -g) docker compose run --rm si train" >&2
      exit 2
    fi
    exec python train_final.py
    ;;
  serve)
    if [ ! -f "$MODEL" ]; then
      echo "No model at /app/$MODEL. Train one first: docker compose run --rm si train" >&2
      exit 2
    fi
    exec gunicorn --workers "${WORKERS:-2}" --bind "0.0.0.0:${PORT:-8000}" --timeout 120 \
         --access-logfile - --error-logfile - app:app
    ;;
  test)
    need_data
    exec python test_api.py
    ;;
  *)
    exec "$@"
    ;;
esac
