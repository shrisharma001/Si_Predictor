# si-predictor: prediction API and operator interface for hot-metal silicon.
# The dataset and the trained models are NOT baked into the image. They are mounted at run time:
#   ./data  -> /app/data   (read-only, the dataset)
#   ./model_docker -> /app/model  (the container trains its models here, with its own library versions)
# Quick start (see README, "Docker"):
#   docker compose build
#   docker compose run --rm si train
#   docker compose up            then open http://localhost:8000
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HOME=/tmp

# OpenMP runtime for XGBoost and scikit-learn
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements-docker.txt .
RUN pip install -r requirements-docker.txt

COPY si_pipeline.py furnace.py health.py app.py train_final.py run_experiments.py run_extra.py test_api.py test_furnace.py test_health.py docker-entrypoint.sh ./
COPY static ./static
COPY examples ./examples

# Run as an unprivileged user. model/ and results/ stay writable for any user id, so the
# container can also run as the host user (docker compose passes HOST_UID/HOST_GID).
RUN chmod 755 docker-entrypoint.sh \
 && mkdir -p data model results \
 && chmod 777 model results \
 && useradd --uid 1000 --no-create-home --shell /usr/sbin/nologin app
USER 1000:1000

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"

ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["serve"]