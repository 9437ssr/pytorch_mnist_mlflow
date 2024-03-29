#/bin/bash

# Launch our mlflow tracking server
set -e
set -x

mlflow server --default-artifact-root=${ARTIFACT_STORE} --host ${SERVER_HOST} --port ${SERVER_PORT}
