#!/bin/bash
# serving_entrypoint.sh

mlflow models serve -m $MODEL_URI -h $SERVER_HOST -p $SERVER_PORT --no-conda