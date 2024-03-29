FROM python:3.8-slim

ENV SERVER_PORT 5000
ENV SERVER_HOST 0.0.0.0
ENV PYTHONPATH /opt/mlflow/
ENV MODEL_URI gs://mlflow-model-artifacts/rnn/pyfunc-rnn
RUN mkdir -p /opt/mlflow/

COPY . /opt/mlflow

RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install -r /opt/mlflow/requirements.txt
RUN pip install google-cloud-storage


ENV GOOGLE_APPLICATION_CREDENTIALS /opt/mlflow/keyfile.json
RUN chmod +x /opt/mlflow/serving_entrypoint.sh
ENTRYPOINT ["/usr/bin/env"]
CMD ["bash", "/opt/mlflow/serving_entrypoint.sh" ]



