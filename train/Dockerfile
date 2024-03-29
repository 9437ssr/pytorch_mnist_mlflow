FROM python:3.8-slim

ARG RUN_ID
ARG MODEL_NAME
ENV SERVER_PORT 5000
ENV SERVER_HOST 0.0.0.0
ENV ARTIFACT_STORE /opt/mlflow/artifactStore
ENV PYTHONPATH /opt/mlflow/artifactStore
RUN mkdir -p /opt/mlflow/scripts \
&& mkdir -p ${ARTIFACT_STORE}

RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
COPY scripts/tracking_entrypoint.sh /opt/mlflow/scripts/
COPY code/ ${ARTIFACT_STORE}
RUN pip install -r ${ARTIFACT_STORE}/requirements.txt
RUN chmod +x /opt/mlflow/scripts/tracking_entrypoint.sh
RUN find /opt/mlflow/artifactStore -type f -exec sed -i 's/\\/\//g' {} +

ENTRYPOINT [ "/bin/bash", "/opt/mlflow/scripts/tracking_entrypoint.sh" ]


