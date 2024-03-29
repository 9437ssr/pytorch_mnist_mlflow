# pytorch_mnist_mlflow

├── deploy    <- contains all required files except "keyfile.json" (secret file for google storage authentication) for deploying application in kubenetes cluster (GKE)
├── notebooks 
    - model_development.ipynb <- this notebook was used for training the model and tracking experiment runs with mlflow in local environment
    - invoke_endpoint.ipynb   <- this notebook was used for invoking the endpoint of deployed application
├── train     <- contains all required files for training the model in a remote server 
