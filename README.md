# pytorch_mnist_mlflow
    
    ├── deploy <- contains all required files except "keyfile.json" (secret file for google storage authentication) for deploying application in kubenetes cluster (GKE)
    
    ├── notebooks 
        - model_development.ipynb <- this notebook was used for training the model and tracking experiment runs with mlflow in local environment
        - invoke_endpoint.ipynb   <- this notebook was used for invoking the endpoint of deployed application in google cloud
        
    ├── train  <- contains all required files for training the model in a remote server 

# steps followed :
    1. "model_development.ipynb" notebook was used to perform several training experiments with different hyperparameters
    2. Various runs were compared through mlflow ui 
![image](https://github.com/9437ssr/pytorch_mnist_mlflow/assets/22223702/fa570155-0857-4f0c-8ab5-638601aae055)

    3. A pyfunc flavoured model was logged with a modified predict method. This was done to enable the model to give single digit as inference output instead of array of probabilities
    4. Finalized model artefacts were uploaded to google cloud storage which represented the MODEL_URI location.
    5. Deployment specific files like Dockerfile, mlflow_serving.yaml were prepared 
    6. Container created out of the Dockerfile was first tested locally in google compute e2 instance
    7. On successful testing the docker image was pushed to Google Artefact Repository (Container Registry) 
    8. GKE cluster was created and connected to the instance terminal
    9. "mlflow_serving.yaml" file configurations were applied 
    10. A LoadBalancer service was created with 8080 port number exposed
![kubectl](https://github.com/9437ssr/pytorch_mnist_mlflow/assets/22223702/fbd3ba75-bbc8-41f0-97d9-4c63567eff97)

    11. "invoke_endpoint.ipynb" notebook was used to test deployed application successfully 
![image](https://github.com/9437ssr/pytorch_mnist_mlflow/assets/22223702/58ed40dc-a1d1-4557-9236-68bfdee93da6)




