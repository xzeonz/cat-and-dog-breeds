# RasKita Backend API Deployment

This directory contains the backend API for the RasKita project, including the FastAPI app and the trained model file.

## Prerequisites

- Docker installed on your machine

## Build Docker Image

From this directory, run the following command to build the Docker image:

```
docker build -t raskita-backend .
```

## Run Docker Container

Run the container with the following command:

```
docker run -d -p 8000:8000 --name raskita-backend-container raskita-backend
```

This will start the FastAPI backend API and expose it on port 8000.

## Test the API

You can test the API by sending requests to:

```
http://localhost:8000/
```

or the prediction endpoint:

```
http://localhost:8000/predict
```

## Stop the Container

To stop the running container:

```
docker stop raskita-backend-container
```

To remove the container:

```
docker rm raskita-backend-container
```

## Notes

- The model file `best_model.pth` is included in the Docker image.
- The backend API loads breed descriptions from an external URL at startup.
- The training folder is not required for deployment.
