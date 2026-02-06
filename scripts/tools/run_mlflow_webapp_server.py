import os
os.environ["MLFLOW_DISABLE_HOST_VALIDATION"] = "true"

import socket
from waitress import serve
import mlflow
from mlflow.server import app
import argparse

print("Imported mlflow and mlflow.server.app", flush=True)

parser = argparse.ArgumentParser(description='Start MLflow server')
parser.add_argument('--port', type=int, required=True)
parser.add_argument('--host', type=str)
args = parser.parse_args()

mlflow_port = args.port
bind_host = args.host or "0.0.0.0"

print(f"Starting MLflow server with Waitress at http://{bind_host}:{mlflow_port}...", flush=True)

serve(
    app,
    host=bind_host,
    port=mlflow_port,
    threads=1,
)
