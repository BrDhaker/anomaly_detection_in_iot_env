'''
Local IoT Anomaly Detection Architecture Diagram Generator
This script generates a diagram illustrating the architecture of the IoT anomaly detection pipeline deployed Locally.
We have to install Graphiz library if not already installed:
https://graphviz.gitlab.io/download/
also diagrams library via pip:
pip install diagrams
'''
from diagrams import Diagram
from diagrams.azure.general import Usericon  # For client
from diagrams.onprem.compute import Server  # For local data processing / scaling
from diagrams.aws.ml import SagemakerModel      # For ML model (Isolation Forest)
from diagrams.programming.framework import Fastapi # For Prediction API
from diagrams.oci.governance import Logging  # For monitoring/logging
from diagrams.onprem.storage import CEPH_OSD        # For storage
from diagrams.aws.analytics import KinesisDataStreams       # For simulator


with Diagram("Local IoT Anomaly Detection Pipeline", show=True, direction="LR"):
    streaming = KinesisDataStreams("IoT Data Ingestion / Streaming \n(Local Simulator)")
    processing = Server("Data Processing\n(Normalization / Scaling)")
    model = SagemakerModel("ML Model\n(Isolation Forest)")
    api = Fastapi("Prediction API\n(FastAPI)")
    client = Usericon("Client\n(Dashboard / Alerts)")
    monitoring = Logging("Monitoring & Logging\n(Metrics, Errors)")
    storage = CEPH_OSD("Local Storage\n(Models, Predictions , Logs)")

    # Flow
    streaming >> processing >> model
    model >> api
    client >> api
    api >> client
    api >> monitoring
    monitoring >> storage
    processing >> monitoring
    model >> storage
