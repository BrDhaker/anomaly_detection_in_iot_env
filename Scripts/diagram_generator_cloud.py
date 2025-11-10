'''
Cloud IoT Anomaly Detection Architecture Diagram Generator (Azure)
This script generates a diagram illustrating the architecture of the IoT anomaly detection pipeline deployed on Azure.
We have to install Graphiz library if not already installed:
https://graphviz.gitlab.io/download/
also diagrams library via pip:
pip install diagrams
'''


from diagrams import Diagram, Cluster
from diagrams.onprem.compute import Server  # For local data processing / scaling
from diagrams.aws.analytics import KinesisDataStreams       # For simulator
from diagrams.azure.ml import MachineLearningServiceWorkspaces  # For ML model
from diagrams.azure.compute import AppServices  # For API service
from diagrams.azure.monitor import Monitor  # For monitoring
from diagrams.azure.storage import BlobStorage  # For storage
from diagrams.azure.general import Usericon

with Diagram("Cloud IoT Anomaly Detection Architecture (Azure)", show=True):
    streaming = KinesisDataStreams("IoT Data Ingestion / Streaming \n(Local Simulator)")
    with Cluster("Docker Container :"):
        model = MachineLearningServiceWorkspaces("Anomaly Model")
        api = AppServices("Web App Service\n(API)")
        processing = Server("Data Processing\n(Normalization / Scaling)")

    client = Usericon("Client\n(Dashboard)")
    monitoring = Monitor("Azure Monitor")
    storage = BlobStorage("Blob Storage\n(Dataset)")

    storage >> streaming  >> processing >> model
    model >> api                   # ML model returns scores to API
    client >> api                  # Client sends requests to API
    api >> client                  # API responds to client
    api >> monitoring              # Monitor API performance and health
    streaming >> storage          # Processed data/logs stored
