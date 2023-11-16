"""A Python Pulumi program:
EzOut TorchServe GPU Kubernetes infrastructure for containerized deployment
"""

import pulumi
import pulumi_gcp as gcp

import logging

logging.basicConfig(level=logging.INFO)
logging.info("Creating GKE cluster...")

# GKE Cluster configuration
cluster = gcp.container.Cluster("torchserve-gpu",
    initial_node_count=1,
    node_config=gcp.container.ClusterNodeConfigArgs(
        machine_type="n2-standard-4",  # Adjust as needed
        oauth_scopes=[
            "https://www.googleapis.com/auth/compute",
            "https://www.googleapis.com/auth/devstorage.read_only",
            "https://www.googleapis.com/auth/logging.write",
            "https://www.googleapis.com/auth/monitoring"
        ],
        guest_accelerator=gcp.container.ClusterNodeConfigGuestAcceleratorArgs(
            type="nvidia-tesla-t4",  # Choose an appropriate GPU type
            count=1
        ),
    ),
    node_version="latest"
)

pulumi.export('cluster_name', cluster.name)
pulumi.export('kubeconfig', cluster.kubeconfig)
