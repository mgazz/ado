# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

from enum import Enum


class DeploymentStrategy(str, Enum):
    """
    Deployment strategy for vLLM environments.

    This enum defines the available deployment strategies for vLLM model serving.
    Each strategy represents a different approach to deploying and managing vLLM
    instances in Kubernetes.
    """

    K8S_DEPLOYMENT = "k8s_deployment"
    """Standard Kubernetes Deployment with Service (default)"""

    KSERVE = "kserve"
    """KServe InferenceService with ServingRuntime"""

    # Future strategies can be added here, e.g.:
    # KNATIVE = "knative"
    # HELM = "helm"


# Made with Bob
