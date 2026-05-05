# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import json
import logging
import uuid
from enum import Enum
from pathlib import Path
from typing import Any

import yaml

PVC_MOUNT_PATH = "/dev/cache"
PVC_NAME = "vllm-support"
logger = logging.getLogger(__name__)

DEFAULT_PVC_TEMPLATE = "pvc.yaml"
DEFAULT_DEPLOYMENT_TEMPLATE = "deployment.yaml"
DEFAULT_SERVICE_TEMPLATE = "service.yaml"
DEFAULT_SERVING_RUNTIME_TEMPLATE = "serving_runtime.yaml"
DEFAULT_INFERENCE_SERVICE_TEMPLATE = "inference_service.yaml"
DEFAULT_SERVICE_MONITOR_TEMPLATE = "service_monitor.yaml"


class VLLMDtype(Enum):
    """
    Type for VLLM
    """

    AUTO = "auto"
    HALF = "half"
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"
    FLOAT = "float"
    FLOAT32 = "float32"


class ComponentsYaml:
    """
    Build components Yamls class
    """

    @staticmethod
    def get_k8s_name(model: str) -> str:
        """
        convert model for kubernetes usage
        :param model: LLM model
        :return: K8s unique name for a given LLM model
        """
        m_parts = model.split("/")

        # Making sure the resulting name is not longer than 63 characters as it is
        # the maximum allowed for a name in kubernetes.
        # Use short UUID (8 chars) to keep total length under 63 chars
        # Format: vllm-<model_prefix>-<short_uuid>
        # Max length: 5 (vllm-) + 25 (model) + 1 (-) + 8 (uuid) = 39 chars
        # This leaves room for KServe suffixes like "-predictor" (10 chars)
        short_uuid = uuid.uuid4().hex[:8]
        name_prefix = m_parts[-1][: min(len(m_parts[-1]), 25)].rstrip("-")
        return f"vllm-{name_prefix.lower()}-{short_uuid}".replace(".", "-")

    @staticmethod
    def _adjust_file_name(f: str) -> Path:
        """
        Adjust file name to local directory
        :param f: file name
        :return: adjusted file name (Path)
        """
        parent_path = Path(__file__).parents[0]
        return parent_path / f

    def _get_full_path_from_cwd(f: str) -> Path:
        """
        Adjust file name to cwd
        :param f: file name
        :return: adjusted file name (Path)
        """
        return Path.cwd() / f

    @staticmethod
    def deployment_yaml(
        k8s_name: str,
        model: str,
        gpu_type: str = "NVIDIA-A100-80GB-PCIe",
        node_selector: dict[str, str] | None = None,
        image: str = "vllm/vllm-openai:v0.6.3",
        image_pull_secret_name: str = "",
        n_gpus: int = 1,
        n_cpus: int = 8,
        memory: str = "128Gi",
        max_batch_tokens: int = 16384,
        gpu_memory_utilization: float = 0.9,
        dtype: VLLMDtype = VLLMDtype.AUTO,
        cpu_offload: int = 0,
        max_num_seq: int = 256,
        template: str | None = None,
        claim_name: str | None = None,
        hf_token: str | None = None,
        enforce_eager: bool = False,
        skip_tokenizer_init: bool = False,
        io_processor_plugin: str | None = None,
        otel_traces_endpoint: str | None = None,
    ) -> dict[str, Any]:
        """
        Generate deployment yaml
        :param k8s_name: deployment name
        :param model: model name
        :param gpu_type: gpu type, for example NVIDIA-A100-80GB-PCIe, Tesla-V100-PCIE-16GB, etc.
        :param node_selector: optional node selector
        :param image: image name to use
        :param image_pull_secret_name: name of the image pull secret
        :param n_gpus: number of GPUs to use in VLLM
        :param n_cpus: number of CPUs for VLLM pod
        :param memory: memory for VLLM pod
        :param max_batch_tokens: Vllm parameter - maximum number of batched tokens per iteration
        :param gpu_memory_utilization: VLLM parameter - GPU memory utilization
        :param dtype: VLLM parameter - data type for model weights and activations
        :param cpu_offload: VLLM parameter - the space in GiB to offload to CPU, per GPU
        :param max_num_seq: VLLM parameter - Maximum number of sequences per iteration.
        :param template: template for deployment yaml
        :param claim_name: PVC name
        :param hf_token: huggingface token
        :param enforce_eager: flag to enforce using Pytorch eager mode
        :param skip_tokenizer_init: flag to skip tokenizer initialization in vLLM
        :param io_processor_plugin: name of the IO processor plugin to be used by vLLM
        :return:
        """
        if node_selector is None:
            node_selector = {}

        # read template
        if template is None:
            logger.debug("Using default Deployment template")
            template_file = ComponentsYaml._adjust_file_name(
                DEFAULT_DEPLOYMENT_TEMPLATE
            )
        else:
            template_file = ComponentsYaml._get_full_path_from_cwd(template)
        logger.debug(
            f"Creating Deployment from template file: {template_file.absolute()}"
        )

        try:
            template_data = template_file.read_text()
            deployment_yaml = yaml.safe_load(template_data)
        except Exception as exception:
            error_string = f"Exception reading deployment yaml template {exception}"
            logger.error(error_string)
            raise ValueError(error_string) from exception

        # Update metadata
        metadata = deployment_yaml["metadata"]
        metadata["name"] = k8s_name
        metadata["labels"]["app.kubernetes.io/instance"] = k8s_name

        # update spec
        spec = deployment_yaml["spec"]
        # selector
        spec["selector"]["matchLabels"]["app.kubernetes.io/instance"] = k8s_name

        # update template
        d_template = spec["template"]
        # template metadata
        d_template["metadata"]["labels"]["app.kubernetes.io/instance"] = k8s_name

        # update template spec
        spec = d_template["spec"]
        # node selector
        spec["nodeSelector"] = {"nvidia.com/gpu.product": gpu_type}

        if len(node_selector) > 0:
            spec["nodeSelector"].update(node_selector)
        # image pull secret
        if image_pull_secret_name is not None and image_pull_secret_name != "":
            spec["imagePullSecrets"] = [{"name": image_pull_secret_name}]
        # volumes
        if claim_name is not None:
            spec["volumes"].extend(
                [{"name": PVC_NAME, "persistentVolumeClaim": {"claimName": claim_name}}]
            )

        vllm_serve_args = [
            model,
            "--max-num-batched-tokens",
            f"{max_batch_tokens}",
            "--gpu-memory-utilization",
            f"{gpu_memory_utilization}",
            "--cpu-offload-gb",
            f"{cpu_offload}",
            "--max-num-seq",
            f"{max_num_seq}",
            "--tensor-parallel-size",
            f"{n_gpus}",
            "--dtype",
            dtype.value,
        ]

        if enforce_eager:
            vllm_serve_args.append("--enforce-eager")
        if skip_tokenizer_init:
            vllm_serve_args.append("--skip-tokenizer-init")
        if io_processor_plugin is not None:
            vllm_serve_args.append("--io-processor-plugin")
            vllm_serve_args.append(io_processor_plugin)
            vllm_serve_args.append("--enable-mm-embeds")
        if otel_traces_endpoint is not None:
            vllm_serve_args.append("--otlp-traces-endpoint")
            vllm_serve_args.append(otel_traces_endpoint)

        # container
        container = spec["containers"][0]
        # command + args
        container["command"] = ["vllm", "serve"]
        container["args"] = vllm_serve_args
        # image
        container["image"] = image
        # resources
        requests = container["resources"]["requests"]
        requests["cpu"] = str(n_cpus)
        requests["memory"] = memory
        requests["nvidia.com/gpu"] = str(n_gpus)
        limits = container["resources"]["limits"]
        limits["cpu"] = str(n_cpus)
        limits["memory"] = memory
        limits["nvidia.com/gpu"] = str(n_gpus)

        if container.get("env") is None:
            container["env"] = []
        if hf_token is not None:
            container["env"].append({"name": "HF_TOKEN", "value": hf_token})
        if claim_name is not None:
            container["env"].extend(
                [
                    {
                        "name": "HOME",
                        "value": "/tmp",  # noqa: S108
                    },
                    {
                        "name": "HF_HOME",
                        "value": f"{PVC_MOUNT_PATH}/transformers_cache",
                    },
                ]
            )
        if logging.root.level == logging.DEBUG:
            container["env"].append({"name": "VLLM_LOGGING_LEVEL", "value": "DEBUG"})
        if otel_traces_endpoint is not None:
            container["env"].append(
                {
                    "name": "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
                    "value": otel_traces_endpoint,
                }
            )
        # volume mounts
        if claim_name is not None:
            container["volumeMounts"].extend(
                [
                    {"name": PVC_NAME, "mountPath": PVC_MOUNT_PATH},
                ]
            )

        logger.debug(json.dumps(deployment_yaml, indent=2))
        return deployment_yaml

    @staticmethod
    def service_yaml(k8s_name: str, template: str | None = None) -> dict[str, Any]:
        """
        Generate service yaml for a given model
        :param k8s_name: K8s unique name
        :param template: template for service yaml
        :return: service yaml
        """
        # read template
        if template is None:
            logger.debug("Using default Service template")
            template_file = ComponentsYaml._adjust_file_name(DEFAULT_SERVICE_TEMPLATE)
        else:
            template_file = ComponentsYaml._get_full_path_from_cwd(template)
        logger.debug(f"Creating Service from template file: {template_file.absolute()}")

        try:
            template_data = template_file.read_text()
            service_yaml = yaml.safe_load(template_data)
        except Exception as exception:
            error_string = f"Exception reading service yaml template {exception}"
            logger.error(error_string)
            raise ValueError(error_string) from exception

        # Update metadata
        metadata = service_yaml["metadata"]
        metadata["name"] = k8s_name
        metadata["labels"]["app.kubernetes.io/instance"] = k8s_name

        # update selector
        service_yaml["spec"]["selector"]["app.kubernetes.io/instance"] = k8s_name
        return service_yaml

    @staticmethod
    def pvc_yaml(pvc_name: str, template: str | None = None) -> dict[str, Any]:
        """
        Generate pvc yaml
        :param pvc_name: name of the PVC claim
        :param template: template for pvc yaml
        :return: pvc yaml
        """
        # read template
        if template is None:
            logger.debug("Using default PVC template")
            template_file = ComponentsYaml._adjust_file_name(DEFAULT_PVC_TEMPLATE)
        else:
            template_file = ComponentsYaml._get_full_path_from_cwd(template)
        logger.debug(f"Creating PVC from template file: {template_file.absolute()}")

        try:
            template_data = template_file.read_text()
            pvc_yaml = yaml.safe_load(template_data)
        except Exception as exception:
            error_string = f"Exception reading pvc yaml template {exception}"
            logger.error(error_string)
            raise ValueError(error_string) from exception

        # Update metadata
        pvc_yaml["metadata"]["name"] = pvc_name
        return pvc_yaml

    @staticmethod
    def serving_runtime_yaml(
        k8s_name: str,
        model: str,
        gpu_type: str = "NVIDIA-A100-80GB-PCIe",
        node_selector: dict[str, str] | None = None,
        image: str = "vllm/vllm-openai:v0.6.3",
        image_pull_secret_name: str = "",
        n_gpus: int = 1,
        n_cpus: int = 8,
        memory: str = "128Gi",
        max_batch_tokens: int = 16384,
        gpu_memory_utilization: float = 0.9,
        dtype: VLLMDtype = VLLMDtype.AUTO,
        cpu_offload: int = 0,
        max_num_seq: int = 256,
        template: str | None = None,
        claim_name: str | None = None,
        hf_token: str | None = None,
        enforce_eager: bool = False,
        skip_tokenizer_init: bool = False,
        io_processor_plugin: str | None = None,
        otel_traces_endpoint: str | None = None,
        namespace: str = "vllm-testing",
    ) -> dict[str, Any]:
        """
        Generate ServingRuntime yaml for KServe
        :param k8s_name: ServingRuntime name
        :param model: model name
        :param gpu_type: gpu type, for example NVIDIA-A100-80GB-PCIe, Tesla-V100-PCIE-16GB, etc.
        :param node_selector: optional node selector
        :param image: image name to use
        :param image_pull_secret_name: name of the image pull secret
        :param n_gpus: number of GPUs to use in VLLM
        :param n_cpus: number of CPUs for VLLM pod
        :param memory: memory for VLLM pod
        :param max_batch_tokens: Vllm parameter - maximum number of batched tokens per iteration
        :param gpu_memory_utilization: VLLM parameter - GPU memory utilization
        :param dtype: VLLM parameter - data type for model weights and activations
        :param cpu_offload: VLLM parameter - the space in GiB to offload to CPU, per GPU
        :param max_num_seq: VLLM parameter - Maximum number of sequences per iteration.
        :param template: template for ServingRuntime yaml
        :param claim_name: PVC name
        :param hf_token: huggingface token
        :param enforce_eager: flag to enforce using Pytorch eager mode
        :param skip_tokenizer_init: flag to skip tokenizer initialization in vLLM
        :param io_processor_plugin: name of the IO processor plugin to be used by vLLM
        :param otel_traces_endpoint: OpenTelemetry traces endpoint URL
        :param namespace: Kubernetes namespace
        :return: ServingRuntime yaml
        """
        if node_selector is None:
            node_selector = {}

        # read template
        if template is None:
            logger.debug("Using default ServingRuntime template")
            template_file = ComponentsYaml._adjust_file_name(
                DEFAULT_SERVING_RUNTIME_TEMPLATE
            )
        else:
            template_file = ComponentsYaml._get_full_path_from_cwd(template)
        logger.debug(
            f"Creating ServingRuntime from template file: {template_file.absolute()}"
        )

        try:
            template_data = template_file.read_text()
            serving_runtime_yaml = yaml.safe_load(template_data)
        except Exception as exception:
            error_string = f"Exception reading ServingRuntime yaml template {exception}"
            logger.error(error_string)
            raise ValueError(error_string) from exception

        # Update metadata
        metadata = serving_runtime_yaml["metadata"]
        metadata["name"] = k8s_name
        metadata["namespace"] = namespace
        metadata["labels"]["app"] = k8s_name

        # Build vLLM serve args (reuse logic from deployment_yaml)
        vllm_serve_args = [
            model,
            "--max-num-batched-tokens",
            f"{max_batch_tokens}",
            "--gpu-memory-utilization",
            f"{gpu_memory_utilization}",
            "--cpu-offload-gb",
            f"{cpu_offload}",
            "--max-num-seq",
            f"{max_num_seq}",
            "--tensor-parallel-size",
            f"{n_gpus}",
            "--dtype",
            dtype.value,
        ]

        if enforce_eager:
            vllm_serve_args.append("--enforce-eager")
        if skip_tokenizer_init:
            vllm_serve_args.append("--skip-tokenizer-init")
        if io_processor_plugin is not None:
            vllm_serve_args.append("--io-processor-plugin")
            vllm_serve_args.append(io_processor_plugin)
            vllm_serve_args.append("--enable-mm-embeds")
        if otel_traces_endpoint is not None:
            vllm_serve_args.append("--otlp-traces-endpoint")
            vllm_serve_args.append(otel_traces_endpoint)

        # Update spec
        spec = serving_runtime_yaml["spec"]

        # Update container
        container = spec["containers"][0]
        container["command"] = ["vllm", "serve"]
        container["args"] = vllm_serve_args
        container["image"] = image

        # Update resources
        container["resources"]["limits"]["cpu"] = str(n_cpus)
        container["resources"]["limits"]["memory"] = memory
        container["resources"]["limits"]["nvidia.com/gpu"] = str(n_gpus)
        container["resources"]["requests"]["cpu"] = str(n_cpus)
        container["resources"]["requests"]["memory"] = memory
        container["resources"]["requests"]["nvidia.com/gpu"] = str(n_gpus)

        # Environment variables
        env_vars = []
        if hf_token is not None:
            env_vars.append({"name": "HF_TOKEN", "value": hf_token})
        if claim_name is not None:
            env_vars.extend(
                [
                    {"name": "HOME", "value": "/tmp"},  # noqa: S108
                    {
                        "name": "HF_HOME",
                        "value": f"{PVC_MOUNT_PATH}/transformers_cache",
                    },
                ]
            )
        if logging.root.level == logging.DEBUG:
            env_vars.append({"name": "VLLM_LOGGING_LEVEL", "value": "DEBUG"})
        if otel_traces_endpoint is not None:
            env_vars.append(
                {
                    "name": "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
                    "value": otel_traces_endpoint,
                }
            )
        container["env"] = env_vars

        # Volume mounts and volumes for PVC
        if claim_name is not None:
            container["volumeMounts"] = [
                {"name": PVC_NAME, "mountPath": PVC_MOUNT_PATH}
            ]
            spec["volumes"] = [
                {"name": PVC_NAME, "persistentVolumeClaim": {"claimName": claim_name}}
            ]

        # Image pull secrets
        if image_pull_secret_name is not None and image_pull_secret_name != "":
            spec["imagePullSecrets"] = [{"name": image_pull_secret_name}]

        logger.debug(json.dumps(serving_runtime_yaml, indent=2))
        return serving_runtime_yaml

    @staticmethod
    def inference_service_yaml(
        k8s_name: str,
        gpu_type: str = "NVIDIA-A100-80GB-PCIe",
        node_selector: dict[str, str] | None = None,
        template: str | None = None,
        namespace: str = "vllm-testing",
        max_replicas: int = 1,
    ) -> dict[str, Any]:
        """
        Generate InferenceService yaml for KServe
        :param k8s_name: InferenceService name (must match ServingRuntime name)
        :param gpu_type: gpu type, for example NVIDIA-A100-80GB-PCIe, Tesla-V100-PCIE-16GB, etc.
        :param node_selector: optional node selector
        :param template: template for InferenceService yaml
        :param namespace: Kubernetes namespace
        :param max_replicas: maximum number of replicas for autoscaling (default: 1)
        :return: InferenceService yaml
        """
        if node_selector is None:
            node_selector = {}

        # read template
        if template is None:
            logger.debug("Using default InferenceService template")
            template_file = ComponentsYaml._adjust_file_name(
                DEFAULT_INFERENCE_SERVICE_TEMPLATE
            )
        else:
            template_file = ComponentsYaml._get_full_path_from_cwd(template)
        logger.debug(
            f"Creating InferenceService from template file: {template_file.absolute()}"
        )

        try:
            template_data = template_file.read_text()
            inference_service_yaml = yaml.safe_load(template_data)
        except Exception as exception:
            error_string = (
                f"Exception reading InferenceService yaml template {exception}"
            )
            logger.error(error_string)
            raise ValueError(error_string) from exception

        # Update metadata
        metadata = inference_service_yaml["metadata"]
        metadata["name"] = k8s_name
        metadata["namespace"] = namespace

        # Update spec
        spec = inference_service_yaml["spec"]
        predictor = spec["predictor"]

        # Set runtime reference (must match ServingRuntime name)
        predictor["runtime"] = k8s_name

        # Set max replicas for autoscaling
        predictor["maxReplicas"] = max_replicas

        # Update node selector
        predictor["nodeSelector"] = {"nvidia.com/gpu.product": gpu_type}
        if len(node_selector) > 0:
            predictor["nodeSelector"].update(node_selector)

        # Update node affinity if node_selector is provided
        if len(node_selector) > 0:
            node_affinity = {
                "requiredDuringSchedulingIgnoredDuringExecution": {
                    "nodeSelectorTerms": [
                        {
                            "matchExpressions": [
                                {
                                    "key": key,
                                    "operator": "In",
                                    "values": [value],
                                }
                                for key, value in node_selector.items()
                            ]
                        }
                    ]
                }
            }
            predictor["affinity"]["nodeAffinity"] = node_affinity

        # Update pod anti-affinity label selector
        predictor["affinity"]["podAntiAffinity"][
            "requiredDuringSchedulingIgnoredDuringExecution"
        ][0]["labelSelector"]["matchExpressions"][0]["values"] = [k8s_name]

    @staticmethod
    def service_monitor_yaml(
        k8s_name: str,
        namespace: str = "vllm-testing",
        template: str | None = None,
    ) -> dict[str, Any]:
        """
        Generate ServiceMonitor yaml for OpenShift Prometheus monitoring
        :param k8s_name: ServiceMonitor name (must match InferenceService name)
        :param namespace: Kubernetes namespace
        :param template: template for ServiceMonitor yaml
        :return: ServiceMonitor yaml
        """
        # read template
        if template is None:
            logger.debug("Using default ServiceMonitor template")
            template_file = ComponentsYaml._adjust_file_name(
                DEFAULT_SERVICE_MONITOR_TEMPLATE
            )
        else:
            template_file = ComponentsYaml._get_full_path_from_cwd(template)
        logger.debug(
            f"Creating ServiceMonitor from template file: {template_file.absolute()}"
        )

        try:
            template_data = template_file.read_text()
            service_monitor_yaml = yaml.safe_load(template_data)
        except Exception as exception:
            error_string = f"Exception reading ServiceMonitor yaml template {exception}"
            logger.error(error_string)
            raise ValueError(error_string) from exception

        # Update metadata
        metadata = service_monitor_yaml["metadata"]
        metadata["name"] = k8s_name
        metadata["namespace"] = namespace
        metadata["labels"]["app"] = k8s_name

        # Update spec selector to match InferenceService labels
        service_monitor_yaml["spec"]["selector"]["matchLabels"][
            "serving.kserve.io/inferenceservice"
        ] = k8s_name

        logger.debug(json.dumps(service_monitor_yaml, indent=2))
        return service_monitor_yaml


if __name__ == "__main__":
    t_model = "meta-llama/Llama-3.1-8B-Instruct"
    t_k8s_name = ComponentsYaml.get_k8s_name(model=t_model)
    deployment = ComponentsYaml.deployment_yaml(
        k8s_name=t_k8s_name,
        model=t_model,
        claim_name="vllm-support",
        node_selector={"kubernetes.io/hostname": "cpu16"},
        image="quay.io/dataprep1/data-prep-kit/vllm_image:0.1",
    )
    print(f"Deployment YAML: \n{yaml.dump(deployment)}")
    service = ComponentsYaml.service_yaml(k8s_name=t_k8s_name)
    print(f"Service YAML: \n{yaml.dump(service)}")
    pvc = ComponentsYaml.pvc_yaml(pvc_name="vllm-pvc")
    print(f"PVC YAML: \n{yaml.dump(pvc)}")
