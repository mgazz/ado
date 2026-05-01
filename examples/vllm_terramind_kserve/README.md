# Prithvi Geospatial Model Testing with KServe

This example demonstrates how to test the IBM-NASA Prithvi geospatial model
using the vLLM performance actuator with KServe deployment mode.

## Overview

This example includes:

- **actuatorconfiguration.yaml**: Configures the vLLM actuator to use KServe
  deployment strategy with Prometheus metrics enabled
- **discoveryspace.yaml**: Defines the parameter space for testing Prithvi
  model configurations
- **operation.yaml**: Configures a random walk exploration using sequential
  grouped sampling

## Prerequisites

1. Access to a Kubernetes/OpenShift cluster with:
   - KServe installed and configured
   - At least one GPU node (NVIDIA A100 or H100)
   - Appropriate RBAC permissions for ServingRuntime and InferenceService CRDs

2. ado installed and configured

3. Access to the required container images and models

## Quick Start

### 1. Create the Actuator Configuration

First, create the actuator configuration that enables KServe deployment:

```bash
ado create actuatorconfiguration -f actuatorconfiguration.yaml
```

This configuration:

- Enables KServe deployment mode (`deployment_strategy: kserve`)
- Configures Prometheus metrics collection
- Sets up OpenTelemetry tracing
- Configures the target namespace and environment limits

### 2. Create the Discovery Space

Create the discovery space that defines the parameter space to explore:

```bash
ado create space -f discoveryspace.yaml --use-default-sample-store
```

This space explores:

- Different GPU types (A100, H100)
- Various CPU and memory configurations
- Multiple vLLM parameters (batch tokens, sequence length, etc.)
- Different workload intensities (request rates, number of prompts)

### 3. Run the Operation

Execute the exploration using the random walk operator:

```bash
ado create operation -f operation.yaml \
    --use-latest space \
    --use-latest actuatorconfiguration
```

The operation uses sequential grouped sampling to efficiently reuse KServe
InferenceService deployments across multiple measurements.

## KServe Deployment Features

### Prometheus Metrics

The InferenceService is automatically configured with Prometheus annotations:

- `prometheus.io/scrape: "true"` - Enables Prometheus scraping
- `prometheus.io/path: "/metrics"` - Specifies the metrics endpoint
- `prometheus.io/port: "8000"` - Specifies the port for metrics

These annotations allow Prometheus to automatically discover and scrape metrics
from the vLLM inference service.

### Observability

The configuration includes OpenTelemetry tracing support via the
`otel_traces_endpoint` parameter, enabling distributed tracing of inference
requests.

## Monitoring Progress

### Check Operation Status

```bash
ado show summary operation --use-latest
```

### View Entities and Results

```bash
# View measured entities
ado show entities space --use-latest

# Export results to CSV
ado show entities space --use-latest -o csv --output-file results.csv
```

### Monitor KServe Resources

```bash
# Check InferenceService status
kubectl get inferenceservices -n cp-testing

# Check ServingRuntime status
kubectl get servingruntimes -n cp-testing

# View InferenceService details
kubectl describe inferenceservice <name> -n cp-testing
```

## Customization

### Custom KServe Templates

You can provide custom YAML templates for ServingRuntime and InferenceService
resources by uncommenting and setting these parameters in
`actuatorconfiguration.yaml`:

```yaml
parameters:
  serving_runtime_template: "path/to/custom/serving_runtime.yaml"
  inference_service_template: "path/to/custom/inference_service.yaml"
```

### Adjusting the Parameter Space

Modify `discoveryspace.yaml` to:

- Add or remove GPU types
- Adjust CPU/memory ranges
- Change vLLM configuration parameters
- Modify workload parameters

### Changing Sampling Strategy

Edit `operation.yaml` to adjust:

- `numberEntities`: Number of entities to sample
- `batchSize`: Number of entities to process in parallel
- `samplerConfig.mode`: Sampling strategy (sequential, random, etc.)
- `grouping`: Properties that define deployment groups

## Troubleshooting

### Check InferenceService Logs

```bash
kubectl logs -n cp-testing -l serving.kserve.io/inferenceservice=<name>
```

### Verify Prometheus Metrics

```bash
# Port-forward to the inference service
kubectl port-forward -n cp-testing svc/<service-name> 8000:8000

# Check metrics endpoint
curl http://localhost:8000/metrics
```

### Debug Operation Issues

```bash
# View operation details
ado show details operation --use-latest

# Check for failed measurements
ado show results operation --use-latest --include-failed
```

## References

- [vLLM Performance Actuator Documentation](../../plugins/actuators/vllm_performance/README.md)
- [KServe Documentation](https://kserve.github.io/website/)
- [ado Documentation](https://ibm.github.io/ado/)

## Made with Bob
