# OpenShift Prometheus Metrics for KServe vLLM Deployments

## Problem

When deploying vLLM with KServe on OpenShift, Prometheus metrics are not automatically scraped even though:

1. User workload monitoring is enabled (`enableUserWorkload: true`)
2. Prometheus annotations are present in the InferenceService

## Root Cause

OpenShift's user workload monitoring requires a **ServiceMonitor** custom resource to discover and scrape metrics from services. The Prometheus annotations alone are insufficient.

## Solution

The vLLM performance actuator now automatically creates a ServiceMonitor resource when deploying with the KServe strategy.

### What Was Added

1. **ServiceMonitor YAML Template** (`service_monitor.yaml`)
   - Defines the ServiceMonitor resource structure
   - Configured to match InferenceService labels
   - Scrapes metrics from the `/metrics` endpoint on port `http`

2. **ServiceMonitor Generation** (`build_components.py`)
   - Added `service_monitor_yaml()` method to generate ServiceMonitor resources
   - Follows the same pattern as other KServe resources

3. **ServiceMonitor Management** (`manage_components.py`)
   - `create_service_monitor()` - Creates the ServiceMonitor
   - `check_service_monitor_exists()` - Checks if ServiceMonitor exists
   - `delete_service_monitor()` - Deletes the ServiceMonitor

4. **Automatic Deployment** (`create_environment.py`)
   - ServiceMonitor is automatically created after InferenceService
   - Can be disabled with `enable_service_monitor=False` parameter

### ServiceMonitor Configuration

The ServiceMonitor is configured to:

- Match services with label `serving.kserve.io/inferenceservice: <k8s_name>`
- Scrape metrics from the `http` port
- Use the `/metrics` path
- Scrape every 30 seconds

```yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: <k8s_name>
  namespace: <namespace>
  labels:
    app: <k8s_name>
spec:
  selector:
    matchLabels:
      serving.kserve.io/inferenceservice: <k8s_name>
  endpoints:
  - port: http
    path: /metrics
    interval: 30s
    scheme: http
```

## Usage

### Automatic (Default)

ServiceMonitor is created automatically when using KServe deployment strategy:

```python
from ado_actuators.vllm_performance.k8s.create_environment import create_kserve_environment

create_kserve_environment(
    k8s_name="my-model",
    model="meta-llama/Llama-3.1-8B-Instruct",
    pvc_name="vllm-support",
    namespace="my-namespace",
    # ServiceMonitor is created automatically
)
```

### Disable ServiceMonitor

To disable ServiceMonitor creation:

```python
create_kserve_environment(
    k8s_name="my-model",
    model="meta-llama/Llama-3.1-8B-Instruct",
    pvc_name="vllm-support",
    namespace="my-namespace",
    enable_service_monitor=False,  # Disable ServiceMonitor
)
```

### Custom ServiceMonitor Template

To use a custom ServiceMonitor template:

```python
create_kserve_environment(
    k8s_name="my-model",
    model="meta-llama/Llama-3.1-8B-Instruct",
    pvc_name="vllm-support",
    namespace="my-namespace",
    service_monitor_template="path/to/custom/template.yaml",
)
```

## Verification

### Check ServiceMonitor Creation

```bash
oc get servicemonitor -n <namespace>
```

### Check Prometheus Targets

1. Access the Prometheus UI (Thanos Querier in OpenShift)
2. Navigate to Status → Targets
3. Look for your ServiceMonitor target
4. Verify it's in "UP" state

### Query Metrics

```bash
# From Prometheus UI or using PromQL
vllm_request_duration_seconds
vllm_num_requests_running
vllm_gpu_cache_usage_perc
```

## Troubleshooting

### ServiceMonitor Not Created

Check if the ServiceMonitor exists:

```bash
oc get servicemonitor <k8s_name> -n <namespace>
```

If missing, check the actuator logs for errors during creation.

### Metrics Not Appearing in Prometheus

1. **Verify ServiceMonitor exists:**

   ```bash
   oc get servicemonitor <k8s_name> -n <namespace> -o yaml
   ```

2. **Check Service labels match:**

   ```bash
   oc get svc -n <namespace> -l serving.kserve.io/inferenceservice=<k8s_name>
   ```

3. **Verify InferenceService is ready:**

   ```bash
   oc get inferenceservice <k8s_name> -n <namespace>
   ```

4. **Check Prometheus operator logs:**

   ```bash
   oc logs -n openshift-user-workload-monitoring prometheus-operator-<pod-id>
   ```

5. **Verify user workload monitoring is enabled:**

   ```bash
   oc get configmap cluster-monitoring-config -n openshift-monitoring -o yaml
   ```

   Should show `enableUserWorkload: true`

### RBAC Issues

If you see permission errors, ensure the service account has permissions to create ServiceMonitors:

```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: servicemonitor-creator
  namespace: <namespace>
rules:
- apiGroups: ["monitoring.coreos.com"]
  resources: ["servicemonitors"]
  verbs: ["create", "get", "list", "delete"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: servicemonitor-creator-binding
  namespace: <namespace>
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: Role
  name: servicemonitor-creator
subjects:
- kind: ServiceAccount
  name: <service-account-name>
  namespace: <namespace>
```

## References

- [OpenShift Monitoring Documentation](https://docs.openshift.com/container-platform/latest/monitoring/enabling-monitoring-for-user-defined-projects.html)
- [Prometheus Operator ServiceMonitor](https://github.com/prometheus-operator/prometheus-operator/blob/main/Documentation/user-guides/getting-started.md)
- [KServe Metrics](https://kserve.github.io/website/latest/modelserving/observability/prometheus_metrics/)
