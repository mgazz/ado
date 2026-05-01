# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Tests for Kubernetes name generation to ensure names stay within K8s limits."""

import pytest
from ado_actuators.vllm_performance.k8s.yaml_support.build_components import (
    ComponentsYaml,
)


class TestK8sNameGeneration:
    """Test suite for Kubernetes name generation."""

    def test_k8s_name_length_with_kserve_suffix(self) -> None:
        """
        Test that generated names stay within Kubernetes 63-character limit
        even with KServe suffixes like '-predictor'.
        """
        # Test with a long model name that previously caused issues
        long_model = "ibm-granite/granite.geospatial.prithvi-eo-2.0-300m-tl-se"
        k8s_name = ComponentsYaml.get_k8s_name(long_model)

        # Base name should be reasonable
        assert len(k8s_name) <= 50, f"Base name too long: {len(k8s_name)} chars"

        # With KServe's -predictor suffix (10 chars), must stay under 63
        full_name = f"{k8s_name}-predictor"
        assert (
            len(full_name) <= 63
        ), f"Name with suffix exceeds K8s limit: {len(full_name)} chars"

    def test_k8s_name_format(self) -> None:
        """Test that generated names follow expected format."""
        model = "meta-llama/Llama-3.1-8B-Instruct"
        k8s_name = ComponentsYaml.get_k8s_name(model)

        # Should start with vllm-
        assert k8s_name.startswith("vllm-"), "Name should start with 'vllm-'"

        # Should contain lowercase alphanumeric and hyphens only
        assert k8s_name.replace("-", "").isalnum(), "Name should be alphanumeric"
        assert k8s_name.islower(), "Name should be lowercase"

        # Should end with 8-character UUID
        parts = k8s_name.split("-")
        uuid_part = parts[-1]
        assert len(uuid_part) == 8, f"UUID part should be 8 chars, got {len(uuid_part)}"
        assert all(
            c in "0123456789abcdef" for c in uuid_part
        ), "UUID should be hexadecimal"

    def test_k8s_name_uniqueness(self) -> None:
        """Test that multiple calls generate unique names."""
        model = "test/model"
        names = [ComponentsYaml.get_k8s_name(model) for _ in range(10)]

        # All names should be unique due to UUID
        assert len(set(names)) == len(names), "Generated names should be unique"

    @pytest.mark.parametrize(
        "model",
        [
            "meta-llama/Llama-3.1-8B-Instruct",
            "ibm-granite/granite.geospatial.prithvi-eo-2.0-300m-tl-se",
            "mistralai/Mistral-7B-Instruct-v0.2",
            "google/gemma-2b",
            "very-long-organization-name/very-long-model-name-with-many-parts",
        ],
    )
    def test_various_model_names(self, model: str) -> None:
        """Test name generation with various model name formats."""
        k8s_name = ComponentsYaml.get_k8s_name(model)

        # Should always be valid K8s name
        assert len(k8s_name) <= 50, f"Name too long for model {model}"
        assert k8s_name.replace("-", "").isalnum(), f"Invalid characters in {k8s_name}"

        # With KServe suffix should stay under limit
        assert len(f"{k8s_name}-predictor") <= 63, "Name with suffix too long"


# Made with Bob
