#!/usr/bin/env python3
"""
Verification script for VLA setup.
Checks that JAX, PyTorch, LeRobot, and SmolVLA are properly installed and working.
"""

import sys


def check_step(description, check_fn):
    """Run a verification check and print result."""
    try:
        result = check_fn()
        print(f"✓ {description}: {result}")
        return True
    except Exception as e:
        print(f"✗ {description}: FAILED")
        print(f"  Error: {e}")
        return False


def main():
    print("=" * 60)
    print("VLA Setup Verification")
    print("=" * 60)
    print()

    all_passed = True

    # Check 1: JAX import and CUDA
    def check_jax():
        import jax
        backend = jax.default_backend()
        if backend != "gpu" and backend != "cuda":
            raise RuntimeError(f"JAX backend is '{backend}', expected 'gpu' or 'cuda'")
        return f"{jax.__version__} (backend: {backend})"

    all_passed &= check_step("JAX installed with CUDA", check_jax)

    # Check 2: PyTorch import and CUDA
    def check_pytorch():
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("PyTorch CUDA not available")
        return f"{torch.__version__} (CUDA available: True)"

    all_passed &= check_step("PyTorch installed with CUDA", check_pytorch)

    # Check 3: LeRobot import
    def check_lerobot():
        import lerobot
        return f"version {lerobot.__version__}"

    all_passed &= check_step("LeRobot installed", check_lerobot)

    # Check 4: SmolVLA model download and load
    print("\nDownloading SmolVLA model (450M params, ~1.8GB)...")
    print("This may take a few minutes on first run...")

    def check_smolvla():
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
        return "model loaded successfully"

    all_passed &= check_step("SmolVLA model downloaded and loaded", check_smolvla)

    # Check 5: SO-101 MuJoCo model load
    def check_so101():
        import mujoco
        model = mujoco.MjModel.from_xml_path('assets/so101/so101.xml')
        return f"loaded ({model.nq} DOF, {model.nu} actuators)"

    all_passed &= check_step("SO-101 MuJoCo model loaded", check_so101)

    # Check 6: Test inference on dummy observation
    def check_inference():
        import torch
        import numpy as np
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors

        # Load model
        policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base").eval()

        # Create preprocessor
        preprocessor, postprocessor = make_smolvla_pre_post_processors(
            policy.config,
            dataset_stats=None
        )

        # Create dummy observation (use camera1, camera2, camera3 naming)
        observation = {
            'observation.images.camera1': torch.rand(1, 3, 256, 256),
            'observation.images.camera2': torch.rand(1, 3, 256, 256),
            'observation.images.camera3': torch.rand(1, 3, 256, 256),
            'observation.state': torch.rand(1, 6),
            'task': "Pick the red cube",  # String, not list
        }

        # Preprocess observation
        processed_obs = preprocessor(observation)

        # Run inference
        with torch.inference_mode():
            actions = policy.select_action(processed_obs)

        if actions is None or len(actions.shape) == 0:
            raise RuntimeError("Inference failed to produce actions")

        return f"action shape: {actions.shape}"

    all_passed &= check_step("Test inference passed", check_inference)

    # Summary
    print()
    print("=" * 60)
    if all_passed:
        print("✓ All checks passed! SmolVLA setup complete.")
        print()
        print("Next steps:")
        print("  1. Run demo: python scripts/demo_vla_inference.py")
        print("  2. See README.md for more information")
        return 0
    else:
        print("✗ Some checks failed. Please review the errors above.")
        print()
        print("Common fixes:")
        print("  - Ensure NVIDIA drivers are installed (nvidia-smi)")
        print("  - Ensure venv is activated: source venv/bin/activate")
        print("  - Try reinstalling: pip install --upgrade \"lerobot[smolvla]\"")
        return 1


if __name__ == "__main__":
    sys.exit(main())
