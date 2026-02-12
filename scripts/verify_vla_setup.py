#!/usr/bin/env python3
"""
Verification script for VLA setup with automatic GPU detection.
Tests SmolVLA with GPU if available, falls back to CPU otherwise.
Includes performance benchmarking.
"""

import sys
import time


def check_step(description, check_fn):
    """Run a verification check and print result."""
    try:
        result = check_fn()
        print(f"✓ {description}: {result}")
        return True
    except Exception as e:
        print(f"✗ {description}: FAILED")
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("=" * 60)
    print("VLA Setup Verification (Auto GPU Detection)")
    print("=" * 60)
    print()

    all_passed = True

    # Check 1: PyTorch and GPU detection
    def check_pytorch_gpu():
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            compute_cap = torch.cuda.get_device_capability(0)
            return f"{torch.__version__} (GPU: {gpu_name}, sm_{compute_cap[0]}{compute_cap[1]})"
        else:
            return f"{torch.__version__} (CPU mode - no GPU detected)"

    all_passed &= check_step("PyTorch GPU support", check_pytorch_gpu)

    # Check 2: JAX GPU support
    def check_jax_gpu():
        import jax
        backend = jax.default_backend()
        devices = jax.devices()
        return f"{jax.__version__} (backend: {backend}, devices: {len(devices)})"

    all_passed &= check_step("JAX GPU support", check_jax_gpu)

    # Check 3: LeRobot
    def check_lerobot():
        import lerobot
        return f"version {lerobot.__version__}"

    all_passed &= check_step("LeRobot installed", check_lerobot)

    # Check 4: Determine device for SmolVLA
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n>>> Using device: {device.upper()}")
    if device == "cuda":
        print(f">>> GPU: {torch.cuda.get_device_name(0)}")
        print(f">>> Compute Capability: sm_{torch.cuda.get_device_capability(0)[0]}{torch.cuda.get_device_capability(0)[1]}")
    print()

    # Check 5: SmolVLA model load
    print("Loading SmolVLA model (450M params, ~1.8GB)...")
    print("This may take a minute on first run...")

    def check_smolvla():
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
        policy = policy.to(device).eval()

        # Verify model is on correct device
        first_param_device = next(policy.parameters()).device
        return f"loaded on {first_param_device}"

    all_passed &= check_step("SmolVLA model loaded", check_smolvla)

    # Check 6: SO-101 MuJoCo model
    def check_so101():
        import mujoco
        model = mujoco.MjModel.from_xml_path('assets/so101/so101.xml')
        return f"loaded ({model.nq} DOF, {model.nu} actuators)"

    all_passed &= check_step("SO-101 MuJoCo model loaded", check_so101)

    # Check 7: Single inference test
    def check_inference():
        import torch
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors

        # Load model on detected device
        policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
        policy = policy.to(device).eval()

        # Create preprocessor
        preprocessor, postprocessor = make_smolvla_pre_post_processors(
            policy.config,
            dataset_stats=None
        )

        # Create dummy observation on device
        observation = {
            'observation.images.camera1': torch.rand(1, 3, 256, 256, device=device),
            'observation.images.camera2': torch.rand(1, 3, 256, 256, device=device),
            'observation.images.camera3': torch.rand(1, 3, 256, 256, device=device),
            'observation.state': torch.rand(1, 6, device=device),
            'task': "Pick the red cube",
        }

        # Preprocess
        processed_obs = preprocessor(observation)

        # Run inference
        with torch.inference_mode():
            actions = policy.select_action(processed_obs)

        if actions is None or len(actions.shape) == 0:
            raise RuntimeError("Inference failed")

        return f"action shape: {actions.shape}, device: {actions.device}"

    all_passed &= check_step("Inference test", check_inference)

    # Check 8: Performance benchmark
    print("\n" + "=" * 60)
    print("Performance Benchmark")
    print("=" * 60)

    def benchmark_inference():
        import torch
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors

        print(f"Running 10 inference iterations on {device.upper()}...")

        # Load model
        policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
        policy = policy.to(device).eval()

        # Create preprocessor
        preprocessor, postprocessor = make_smolvla_pre_post_processors(
            policy.config,
            dataset_stats=None
        )

        # Warmup run (for both GPU and CPU)
        print(f"  Warmup run ({device.upper()} initialization)...")
        observation = {
            'observation.images.camera1': torch.rand(1, 3, 256, 256, device=device),
            'observation.images.camera2': torch.rand(1, 3, 256, 256, device=device),
            'observation.images.camera3': torch.rand(1, 3, 256, 256, device=device),
            'observation.state': torch.rand(1, 6, device=device),
            'task': "Pick the red cube",
        }
        processed_obs = preprocessor(observation)

        warmup_start = time.time()
        with torch.inference_mode():
            _ = policy.select_action(processed_obs)

        if device == "cuda":
            torch.cuda.synchronize()

        warmup_time = time.time() - warmup_start
        print(f"  Warmup completed: {warmup_time*1000:.2f} ms")

        # Benchmark runs
        num_runs = 10
        times = []

        for i in range(num_runs):
            # Create fresh observation each time
            observation = {
                'observation.images.camera1': torch.rand(1, 3, 256, 256, device=device),
                'observation.images.camera2': torch.rand(1, 3, 256, 256, device=device),
                'observation.images.camera3': torch.rand(1, 3, 256, 256, device=device),
                'observation.state': torch.rand(1, 6, device=device),
                'task': "Pick the red cube",
            }

            processed_obs = preprocessor(observation)

            start_time = time.time()
            with torch.inference_mode():
                actions = policy.select_action(processed_obs)

            # Synchronize GPU if using CUDA
            if device == "cuda":
                torch.cuda.synchronize()

            elapsed = time.time() - start_time
            times.append(elapsed)

            print(f"  Run {i+1:2d}/{num_runs}: {elapsed*1000:.2f} ms")

        # Statistics
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)

        print()
        print(f"  Average: {avg_time*1000:.2f} ms")
        print(f"  Min:     {min_time*1000:.2f} ms")
        print(f"  Max:     {max_time*1000:.2f} ms")

        return f"{avg_time*1000:.2f} ms average ({device.upper()})"

    all_passed &= check_step("\nPerformance benchmark completed", benchmark_inference)

    # Summary
    print()
    print("=" * 60)
    if all_passed:
        print("✓ All checks passed! SmolVLA setup complete!")
        print()
        if device == "cuda":
            print("GPU Setup Complete:")
            print("  - PyTorch with CUDA support")
            print(f"  - GPU: {torch.cuda.get_device_name(0)}")
            print(f"  - Compute Capability: sm_{torch.cuda.get_device_capability(0)[0]}{torch.cuda.get_device_capability(0)[1]}")
            print("  - JAX/MJX on GPU for physics")
            print("  - SmolVLA on GPU for VLA inference")
        else:
            print("CPU Setup Complete:")
            print("  - PyTorch in CPU mode")
            print("  - JAX for physics")
            print("  - SmolVLA on CPU for VLA inference")
            print()
            print("Note: For better performance, use an NVIDIA GPU with CUDA support.")
        print()
        print("Next: Run the demo!")
        print("  python scripts/demo_vla_inference.py")
        return 0
    else:
        print("✗ Some checks failed. See errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
