import jax
import mujoco
from mujoco import mjx
import time

print("GPU:", jax.devices())

model = mujoco.MjModel.from_xml_path('assets/franka_mjx_simple.xml')
mjx_model = mjx.put_model(model)

# Test with different batch sizes
for batch_size in [1, 16, 64, 256]:
    print(f"\n--- Batch size: {batch_size} ---")
    
    # Create batched initial state
    mjx_data = jax.vmap(lambda _: mjx.make_data(mjx_model))(jax.numpy.arange(batch_size))
    
    # JIT compile
    step_fn = jax.jit(jax.vmap(lambda d: mjx.step(mjx_model, d)))
    
    # Warmup
    mjx_data = step_fn(mjx_data)
    mjx_data.qpos.block_until_ready()
    
    # Benchmark
    n_steps = 1000
    start = time.time()
    for _ in range(n_steps):
        mjx_data = step_fn(mjx_data)
    mjx_data.qpos.block_until_ready()
    elapsed = time.time() - start
    
    total_steps = n_steps * batch_size
    print(f"  {total_steps} total steps in {elapsed:.3f}s")
    print(f"  {total_steps/elapsed:.0f} steps/sec")
    print(f"  {elapsed/n_steps*1000:.2f} ms/batch")