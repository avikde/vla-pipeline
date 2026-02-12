import mujoco
import time

model = mujoco.MjModel.from_xml_path('assets/so101/so101.xml')
data = mujoco.MjData(model)

# Benchmark CPU
n_steps = 10000
start = time.time()
for _ in range(n_steps):
    mujoco.mj_step(model, data)
elapsed = time.time() - start

print(f"CPU: {n_steps/elapsed:.0f} steps/sec")
