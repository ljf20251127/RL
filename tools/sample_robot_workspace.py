import numpy as np
import mujoco
from gym_env.MyMujoco import MyMujoco

mujoco_env = MyMujoco(
    "robot_models/franka_emika_panda/scene.xml"
)

site_name = "panda_hand_tcp"

site_id = mujoco.mj_name2id(
    mujoco_env.model,
    mujoco.mjtObj.mjOBJ_SITE,
    site_name,
)

if site_id < 0:
    raise RuntimeError(
        f"Cannot find site: {site_name}"
    )

print("site id:", site_id)
pos = mujoco_env.data.site_xpos[site_id].copy()
print(pos)

'''
num_samples = 100000
positions = []
for _ in range(num_samples):
    q = mujoco_env.get_random_angles()
    mujoco_env.data.qpos[:] = q
    mujoco.mj_forward(
        mujoco_env.model,
        mujoco_env.data
    )
    pos = mujoco_env.data.site_xpos[site_id].copy()
    positions.append(pos)

positions = np.array(positions)

xyz_min = positions.min(axis=0)

xyz_max = positions.max(axis=0)

xyz_mean = positions.mean(axis=0)

print("\nWorkspace:")

print("x range:", xyz_min[0], xyz_max[0])

print("y range:", xyz_min[1], xyz_max[1])

print("z range:", xyz_min[2], xyz_max[2])

print("\nmean:")

print(xyz_mean)

'''