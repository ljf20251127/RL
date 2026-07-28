import mujoco
import numpy as np

mj_model = mujoco.MjModel.from_xml_path("robot_models/franka_emika_panda/scene.xml")

# 从 model 中提取关节限位（自动匹配）
q_min = np.array([mj_model.jnt_range[i][0] for i in range(mj_model.njnt)])
q_max = np.array([mj_model.jnt_range[i][1] for i in range(mj_model.njnt)])

q_min = q_min[:-2]
q_max = q_max[:-2]
q_mean = 0.5 * (q_min + q_max)

print("q_min:", q_min)
print("q_max:", q_max)
print("q_mean:", q_mean)

q_random = np.random.uniform(q_min, q_max)

print("q_random:", q_random)

# 没有site
for i in range(mj_model.nsite):
    print(i, mj_model.site(i).name)

print("mj_model.nsite:", mj_model.nsite)

# 查看所有 body 名称
# for i in range(mj_model.nbody):
#    print(f"Body {i}: {mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_BODY, i)}")

mj_data = mujoco.MjData(
    mj_model
)

site_id = mujoco.mj_name2id(
    mj_model,
    mujoco.mjtObj.mjOBJ_SITE,
    "panda_hand_tcp"
)
q = np.array([
    0.0,
    -0.785398,
    0.0,
    -2.35619,
    0.0,
    1.5708,
    0.785398,
    0,
    0,
])

mj_data.qpos[:9] = q

mujoco.mj_forward(
    mj_model,
    mj_data
)

mujoco_pos = mj_data.site_xpos[site_id]
    
print("MuJoCo pos:", mujoco_pos)
