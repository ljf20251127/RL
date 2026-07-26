import mujoco
import numpy as np

model = mujoco.MjModel.from_xml_path("franka_emika_panda/scene.xml")

# 从 model 中提取关节限位（自动匹配）
q_min = np.array([model.jnt_range[i][0] for i in range(model.njnt)])
q_max = np.array([model.jnt_range[i][1] for i in range(model.njnt)])

q_min = q_min[:-2]
q_max = q_max[:-2]

print("q_min:", q_min)
print("q_max:", q_max)

q_random = np.random.uniform(q_min, q_max)

print("q_random:", q_random)

# 没有site
for i in range(model.nsite):
    print(i, model.site(i).name)

print("model.nsite:", model.nsite)

# 查看所有 body 名称
for i in range(model.nbody):
    print(f"Body {i}: {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)}")


    
