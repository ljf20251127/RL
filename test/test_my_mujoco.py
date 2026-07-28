import mujoco
import mujoco.viewer
from gym_env.MyMujoco import MyMujoco

mujoco_env = MyMujoco(
    "robot_models/franka_emika_panda/scene.xml"
)
print(mujoco_env.get_random_angles())

# 打开 Viewer
with mujoco.viewer.launch_passive(mujoco_env.model, mujoco_env.data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(mujoco_env.model, mujoco_env.data)
        viewer.sync()