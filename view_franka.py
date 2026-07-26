import mujoco
import mujoco.viewer

# 修改成你的实际路径
xml_path = "./mujoco_menagerie/franka_emika_panda/scene.xml"

# 加载模型
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

# 打开 Viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()