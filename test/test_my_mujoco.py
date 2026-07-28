from gym_env.MyMujoco import MyMujoco

mujoco_env = MyMujoco(
    "robot_models/franka_emika_panda/scene.xml"
)
print(mujoco_env.get_random_angles())