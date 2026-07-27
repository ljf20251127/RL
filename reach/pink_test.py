import numpy as np
import qpsolvers

from robot_descriptions.loaders.pinocchio import load_robot_description

import pink
from pink import solve_ik
from pink.tasks import FrameTask, PostureTask
from pink.utils import custom_configuration_vector
import pinocchio as pin

# ==========================
# 1. 加载 Panda URDF
# ==========================



urdf_path = "robot_models/panda.urdf"

model = pin.buildModelFromUrdf(
    urdf_path
)

data = model.createData()


# ==========================
# 2. 定义任务
# ==========================

# Panda末端frame名称
frame_name = "panda_hand_tcp"


# 控制末端位置
end_effector_task = FrameTask(
    frame_name,
    position_cost=1.0,
    orientation_cost=0.0,   # 只控制位置，不控制姿态
    lm_damping=1.0,
)


# 保持关节姿态，避免IK多解跳变
posture_task = PostureTask(
    cost=1e-3,
)


tasks = [
    end_effector_task,
    posture_task,
]


# ==========================
# 3. 设置初始关节角
# ==========================

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


configuration = pink.Configuration(
    model,
    data,
    q,
)


# configuration.update(q)


# posture目标设为当前姿态
posture_task.set_target_from_configuration(
    configuration
)


# ==========================
# 4. 查看当前末端位置
# ==========================

T_current = configuration.get_transform_frame_to_world(
    frame_name
)

print("当前末端位置:")
print(T_current.translation)


# ==========================
# 5. 设置目标末端位置
# ==========================

target_position = np.array([
    0.5,
    0.2,
    0.4,
])


# 使用当前姿态，只改变位置

T_target = T_current.copy()

T_target.translation = target_position


end_effector_task.set_target(
    T_target
)


# ==========================
# 6. IK循环
# ==========================

dt = 0.01


solver = "daqp"


for i in range(200):

    velocity = solve_ik(
        configuration,
        tasks,
        dt,
        solver=solver,
    )


    configuration.integrate_inplace(
        velocity,
        dt,
    )
    T_now = configuration.get_transform_frame_to_world(frame_name)
    error = np.linalg.norm(
        target_position - T_now.translation
    )
    if error < 1e-4:
        print("error<1e-4 step:", i)
        break


    if i % 20 == 0:

        T_now = configuration.get_transform_frame_to_world(
            frame_name
        )

        print(
            "step:",
            i,
            "position:",
            T_now.translation
        )


# ==========================
# 7. 最终关节角
# ==========================

q_final = configuration.q


print("\n最终关节角:")
print(q_final)


# ==========================
# 8. 验证最终末端位置
# ==========================

T_final = configuration.get_transform_frame_to_world(
    frame_name
)


print("\n最终末端位置:")
print(T_final.translation)


print("\n误差:")
print(
    np.linalg.norm(
        T_final.translation - target_position
    )
)