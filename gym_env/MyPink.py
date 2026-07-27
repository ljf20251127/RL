import numpy as np
import pink
import pinocchio as pin
from pink.tasks import FrameTask, PostureTask

class MyPink:
    def __init__(self, urdf_path, frame_name, q, dt=0.01, solver="daqp"):

        self.frame_name = frame_name
        self.dt = dt
        self.solver = solver

        model = pin.buildModelFromUrdf(
        urdf_path
        )
        data = model.createData()

        # 初始configuration
        self.configuration = pink.Configuration(
            model,
            data,
            q,
        )

        # 控制末端位置
        self.end_effector_task = FrameTask(
            self.frame_name,
            position_cost=1.0,
            orientation_cost=0.0,   # 只控制位置，不控制姿态
            lm_damping=1.0,
        )


        # 保持关节姿态，避免IK多解跳变
        self.posture_task = PostureTask(
            cost=1e-3,
        )


        self.tasks = [
            self.end_effector_task,
            self.posture_task,
        ]

    def solve(self, target_position):
        # 姿态目标即为原姿态
        self.posture_task.set_target_from_configuration(
            self.configuration
        )
        T_current = self.configuration.get_transform_frame_to_world(
            self.frame_name
        )

        # 只改变位置，不改变其他
        T_target = T_current.copy()
        T_target.translation = target_position
        self.end_effector_task.set_target(
            T_target
        )

        # IK循环
        for _ in range(200):
            velocity = pink.solve_ik(
                self.configuration,
                self.tasks,
                self.dt,
                self.solver
            )
            self.configuration.integrate_inplace(
                velocity,
                self.dt
            )

            # 对比误差
            T_now = self.configuration.get_transform_frame_to_world(self.frame_name)
            error = np.linalg.norm(
                target_position - T_now.translation
            )
            if error < 1e-4:
                break

        return self.configuration.q
