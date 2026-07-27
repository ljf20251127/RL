import pink
import pinocchio as pin
from pink.tasks import FrameTask, PostureTask

class MyPink:
    def __init__(self, urdf_path, frame_name, q):
        # 初始关节角度
        self.q = q

        model = pin.buildModelFromUrdf(
        urdf_path
        )

        data = model.createData()

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

        
