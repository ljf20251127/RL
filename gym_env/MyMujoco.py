import numpy as np
import mujoco

class MyMujoco:
    # origin_q须在关节限位内，9个关节
    def __init__(self, xml_path, origin_q=None):
        self.model = mujoco.MjModel.from_xml_path(xml_path)

        # 9个关节限位
        self.q_min = np.array([self.model.jnt_range[i][0] for i in range(self.model.njnt)])
        self.q_max = np.array([self.model.jnt_range[i][1] for i in range(self.model.njnt)])

        if origin_q is None:
            # 默认为限位中值，2个夹抓关节设为0
            self.origin_q = np.array([
                0.0,
                0.0,
                0.0,
                -1.57,
                0.0,
                1.87,
                0.0,
                0.0,
                0.0
            ], dtype=float)
        else:
            self.origin_q = origin_q.copy()

        self.data = mujoco.MjData(
            self.model
        )

    #随机关节角度，2个夹抓设为0
    def get_random_angles(self):
        random_q = np.zeros(self.model.nq)
        random_q[:7] = np.random.uniform(
            self.q_min[:7], 
            self.q_max[:7]
        )
        random_q[7:] = 0.0
        return random_q
    
    def reset(self):
        self.data.qpos[:] = self.origin_q.copy()
        self.data.qvel[:] = 0
        self.data.ctrl[:] = 0

        mujoco.mj_forward(
        self.model,
        self.data
    )
