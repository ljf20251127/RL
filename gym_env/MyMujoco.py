import numpy as np
import mujoco

class MyMujoco:
    # origin_q须在关节限位内，可为9个关节，也可为7个
    # frame_skip为每次step，mujoco执行多少次mj_step
    # self.origin_q是9个关节的角度，两个夹抓设为0
    def __init__(self, xml_path, origin_q=None,frame_skip=11):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.frame_skip = frame_skip

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
            ], dtype=float)
        else: 
            if len(origin_q) < 7:
                raise ValueError(f"origin_q must has length >= 7, but got {len(origin_q)}")
            else:
                self.origin_q = origin_q[:7].copy()
        self.origin_q = np.append(self.origin_q, [0.0, 0.0])

        self.data = mujoco.MjData(
            self.model
        )
    
        # 将robot设为初始状态
        self.reset()

        # 定义site
        site_name = "panda_hand_tcp"
        self.site_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_SITE,
            site_name,
        )
        if self.site_id < 0:
            raise RuntimeError(
                f"Cannot find site: {site_name}"
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
    # q_target可以为9个关节，也可为7个
    # 执行frame_skip次mj_step
    def step(self, q_target):
        self.data.ctrl[:7] = q_target[:7]
        for _ in range(self.frame_skip):
            mujoco.mj_step(
                self.model,
                self.data
            )

    def get_pos(self):
        return self.data.site_xpos[self.site_id].copy()




