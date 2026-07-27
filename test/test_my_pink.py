from gym_env.MyPink import MyPink
import numpy as np
import time

t0 = time.time()
q = np.array([
    0.0,
    -0.785398,
    0.0,
    -2.35619,
    0.0,
    1.5708,
    0.785398,
    0.0,
    0.0,
])

pink_env = MyPink(
    "robot_models/panda.urdf",
    "panda_hand_tcp",
    q
)
print("init:", time.time() - t0)
target_position = np.array([
    0.5,
    0.2,
    0.4,
])

result_q = pink_env.solve(target_position)

print(result_q)
print("solve:", time.time() - t0)