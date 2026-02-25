import numpy as np
import pyrealsense2 as rs
from xarm.wrapper import XArmAPI
import cv2
import time
import threading
from collections import deque

from openpi.policies import policy_config
from openpi.shared import download
from openpi.training import config as _config
from openpi.models.tokenizer import PaligemmaTokenizer


# =========================
# User inputs
# =========================
FPS = 60.0 # still finding upper limit on this
DT = 1.0 / FPS
CONTROL_HZ = 150.0 # multiple of 10
PREDICTION_HORIZON = 20
MIN_EXECUTION_HORIZON = 10
ROBOT_DOF = 7

mutex = threading.Lock()
condition_variable = threading.Condition(mutex)

delay_init = 5
buffer_size = 5


# =========================
# Policy Setup
# =========================
config = _config.get_config("pi05_xarm_finetune")
checkpoint_dir = download.maybe_download(
    "/home/admin/openpi/checkpoints/pi05_xarm_finetune/clara_training1/25000"
)

policy = policy_config.create_trained_policy(config, checkpoint_dir)


# =========================
# XArm Setup
# =========================
arm = XArmAPI('192.168.1.222')

if arm.get_state() != 0:
    arm.clean_error()
    time.sleep(0.5)

arm.motion_enable(enable=True)
arm.set_mode(1)
arm.set_state(0)
arm.set_gripper_enable(enable=True)
arm.set_gripper_mode(0)


# =========================
# RealSense Camera Setup
# =========================
ctx = rs.context()
devices = ctx.query_devices()

if len(devices) < 2:
    raise RuntimeError("Need at least two RealSense cameras connected")

serials = [dev.get_info(rs.camera_info.serial_number) for dev in devices]
print("Found cameras:", serials)

pipelines = []
configs = []

for serial in serials:
    pipeline = rs.pipeline()
    config_rs = rs.config()
    config_rs.enable_device(serial)
    config_rs.enable_stream(rs.stream.color, 320, 240, rs.format.rgb8, 30)
    pipeline.start(config_rs)
    pipelines.append(pipeline)
    configs.append(config_rs)


# =========================
# Observation
# =========================
def get_observation():
    frames_wrist = pipelines[1].wait_for_frames()
    frames_exterior = pipelines[0].wait_for_frames()

    wrist = frames_wrist.get_color_frame()
    exterior = frames_exterior.get_color_frame()

    a = np.asanyarray(wrist.get_data())
    b = np.asanyarray(exterior.get_data())

    pose = arm.get_position()[1]
    pose[3] = pose[3] % 360
    pose[5] = pose[5] % 360
    angles_rad = (np.array(pose[3:6]) * np.pi / 180).tolist()
    state = np.array(pose[:3] + angles_rad, dtype=np.float32)

    _, g_p = arm.get_gripper_position()
    g_p = np.array((g_p - 850) / -860)

    observation = {
        "observation/exterior_image_1_left": b,
        "observation/wrist_image_left": a,
        "observation/gripper_position": g_p,
        "observation/joint_position": state,
        "prompt": "Pick up the bag and place it on the blue x",
    }

    return observation

# =========================
# Command Interpolation
# =========================
def interpolate_action(state, goal):
    delta_increment = (goal - state) / (DT * CONTROL_HZ * 6)

    for i in range(int(DT * CONTROL_HZ)):
        start_time = time.perf_counter()
        state += delta_increment
        command = state.copy()
        command[3] = (command[3]+ 180) % 360 -180
        command[5] = (command[5]+ 180) % 360 -180

        x, y, z, roll, pitch, yaw = command
        print("Interpolation:")
        print(x, y, z, roll, pitch, yaw)

        arm.set_servo_cartesian(command, speed=100, mvacc=1000)

        time_left = (1 / CONTROL_HZ) - (time.perf_counter() - start_time)
        time.sleep(max(time_left,0))

# =========================
# Action Getter
# =========================
def get_action(observation_next):
    global t, observation_curr

    with condition_variable:
        t += 1

        observation_curr = observation_next
        condition_variable.notify()

        action = action_curr[t - 1, :].copy()

    return action

# =========================
# Guided Inference
# =========================
def guided_inference(policy, observation, action_prev, delay, time_since_last_inference):
    H = PREDICTION_HORIZON
    i = np.arange(delay, H - time_since_last_inference)
    c = (H - time_since_last_inference - i) / (H - time_since_last_inference - delay + 1)

    W = np.ones(H)
    W[0:delay] = 1.0
    W[delay:H - time_since_last_inference] = c * (np.exp(c) - 1) / (np.exp(1) - 1)
    W[H - time_since_last_inference:] = 0.0

    T, robot_dof = action_prev.shape
    if T < H:
        action_prev = np.pad(action_prev, ((0, H - T), (0, 0)), mode='constant')

    v_pi = np.array(policy.infer(observation)["actions"])
    v_pi = v_pi[:H, :ROBOT_DOF]  # ensure correct shape

    A = action_prev.copy()
    action_estimate = A*W[:,None] + v_pi*(1-W[:, None])

    return action_estimate[:H, :ROBOT_DOF]


# =========================
# Inference Loop
# =========================
def inference_loop():
    global t, action_curr, observation_curr

    Q = deque([delay_init], maxlen=buffer_size)

    while True:
        with condition_variable:
            while t < MIN_EXECUTION_HORIZON:
                condition_variable.wait()

            time_since_last_inference = t
            # Remove actions that have already been executed
            action_prev = action_curr[
                time_since_last_inference:PREDICTION_HORIZON
            ].copy() 

            delay = max(Q)
            print("Delay: ", delay)
            obs = observation_curr.copy()

        # ---- lock released ----

        action_new = guided_inference(
            policy,
            obs,
            action_prev,
            delay,
            time_since_last_inference
        )

        action_curr[:action_new.shape[0], :] = action_new
        t = t - time_since_last_inference
        Q.append(t)


# =========================
# Execution Loop
# =========================
def execution_loop():
    global t
    
    while True:
        print("t:", t)
        t0 = time.perf_counter()

        observation = get_observation()
        command = get_action(observation)

        cmd_joint_pose = command[:6].copy()
        cmd_joint_pose[3:6] = cmd_joint_pose[3:6] / np.pi * 180

        pose = arm.get_position()[1]
        pose[3] = pose[3] % 360
        pose[5] = pose[5] % 360
        state = np.array(pose, dtype=np.float32)

        print("Current pose:")
        print(pose)
        print("Command pose:")
        print(cmd_joint_pose)

        # execute smooth motion to target via interpolation
        interpolate_action(state, cmd_joint_pose)
        cmd_gripper_pose = (command[6]) * -860 + 850 # unnormalize the gripper action
        arm.set_gripper_position(cmd_gripper_pose)

        time_left = DT - (time.perf_counter() - t0)
        time.sleep(max(time_left, 0))


# =========================
# Shared State
# =========================
t = 0
observation_curr = get_observation()
action_curr = np.array(policy.infer(observation_curr)["actions"], dtype=np.float32)

# =========================
# Thread Startup
# =========================
if __name__ == "__main__":
    print("Starting control system...")

    infer_thread = threading.Thread(target=inference_loop, daemon=True)
    exec_thread = threading.Thread(target=execution_loop, daemon=True)

    infer_thread.start()
    exec_thread.start()

    while True:
        time.sleep(1)