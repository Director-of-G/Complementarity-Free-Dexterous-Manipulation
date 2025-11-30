from copy import deepcopy
import argparse
import time
import os
import numpy as np
from scipy.spatial.transform import Rotation as SciR
import pickle

from examples.mpc.allegro.ball.params import ExplicitMPCParams
from planning.mpc_explicit import MPCExplicit
from envs.allegro_env import MjSimulator
from envs.allegro_env_rpy_joint import MjSimulator as MjSimulatorRPYJoint
from contact.allegro_collision_detection import Contact

from utils import metrics


MUJOCO_DEFAULT_TIMESTEP = 0.002

def main(result_folder=None):
    if result_folder is None:
        result_folder = f'{time.strftime("%Y%m%d-%H%M%S")}'
    save_dir_root = "/home/jyp/research/inhand_manipulation/dex_playground/ros2_ws/src/complementarity_free_control/src/Complementarity-Free-Dexterous-Manipulation/data"
    save_dir_name = os.path.join(save_dir_root, result_folder)

    if not os.path.exists(save_dir_name):
        os.makedirs(save_dir_name)
    rand_so3_data = np.load('/home/jyp/research/inhand_manipulation/ros2_ws/src/inhand_lowlevel/leap_ros2/scripts/journal/data/quat_targets-250318.npy')
    num_targets = len(rand_so3_data)

    # -------------------------------
    #       loop trials
    # -------------------------------
    save_flag=False
    if save_flag:
        save_dir = './examples/mpc/allegro/ball/save/'
        prefix_data_name = 'ours_'
        save_data = dict()

    trial_num = 20
    success_pos_threshold = 0.02
    success_quat_threshold = 0.04
    consecutive_success_time_threshold = 20
    max_rollout_length = 500

    # ----------------------------------------
    #       journal experiment parameters
    # ----------------------------------------
    # timeout = 60.0
    mpc_freq = 30
    timeout = 60.0
    regrasp_every_niter = 100
    verbose_every_niter = 100

    trial_count = 0
    # for trial_count in range(num_targets):
    for trial_count in [0]:
        cmd_buf = []
        cmd_stamp = []
        state_buf = []
        state_stamp = []

        # -------------------------------
        #        init parameters
        # -------------------------------
        param = ExplicitMPCParams(rand_seed=trial_count, target_type='rotation')
        param.target_q_ = rand_so3_data[trial_count]
        # param.target_q_ = SciR.from_euler('xyz', [np.pi / 6, 0, 0]).as_quat()[[3, 0, 1, 2]]
        timestep = param.frame_skip_ * MUJOCO_DEFAULT_TIMESTEP

        # -------------------------------
        #        init contact
        # -------------------------------
        contact = Contact(param)

        # -------------------------------
        #        init envs
        # -------------------------------
        env = MjSimulatorRPYJoint(param)
        param_grad_compute = deepcopy(param)
        param_grad_compute.model_path_ = '/home/jyp/research/inhand_manipulation/dex_playground/ros2_ws/src/complementarity_free_control/src/Complementarity-Free-Dexterous-Manipulation/envs/xmls/env_allegro_ball_nominal.xml'
        env_grad_compute = MjSimulatorRPYJoint(param_grad_compute, launch_viewer=False)
        # env_grad_compute = env

        # -------------------------------
        #        init planner
        # -------------------------------
        mpc = MPCExplicit(param)
        explicit_model = mpc.model

        # -------------------------------
        #        MPC rollout
        # -------------------------------
        mpc_iter = 0
        consecutive_success_time = 0
        mpc_solve_time = []

        t_wall_clock_start = time.time()
        rollout_q_traj = []
        for mpc_iter in range(int(timeout * mpc_freq)):
            if not env.dyn_paused_:
                verbose = mpc_iter % verbose_every_niter == 0
                regrasp = mpc_iter % regrasp_every_niter == 0
                if regrasp:
                    print("Regrasping...")
                    env.reset_fingers_qpos()

                # get state
                curr_q = env.get_state()
                env_grad_compute.reset_env_with_qpos(curr_q[7:], curr_q[:7])

                rollout_q_traj.append(curr_q)

                # -----------------------
                #     contact detect
                # -----------------------
                phi_vec, jac_mat = contact.detect_once(env_grad_compute)
                # jac_mat[:, 3:6] = -jac_mat[:, 3:6]

                # -----------------------
                #        planning
                # -----------------------
                obj_quat = curr_q[3:7]
                target_quat = param.target_q_
                if np.dot(obj_quat, target_quat) < 0:
                    target_quat = -target_quat

                t_mpc_start = time.time()
                sol = mpc.plan_once(
                    param.target_p_,
                    # param.target_q_,
                    target_quat,
                    curr_q,
                    phi_vec,
                    jac_mat,
                    sol_guess=param.sol_guess_,
                    robot_qpos0=param.init_robot_qpos_,
                    verbose=verbose)
                mpc_solve_time.append(time.time() - t_mpc_start)

                param.sol_guess_ = sol['sol_guess']
                action = sol['action']

                # # eval cost terms
                cost_param = np.hstack([param.target_p_, param.target_q_, phi_vec, jac_mat.flatten('F'), param.init_robot_qpos_])
                # contact_cost = param.contact_cost_fn(curr_q, cost_param).full().item()
                quaternion_cost = param.quaternion_cost_fn(curr_q, cost_param).full().item()
                angle_error = param.angle_error_fn(curr_q, cost_param).full().item()
                # robot_qpos_reg_cost = param.robot_qpos_reg_cost_fn(curr_q, cost_param).full().item()
                # control_cost = param.control_cost_fn(action).full().item()
                
                if verbose:
                    print("Step: ", mpc_iter)
                    # print("--------------------")
                    # print(f"contact cost: {contact_cost}\nquaternion cost: {quaternion_cost}\nrobot qpos reg cost: {robot_qpos_reg_cost}\ncontrol cost: {control_cost}")
                    print(f"opt cost: {sol['cost_opt']}")
                    print(f"angle error: {angle_error}")
                    # print("--------------------")

                # -----------------------
                #        simulate
                # -----------------------
                # env.step(action)
                env.step(action)

                # -----------------------
                #       record data for journal
                # -----------------------
                cmd_buf.append(env.latest_cmd.tolist())
                full_state = env.get_state()
                minimum_state = full_state[3:]  # no object position
                state_buf.append(minimum_state.tolist())
                curr_time = mpc_iter * (1 / mpc_freq)
                cmd_stamp.append(curr_time)
                state_stamp.append(curr_time)

                if verbose:
                    wall_clock_time = time.time() - t_wall_clock_start
                    sim_time = curr_time
                    print(f"Wall clock time: {wall_clock_time:.2f}s, Simulated time: {sim_time:.2f}s, Real-time factor: {sim_time / wall_clock_time:.2f}x")

                # -----------------------
                #        success check
                # -----------------------
                curr_q = env.get_state()
                if (metrics.comp_quat_error(curr_q[3:7], param.target_q_) < success_quat_threshold):
                    consecutive_success_time = consecutive_success_time + 1
                else:
                    consecutive_success_time = 0

        # -------------------------------
        #        close viewer
        # -------------------------------
        env.viewer_.close()

        data = {
            'high_freq': 1 / np.mean(mpc_solve_time),
            'low_freq': np.nan,
            'cmd_buf': cmd_buf,
            'cmd_stamp': cmd_stamp,
            'state_buf': state_buf,
            'state_stamp': state_stamp
        }

        print(f"state_buf's shape: ", len(state_buf))

        pickle.dump(data, open(f'{save_dir_name}/goal_{trial_count}.pkl', 'wb'))

    print("=== All trials completed. ===")


if __name__ == "__main__":
    # use args
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_folder', type=str, default=None, help='Folder name to save results.')
    args = parser.parse_args()

    main(result_folder=args.result_folder)
