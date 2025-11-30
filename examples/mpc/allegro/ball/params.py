import casadi as cs
import numpy as np
from scipy.spatial.transform import Rotation as SciR

import envs.allegro_fkin as allegro_fk
from utils import rotations


def generate_quat_target():
    deg_limit = 135

    quat_target = None
    while True:
        rand_so3 = SciR.random()
        rvec = rand_so3.as_rotvec()
        ang = np.linalg.norm(rvec)
        if np.abs(ang) < np.deg2rad(deg_limit):
            quat_target = rand_so3.as_quat()[[3, 0, 1, 2]]
            break

    return quat_target

def quat_wxyz_to_xyzw(q):
    # q 可以是 numpy 或 casadi，都支持下标访问
    return cs.vcat([q[1], q[2], q[3], q[0]])

def quat_angle(q1_wxyz, q2_wxyz):
    """
    输入: q1_wxyz (4,), q2_wxyz (4,)  皆为 wxyz 排序
    输出: 两个 quaternion 之间的旋转角（弧度）
    """
    # -------- Convert to xyzw --------
    q1 = quat_wxyz_to_xyzw(q1_wxyz)
    q2 = quat_wxyz_to_xyzw(q2_wxyz)

    # -------- Normalize (required) --------
    q1 = q1 / cs.norm_2(q1)
    q2 = q2 / cs.norm_2(q2)

    # -------- Quaternion inner product --------
    dot = cs.dot(q1, q2)

    # Clamp 以防数值误差导致 acos 域错误
    dot = cs.fmax(cs.fmin(dot, 1.0), -1.0)

    # -------- Angle difference --------
    return 2.0 * cs.acos(cs.fabs(dot))

class ExplicitMPCParams:
    def __init__(self, rand_seed=1, target_type='rotation'):
        # ---------------------------------------------------------------------------------------------
        #      simulation parameters 
        # ---------------------------------------------------------------------------------------------
        # self.model_path_ = 'envs/xmls/env_allegro_ball.xml'
        self.model_path_ = '/home/jyp/research/inhand_manipulation/dex_playground/ros2_ws/src/complementarity_free_control/src/Complementarity-Free-Dexterous-Manipulation/envs/xmls/env_allegro_ball.xml'
        self.object_names_ = ['object_geom']

        self.h_ = 0.1
        self.frame_skip_ = int(17)  # Avg. MPC solve time around 0.04s, that is 20 * MuJoCo's timestep 0.002 

        # system dimensions:
        self.n_robot_qpos_ = 16
        self.n_qpos_ = 23   # was 20
        self.n_qvel_ = 22   # was 19
        self.n_cmd_ = 16

        # ---------------------------------------------------------------------------------------------
        #      initial state and target state
        # ---------------------------------------------------------------------------------------------
        np.random.seed(42 + rand_seed)

        self.init_robot_qpos_ = np.array([
            0.2, 0.95, 1.0, 1.0,
            0.0, 0.6, 1.0, 1.0,
            -0.2, 0.95, 1.0, 1.0,
            0.6, 1.95, 1.0, 1.0
        ])

        # random init and target pose for object
        if target_type == 'rotation':
            init_obj_pos = np.hstack([-0.035, 0.0, 0.072])
            init_yaw_angle = 0.0
            init_obj_quat_rand = rotations.rpy_to_quaternion(np.hstack([init_yaw_angle, 0, 0]))
            self.init_obj_qpos_ = np.hstack((init_obj_pos, init_obj_quat_rand))

            self.target_p_ = np.array([-0.035, 0.0, 0.072])   # not used
            yaw_angle = init_yaw_angle + np.random.uniform(-np.pi/2, np.pi/2)
            # self.target_q_ = rotations.rpy_to_quaternion(np.hstack([yaw_angle, 0, 0]))
            self.target_q_ = generate_quat_target()
            # print("target rpy: ", np.hstack([yaw_angle, 0, 0]))
        else:
            raise ValueError(f'Target type {target_type} not supported')

        # ---------------------------------------------------------------------------------------------
        #      contact parameters
        # ---------------------------------------------------------------------------------------------
        self.mu_object_ = 0.5
        self.n_mj_q_ = self.n_qpos_
        self.n_mj_v_ = self.n_qvel_
        self.max_ncon_ = 40

        # ---------------------------------------------------------------------------------------------
        #      models parameters
        # ---------------------------------------------------------------------------------------------
        self.obj_inertia_ = np.identity(6)
        self.obj_inertia_[0:3, 0:3] = 50 * np.eye(3)
        self.obj_inertia_[3:, 3:] = 0.01 * np.eye(3)
        self.robot_stiff_ = np.diag(self.n_cmd_ * [0.1])

        Q = np.zeros((self.n_qvel_, self.n_qvel_))
        Q[:6, :6] = self.obj_inertia_
        Q[6:, 6:] = self.robot_stiff_
        self.Q = Q

        self.obj_mass_ = 0.01
        # self.gravity_ = np.array([0.00, 0.00, -9.8, 0.0, 0.0, 0.0])
        self.gravity_ = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        self.model_params = 0.5

        # ---------------------------------------------------------------------------------------------
        #      planner parameters
        # ---------------------------------------------------------------------------------------------
        self.mpc_horizon_ = 4
        self.ipopt_max_iter_ = 50
        self.mpc_model = 'explicit'

        self.mpc_u_lb_ = -0.1
        self.mpc_u_ub_ = 0.1
        obj_pos_lb = np.array([-0.99, -0.99, 0])
        obj_pos_ub = np.array([0.99, 0.99, 0.99])
        self.mpc_q_lb_ = np.hstack((obj_pos_lb, -1e7 * np.ones(4), -1e7 * np.ones(16)))
        self.mpc_q_ub_ = np.hstack((obj_pos_ub, 1e7 * np.ones(4), 1e7 * np.ones(16)))

        self.sol_guess_ = None

    # ---------------------------------------------------------------------------------------------
    #      cost functions for MPC
    # ---------------------------------------------------------------------------------------------
    def init_cost_fns(self):
        x = cs.SX.sym('x', self.n_qpos_)
        u = cs.SX.sym('u', self.n_cmd_)

        obj_pose = x[0:7]
        ff_qpos = x[7:11]
        mf_qpos = x[11:15]
        rf_qpos = x[15:19]
        tm_qpos = x[19:23]

        # forward kinematics to compute the position of fingertip
        ftp_1_position = allegro_fk.fftp_pos_fd_fn(ff_qpos)
        ftp_2_position = allegro_fk.mftp_pos_fd_fn(mf_qpos)
        ftp_3_position = allegro_fk.rftp_pos_fd_fn(rf_qpos)
        ftp_4_position = allegro_fk.thtp_pos_fd_fn(tm_qpos)

        # target cost
        target_position = cs.SX.sym('target_position', 3)
        target_quaternion = cs.SX.sym('target_quaternion', 4)
        position_cost = cs.sumsqr(obj_pose[0:3] - target_position)
        quaternion_cost = 1 - cs.dot(obj_pose[3:7], target_quaternion) ** 2
        # quaternion_cost = cs.sumsqr(obj_pose[3:7] - target_quaternion)
        angular_error = quat_angle(quat_wxyz_to_xyzw(obj_pose[3:7]), quat_wxyz_to_xyzw(target_quaternion))
        contact_cost = (
                cs.sumsqr(obj_pose[0:3] - ftp_1_position) +
                cs.sumsqr(obj_pose[0:3] - ftp_2_position) +
                cs.sumsqr(obj_pose[0:3] - ftp_3_position) +
                cs.sumsqr(obj_pose[0:3] - ftp_4_position)
        )

        # grasp cost
        obj_v0 = ftp_1_position - x[0:3]
        obj_v1 = ftp_2_position - x[0:3]
        obj_v2 = ftp_3_position - x[0:3]
        obj_v3 = ftp_4_position - x[0:3]
        grasp_closure = cs.sumsqr(obj_v0 / cs.norm_2(obj_v0) + obj_v1 / cs.norm_2(obj_v1)
                                  + obj_v2 / cs.norm_2(obj_v2) + obj_v3 / cs.norm_2(obj_v3))

        # control cost
        control_cost = cs.sumsqr(u)

        # robot qpos regulation cost
        robot_qpos0 = cs.SX.sym('robot_qpos0', self.n_robot_qpos_)
        robot_qpos_reg_cost = cs.sumsqr(x[7:] - robot_qpos0)

        # cost params
        phi_vec = cs.SX.sym('phi_vec', self.max_ncon_ * 4)
        jac_mat = cs.SX.sym('jac_mat', self.max_ncon_ * 4, self.n_qvel_)
        cost_param = cs.vvcat([target_position, target_quaternion, phi_vec, jac_mat, robot_qpos0])

        # base cost
        base_cost = 0.0 * contact_cost + 1.0 * quaternion_cost
        final_cost = 2.0 * quaternion_cost

        path_cost_fn = cs.Function('path_cost_fn', [x, u, cost_param], [base_cost + 1.0 * control_cost + 0.0 * robot_qpos_reg_cost])
        final_cost_fn = cs.Function('final_cost_fn', [x, cost_param], [10 * final_cost + 0.0 * robot_qpos_reg_cost])

        # saved param.
        # path_cost = 2.0 * contact_cost + 0.1 * control_cost + 0.25 * robot_qpos_reg_cost
        # final_cost = 10 * 20 * quaternion_cost + 0.05 * robot_qpos_reg_cost

        # cs functions to evaluate
        self.contact_cost_fn = cs.Function('contact_cost_fn', [x, cost_param], [contact_cost])
        self.quaternion_cost_fn = cs.Function('quaternion_cost_fn', [x, cost_param], [quaternion_cost])
        self.robot_qpos_reg_cost_fn = cs.Function('robot_qpos_reg_cost_fn', [x, cost_param], [robot_qpos_reg_cost])
        self.control_cost_fn = cs.Function('control_cost_fn', [u], [control_cost])

        self.angle_error_fn = cs.Function('angle_error_fn', [x, cost_param], [angular_error])

        return path_cost_fn, final_cost_fn
