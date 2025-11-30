import mujoco

import numpy as np

np.set_printoptions(suppress=True)

from envs.allegro_env import MjSimulator

from utils import rotations


class Contact:
    def __init__(self, param):
        self.param_ = param

    def detect_once(self, simulator: MjSimulator, ret_extras=False):
        mujoco.mj_forward(simulator.model_, simulator.data_)
        mujoco.mj_collision(simulator.model_, simulator.data_)

        # extract the contacts
        n_con = simulator.data_.ncon
        contacts = simulator.data_.contact

        # solve the contact Jacobian
        con_phi_list = []
        con_frame_list = []
        con_pos_list = []
        con_jac_list = []
        con_frame_pmd_list = []

        for i in range(n_con):
            contact_i = contacts[i]

            geom1_name = mujoco.mj_id2name(simulator.model_, mujoco.mjtObj.mjOBJ_GEOM, contact_i.geom1)
            body1_id = simulator.model_.geom_bodyid[contact_i.geom1]
            geom2_name = mujoco.mj_id2name(simulator.model_, mujoco.mjtObj.mjOBJ_GEOM, contact_i.geom2)
            body2_id = simulator.model_.geom_bodyid[contact_i.geom2]

            # print("Contact {}: between {} and {}".format(i, geom1_name, geom2_name))

            # con_jacp_n = None

            # contact between balls and object
            if (geom1_name in self.param_.object_names_):
                breakpoint()
                # contact point
                con_pos = contact_i.pos
                con_dist = contact_i.dist * 0.5
                con_mu = self.param_.mu_object_

                con_frame = -contact_i.frame.reshape((-1, 3)).T
                con_frame_pmd = np.hstack((con_frame, -con_frame[:, -2:]))

                jacp1 = np.zeros((3, self.param_.n_mj_v_))
                if simulator.model_.nv == self.param_.n_mj_v_ - 3:
                    jacp1_wo_obj_pos = np.zeros((3, simulator.model_.nv))
                    mujoco.mj_jac(simulator.model_, simulator.data_, jacp=jacp1_wo_obj_pos, jacr=None, point=con_pos, body=body1_id)
                    jacp1[:, :-6] = jacp1_wo_obj_pos[:, :-3]
                    jacp1[:, -3:] = jacp1_wo_obj_pos[:, -3:]
                else:
                    mujoco.mj_jac(simulator.model_, simulator.data_, jacp=jacp1, jacr=None, point=con_pos, body=body1_id)
                con_jacp1 = con_frame_pmd.T @ jacp1

                jacp2 = np.zeros((3, self.param_.n_mj_v_))
                if simulator.model_.nv == self.param_.n_mj_v_ - 3:
                    jacp2_wo_obj_pos = np.zeros((3, simulator.model_.nv))
                    mujoco.mj_jac(simulator.model_, simulator.data_, jacp=jacp2_wo_obj_pos, jacr=None, point=con_pos, body=body2_id)
                    jacp2[:, :-6] = jacp2_wo_obj_pos[:, :-3]
                    jacp2[:, -3:] = jacp2_wo_obj_pos[:, -3:]
                else:
                    mujoco.mj_jac(simulator.model_, simulator.data_, jacp=jacp2, jacr=None, point=con_pos, body=body2_id)
                con_jacp2 = con_frame_pmd.T @ jacp2

                # hint:
                # jacobian direction: from contact pair to obj
                # obj - contact pair
                con_jacp = -(con_jacp2 - con_jacp1)
                con_jacp_n = con_jacp[0]
                con_jacp_f = con_jacp[1:]
                con_jac = con_jacp_n + con_mu * con_jacp_f

                con_pos_list.append(con_pos - con_dist * con_frame[:, 0])
                con_phi_list.append(con_dist)
                con_frame_list.append(con_frame)
                con_jac_list.append(con_jac)
                con_frame_pmd_list.append(con_frame_pmd)

            elif (geom2_name in self.param_.object_names_):
                # contact point
                con_pos = contact_i.pos
                con_dist = contact_i.dist * 0.5
                con_mu = self.param_.mu_object_

                con_frame = contact_i.frame.reshape((-1, 3)).T
                con_frame_pmd = np.hstack((con_frame, -con_frame[:, -2:]))

                jacp1 = np.zeros((3, self.param_.n_mj_v_))
                if simulator.model_.nv == self.param_.n_mj_v_ - 3:
                    jacp1_wo_obj_pos = np.zeros((3, simulator.model_.nv))
                    mujoco.mj_jac(simulator.model_, simulator.data_, jacp=jacp1_wo_obj_pos, jacr=None, point=con_pos, body=body1_id)
                    jacp1[:, :-6] = jacp1_wo_obj_pos[:, :-3]
                    jacp1[:, -3:] = jacp1_wo_obj_pos[:, -3:]
                else:
                    mujoco.mj_jac(simulator.model_, simulator.data_, jacp=jacp1, jacr=None, point=con_pos, body=body1_id)
                con_jacp1 = con_frame_pmd.T @ jacp1

                jacp2 = np.zeros((3, self.param_.n_mj_v_))
                if simulator.model_.nv == self.param_.n_mj_v_ - 3:
                    jacp2_wo_obj_pos = np.zeros((3, simulator.model_.nv))
                    mujoco.mj_jac(simulator.model_, simulator.data_, jacp=jacp2_wo_obj_pos, jacr=None, point=con_pos, body=body2_id)
                    jacp2[:, :-6] = jacp2_wo_obj_pos[:, :-3]
                    jacp2[:, -3:] = jacp2_wo_obj_pos[:, -3:]
                else:
                    mujoco.mj_jac(simulator.model_, simulator.data_, jacp=jacp2, jacr=None, point=con_pos, body=body2_id)
                con_jacp2 = con_frame_pmd.T @ jacp2

                # hint:
                # jacobian direction: from contact pair to obj
                # obj - contact pair
                con_jacp = (con_jacp2 - con_jacp1)
                con_jacp_n = con_jacp[0]
                con_jacp_f = con_jacp[1:]
                con_jac = con_jacp_n + con_mu * con_jacp_f

                con_pos_list.append(con_pos + con_dist * con_frame[:, 0])
                con_phi_list.append(con_dist)
                con_frame_list.append(con_frame)
                con_jac_list.append(con_jac)
                con_frame_pmd_list.append(con_frame_pmd)

            # if i == 1:
            #     breakpoint()

        contact_detect_result = self.reformat(
            dict(
                con_pos_list=con_pos_list,
                con_phi_list=con_phi_list,
                con_frame_list=con_frame_list,
                con_jac_list=con_jac_list,
                con_frame_pmd_list=con_frame_pmd_list),
            ret_extras=ret_extras
        )

        if ret_extras:
            phi_vec, jac_mat, frame_pmd, con_pos = contact_detect_result
            return phi_vec, jac_mat, frame_pmd, con_pos
        else:
            phi_vec, jac_mat = contact_detect_result
            return phi_vec, jac_mat

    def reformat(self, contacts=None, ret_extras=False):
        # parse the input
        con_pos_list = contacts['con_pos_list']
        con_jac_list = contacts['con_jac_list']
        con_phi_list = contacts['con_phi_list']
        con_frame_pmd_list = contacts['con_frame_pmd_list']

        # fill the phi_vec
        con_pos_mat = np.zeros((self.param_.max_ncon_, 3))
        phi_vec = np.ones((self.param_.max_ncon_ * 4,))  # this is very,very important for soft sensitivity analysis
        jac_mat = np.zeros((self.param_.max_ncon_ * 4, self.param_.n_qvel_))
        frame_pmd_mat = np.zeros((self.param_.max_ncon_ * 5, 3))
        for i in range(len(con_phi_list)):
            con_pos_mat[i, :] = con_pos_list[i]
            phi_vec[4 * i: 4 * i + 4] = con_phi_list[i]
            jac_mat[4 * i: 4 * i + 4] = con_jac_list[i]
            frame_pmd_mat[5 * i: 5 * i + 5] = con_frame_pmd_list[i].T

        jac_mat_reorder = np.zeros((self.param_.max_ncon_ * 4, self.param_.n_qvel_))
        jac_mat_reorder[:, 0:6] = jac_mat[:, -6:]
        jac_mat_reorder[:, 6:] = jac_mat[:, 0:16]

        if ret_extras:
            return phi_vec, jac_mat_reorder, frame_pmd_mat, con_pos_mat
        else:
            return phi_vec, jac_mat_reorder
