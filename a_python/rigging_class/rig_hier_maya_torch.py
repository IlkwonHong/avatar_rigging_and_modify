import math
from operator import index
import pickle as pkl
from matplotlib.pyplot import sca
import numpy as np
from a_python.rigging_class.bone_names import *
import copy
import open3d as o3d
from a_python.utils.rot import _rot_base
import torch


# vertex metadata load
with open('/Users/ik/Desktop/zepeto/a_python/to_py.pkl', 'rb') as f:
    [vert_ids, bone_ids, weights, joint_leg] = pkl.load(f)
# vertex coordinate load
with open('/Users/ik/Desktop/zepeto/a_python/to_py_mesh.pkl', 'rb') as f:
    vertices = pkl.load(f)
# bone head tail load
with open('/Users/ik/Desktop/zepeto/a_python/_bone_head_tail.pkl', 'rb') as f:
    bone_head_tail = pkl.load(f)

# maya joint
with open('/Users/ik/Desktop/zepeto/a_python/maya_joint_pos.pkl', 'rb') as f:
    [joint_mat, joint_ori, joint_leg, joint_pos] = pkl.load(f)
with open('/Users/ik/Desktop/zepeto/data/maya_weight.pkl', 'rb') as f:
    [joint_name, joint_weight, joint_group_verts] = pkl.load(f)

joint_mat = torch.tensor(joint_mat)    
joint_mat = joint_mat.reshape([-1,4,4]).permute([0,2,1])


# to np
vertices = torch.tensor(vertices)
bone_ids = torch.tensor(bone_ids)
vert_ids = torch.tensor(vert_ids)
weights = torch.tensor(weights)
bone_head_tail = torch.tensor(bone_head_tail)
# _vertices = copy.deepcopy(vertices)

# # check weight sum == 1
# weight_sum = torch.zeros([vertices.shape[0]])
# for _idx, _weight in zip(joint_group_verts, joint_weight):
#      weight_sum[_idx] += torch.tensor(_weight)


# upper key length
upper_body = [64, 63, 18, 17, 16, 65]       # head, neck, chestupper, chest, spine, hip
len_joint_upper = torch.linalg.norm(joint_mat[upper_body[:-1],:3,3] - joint_mat[upper_body[1:],:3,3], axis=1)
len_joint_upper = len_joint_upper[1:] 
len_joint_upper = len_joint_upper / len_joint_upper.sum()



# 본별로 인덱스만 가져와서 변위 스케일만 갖다주자! self.vertices[self.idx_vertices_for_bone]* self.weight_vertices_for_bone[:,None]* (v_trans)
class rig_base():
    def __init__(self, idx):
        self.idx = idx
        self.idx_child = idx_hier_maya[idx]
        self.name = bone_idx_maya[idx]
        
        # to be updated while rigging
        self.T_mat = joint_mat[idx]
        self.head = self.T_mat[:3,3]
        self.dir_longi = self.T_mat[:3, 0]/torch.linalg.norm(self.T_mat[:3, 0])
        self.dir_y = self.T_mat[:3,1]/torch.linalg.norm(self.T_mat[:3, 1])
        self.dir_z = self.T_mat[:3, 2]/torch.linalg.norm(self.T_mat[:3, 2])

        if self.idx < 65:
            self.idx_vertices_for_bone = torch.tensor(joint_group_verts[idx])
            # weight unnormalized
            self.weight_vertices_for_bone = torch.tensor(joint_weight[idx]) 
            # # weight normalized
            # self.weight_vertices_for_bone = joint_weight[idx] /  weight_sum[self.idx_vertices_for_bone]
            self.n_verts = self.idx_vertices_for_bone.shape[0]
        else :
            self.idx_vertices_for_bone = torch.tensor([0])
            self.weight_vertices_for_bone = torch.tensor([0]) 
            self.n_verts = self.idx_vertices_for_bone.shape[0]


class rig_class(rig_base):
    def __init__(self, idx):
        super().__init__(idx)
        if self.idx_child is not None:
            self.childs = []
            for idx in self.idx_child:
                self.childs.append(rig_class(idx))
        else :
            self.childs = None
    
    def update_local_coord(self, _T_global):
        self.T_mat = _T_global @ self.T_mat 
        self.head = self.T_mat[:3,3]
        self.dir_longi = self.T_mat[:3, 0]/torch.linalg.norm(self.T_mat[:3, 0])
        self.dir_y = self.T_mat[:3,1]/torch.linalg.norm(self.T_mat[:3, 1])
        self.dir_z = self.T_mat[:3, 2]/torch.linalg.norm(self.T_mat[:3, 2])

    def concat_weights(self, weight_all):
        weight_all[self.idx_vertices_for_bone] += self.weight_vertices_for_bone
        if self.childs is not None:
            for child in self.childs:
                child.concat_weights(weight_all)

    # scaling functions
    # TODO start: maximum weight가 1이 아닌경우에 대비해서 maximum scale 1로 맞춰서 scale 조정
    def scale_update(self, scale, center):
        _dist_vec = self.head - center
        head_trans = _dist_vec * (scale - 1)
        _T_global = torch.eye(4)
        _T_global[:3,3] += head_trans
        self.update_local_coord(_T_global)
        
        if self.childs is not None:
            for child in self.childs:
                child.scale_update(scale, center)




    def scale_y(self, scale, _vertices):
        verts_projected_vec = ((_vertices[self.idx_vertices_for_bone] - self.head) @ self.dir_y[:,None]) * self.dir_y
        to_move = verts_projected_vec * (scale - 1)
        _vertices[self.idx_vertices_for_bone] += self.weight_vertices_for_bone[:,None] * to_move

    def scale_z(self, scale, _vertices):
        verts_projected_vec = ((_vertices[self.idx_vertices_for_bone] - self.head) @ self.dir_z[:,None]) * self.dir_z
        to_move = verts_projected_vec * (scale - 1)
        _vertices[self.idx_vertices_for_bone] += self.weight_vertices_for_bone[:,None] * to_move

    def scale_w_kpts(self, index, scale, _vertices):
        _weight = self.weight_vertices_for_bone[torch.where(self.idx_vertices_for_bone==index)]
        scale_for_weight = 1/_weight*(scale - 1) + 1
        self.scale_y(scale_for_weight, _vertices)
        self.scale_z(scale_for_weight, _vertices)
        return

    def scale_iso_hier(self, scale, _vertices):
        _weight_sum = torch.zeros([vertices.shape[0]])
        self.concat_weights(_weight_sum)
        to_move = (_vertices[:,:] - self.head) * (scale - 1)
        _vertices[:,:] += _weight_sum[:,None] * to_move
        self.scale_update(scale, self.head)

    # 관절 로컬좌표계 기준이라서 그닥 필요없을듯
    def scale_y_hier(self, scale, _vertices):
        _weight_sum = torch.zeros([vertices.shape[0]])
        self.concat_weights(_weight_sum)
        verts_projected_vec = ((_vertices[:,:] - self.head) @ self.dir_y[:,None]) * self.dir_y
        to_move = verts_projected_vec * (scale - 1)
        _vertices[:,:] += _weight_sum[:,None] * to_move

    def scale_z_hier(self, scale, _vertices):
        _weight_sum = torch.zeros([vertices.shape[0]])
        self.concat_weights(_weight_sum)
        verts_projected_vec = ((_vertices[:,:] - self.head) @ self.dir_z[:,None]) * self.dir_z
        to_move = verts_projected_vec * (scale - 1)
        _vertices[:,:] += _weight_sum[:,None] * to_move
    # TODO end

    def trans_head(self, trans, _vertices, coord='world'):
        if coord == 'world':
            _vertices[self.idx_vertices_for_bone] += self.weight_vertices_for_bone[:,None] * trans
            _T_global = torch.eye(4)
            _T_global[:3,3] += trans
            self.update_local_coord(_T_global)
        # elif coord == 'local':

        if self.childs is not None:
            for child in self.childs:
                child.trans_head(trans, _vertices, coord)
    
    def rot_update(self, mat_rot, center):
        _T_global1 = torch.eye(4)
        _T_global1[:3,3] -= center

        _T_global2 = torch.eye(4)
        _T_global2[:3,:3] = mat_rot

        _T_global3 = torch.eye(4)
        _T_global3[:3,3] += center

        _T_global = _T_global3 @ _T_global2 @ _T_global1
        self.update_local_coord(_T_global)

        if self.childs is not None:
            for child in self.childs:
                child.rot_update(mat_rot, center)


    def rot(self, mat_rot, _vertices, inplace=True):
        _weight_sum = torch.zeros([vertices.shape[0]])
        self.concat_weights(_weight_sum)
        
        if inplace:
            _vertices[:,:] -= self.head
            _vertices[:,:] += _weight_sum[:,None] * ((mat_rot @ _vertices.T).T - _vertices )
            _vertices[:,:] += self.head
        else:
            tmp1 = _vertices[:,:] - self.head
            tmp2 = tmp1 + _weight_sum[:,None] * ((mat_rot @ tmp1.T).T - tmp1 )
            _vertices[:,:] = tmp2 + self.head
            # _vertices[:,:] = _vertices[:,:] - self.head
            # _vertices[:,:] = _vertices[:,:] + _weight_sum[:,None] * ((mat_rot @ _vertices[:,:].T).T - _vertices[:,:] )
            # _vertices[:,:] = _vertices[:,:] + self.head
        # TODO : 생각해보기 이거 안하면 try to backward twice가 뜨는데 왜일까?
        with torch.no_grad():
            self.rot_update(mat_rot, self.head)

    def reset_vertices(self, _vertices):
        _vertices[:,:] = copy.deepcopy(vertices)

    # for debug
    def draw_joint(self, _list_joint):
        joint_head = o3d.geometry.TriangleMesh.create_sphere(radius=1)
        joint_head.translate(self.T_mat[:3,3])
        coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5)
        coord.rotate(self.T_mat[:3,:3])
        coord.translate(self.T_mat[:3,3])
        _list_joint.append(coord)
        _list_joint.append(joint_head)
        if self.childs is not None:
            for child in self.childs:
                child.draw_joint( _list_joint)
    
    def print_name(self):
        print(self.name)

        if self.childs is not None:
            for child in self.childs:
                child.print_name()

    def scaling_joint_len(self, _vertices, keypts, img_seg):
        '''전체 크기 조절해서 밑위 길이 맞춤 -> 골반 너비 맞추고 -> 다리 길이 맞춤'''
        if self.name != 'hips':
            return

        # 밑위 길이 기준
        h, w = img_seg.shape
        x = torch.linspace(0, w-1, w)
        y = torch.linspace(0, h-1, h)
        xx, yy = torch.meshgrid(x,y, indexing='xy')

        mask_seg = img_seg ==2
        y_crotch = yy[mask_seg].max()
        x_crotch = xx[mask_seg][torch.argmax(yy[mask_seg])]
        len_crotch_to_pelv = torch.linalg.norm(torch.mean(keypts[[9,12],:2], axis=0) - torch.tensor([x_crotch, y_crotch]))
        len_crotch_to_pelv_maya = torch.linalg.norm((joint_mat[6,:3,3] + joint_mat[15,:3,3])/2 - vertices[3106,:])

        scale_align_croth_to_pelv = len_crotch_to_pelv / len_crotch_to_pelv_maya

        # 하체 스케일링
        len_spine = torch.linalg.norm(((keypts[2,:] + keypts[5,:]) - (keypts[9,:] + keypts[12,:]))/2)
        len_pelvis = torch.linalg.norm((keypts[9,:] - keypts[12,:]))
        len_leg_upper = (torch.linalg.norm(keypts[9,:] - keypts[10,:]) + torch.linalg.norm(keypts[12,:] - keypts[13,:]))/2
        len_leg_lower = (torch.linalg.norm(keypts[10,:] - keypts[11,:]) + torch.linalg.norm(keypts[13,:] - keypts[14,:]))/2

        len_maya_pelvis = torch.linalg.norm(joint_mat[6,:3,3] - joint_mat[15,:3,3])
        len_maya_leg_upper = torch.linalg.norm(joint_mat[6,:3,3] - joint_mat[4,:3,3])
        len_maya_leg_lower = torch.linalg.norm(joint_mat[4,:3,3] - joint_mat[1,:3,3])

        scale_align_pelvis = len_pelvis / (scale_align_croth_to_pelv * len_maya_pelvis)

        scale_align_leg_upper = len_leg_upper / \
            (scale_align_pelvis * scale_align_croth_to_pelv * len_maya_leg_upper)
        scale_align_leg_lower = len_leg_lower /\
            (scale_align_pelvis * scale_align_croth_to_pelv * scale_align_leg_upper * len_maya_leg_lower)

        # 밑위 맞추기 위해서 전체 스케일링
        self.scale_iso_hier(scale_align_croth_to_pelv, _vertices)
        # 골반 너비 맞춤
        self.childs[1].scale_iso_hier(scale_align_pelvis, _vertices)

        # TODO_start : 상체 정리.... 일단은 hipscale이 1이 되로록 했음 이건 나중에 상체 굴곡 보고 최적화로 바꿔야할듯
        #  잘 이해가 안되네....
        self.childs[2].scale_iso_hier(scale_align_pelvis, _vertices)
        # TODOend

        self.childs[1].childs[0].scale_iso_hier(scale_align_leg_upper, _vertices)
        self.childs[1].childs[0].childs[0].childs[0].scale_iso_hier(scale_align_leg_lower, _vertices)
        self.childs[1].childs[1].scale_iso_hier(scale_align_leg_upper, _vertices)
        self.childs[1].childs[1].childs[0].childs[0].scale_iso_hier(scale_align_leg_lower, _vertices)

        print("scales:",scale_align_croth_to_pelv, "," ,scale_align_pelvis,"," ,scale_align_leg_upper, "," ,scale_align_leg_lower)
        return torch.tensor([x_crotch, y_crotch])
            
    
    def len_leg_seg(self, _vertices):
        len_thigh_crotch = torch.linalg.norm(_vertices[3127,:2] - _vertices[3440,:2])
        len_knee = torch.linalg.norm(_vertices[8115,:2] - _vertices[8124,:2])
        len_ankle = torch.linalg.norm(_vertices[7866,:2] - _vertices[7878,:2])
        return len_thigh_crotch, len_knee, len_ankle
    

    # def len_leg_upper(self, _vertices):








# _list_joint = []
# # 디버깅 visualize용
# tmp = rig_class(6)
# tmp = rig_class(20)
# # tmp.draw_joint(_list_joint)

# tmp.scale_iso_hier(1.5)


# tmp.reset_vertices()
# _list_joint = []
# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(_vertices)
# _color = torch.ones([9067,3])*torch.tensor([0.5,0.3,1])
# pcd.colors = o3d.utility.Vector3dVector(_color)

# # _pcd = o3d.geometry.PointCloud()
# # _pcd.points = o3d.utility.Vector3dVector(vertices)
# # _list_joint.append(_pcd)

# _list_joint.append(pcd)


# tmp.draw_joint(_list_joint)
# o3d.visualization.draw_geometries(_list_joint)


















