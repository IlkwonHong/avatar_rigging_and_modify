import math
import pickle as pkl
from shutil import move
import numpy as np
from qtpy import _warn_old_minor_version
from a_python.rigging_class.bone_names import *
import copy
import open3d as o3d
from a_python.utils.rot import _rot_base


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
with open('./data/maya_weight.pkl', 'rb') as f:
    [joint_name, joint_weight, joint_group_verts] = pkl.load(f)
joint_mat = np.asarray(joint_mat)
joint_mat = np.reshape(joint_mat, [-1,4,4]).transpose([0,2,1])
joint_name = np.asarray(joint_name)




# to np
vertices = np.asarray(vertices)
bone_ids = np.asarray(bone_ids)
vert_ids = np.asarray(vert_ids)
weights = np.asarray(weights)
bone_head_tail = np.asarray(bone_head_tail)
_vertices = copy.deepcopy(vertices)



weight_sum = np.zeros([vertices.shape[0]])
for _idx, _weight in zip(joint_group_verts, joint_weight):
     weight_sum[_idx] += _weight


vec1 = joint_mat[1,:3,3] - joint_mat[2,:3,3]
vec2 = joint_mat[1,0,:3]
vec2 = joint_mat[1,:3,0]
np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    # [joint_name, joint_weight, joint_group_verts] = pkl.load(f)

# 본별로 인덱스만 가져와서 변위 스케일만 갖다주자! self.vertices[self.idx_vertices_for_bone]* self.weight_vertices_for_bone[:,None]* (v_trans)
class rig_base():
    def __init__(self, idx):
        self.idx = idx
        self.idx_child = idx_hier_maya[idx]
        self.name = bone_idx_maya[idx]
        
        # to be updated while rigging
        self.T_mat = joint_mat[idx]
        self.head = self.T_mat[:3,3]
        self.dir_longi = self.T_mat[:3, 0]/np.linalg.norm(self.T_mat[:3, 0])
        self.dir_y = self.T_mat[:3,1]/np.linalg.norm(self.T_mat[:3, 1])
        self.dir_z = self.T_mat[:3, 2]/np.linalg.norm(self.T_mat[:3, 2])

        self.idx_vertices_for_bone = np.asarray(joint_group_verts[idx])
        # weight unnormalized
        self.weight_vertices_for_bone = np.asarray(joint_weight[idx]) 
        # # weight normalized
        # self.weight_vertices_for_bone = joint_weight[idx] /  weight_sum[self.idx_vertices_for_bone]
        
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
        self.dir_longi = self.T_mat[:3, 0]/np.linalg.norm(self.T_mat[:3, 0])
        self.dir_y = self.T_mat[:3,1]/np.linalg.norm(self.T_mat[:3, 1])
        self.dir_z = self.T_mat[:3, 2]/np.linalg.norm(self.T_mat[:3, 2])

    def concat_weights(self, weight_all):
        weight_all[self.idx_vertices_for_bone] += self.weight_vertices_for_bone
        if self.childs is not None:
            for child in self.childs:
                child.concat_weights(weight_all)

    # TODO start: maximum weight가 1이 아닌경우에 대비해서 maximum scale 1로 맞춰서 scale 조정
    def scale_y(self, scale):
        verts_projected_vec = ((_vertices[self.idx_vertices_for_bone] - self.head) @ self.dir_y[:,None]) * self.dir_y
        to_move = verts_projected_vec * (scale - 1)
        _vertices[self.idx_vertices_for_bone] += self.weight_vertices_for_bone[:,None] * to_move

    def scale_z(self, scale):
        verts_projected_vec = ((_vertices[self.idx_vertices_for_bone] - self.head) @ self.dir_z[:,None]) * self.dir_z
        to_move = verts_projected_vec * (scale - 1)
        _vertices[self.idx_vertices_for_bone] += self.weight_vertices_for_bone[:,None] * to_move

    # 관절 로컬좌표계 기준이라서 그닥..
    def scale_y_hier(self, scale):
        _weight_sum = np.zeros([vertices.shape[0]])
        self.concat_weights(_weight_sum)
        verts_projected_vec = ((_vertices[:,:] - self.head) @ self.dir_y[:,None]) * self.dir_y
        to_move = verts_projected_vec * (scale - 1)
        _vertices[:,:] += _weight_sum[:,None] * to_move

    def scale_z_hier(self, scale):
        _weight_sum = np.zeros([vertices.shape[0]])
        self.concat_weights(_weight_sum)
        verts_projected_vec = ((_vertices[:,:] - self.head) @ self.dir_z[:,None]) * self.dir_z
        to_move = verts_projected_vec * (scale - 1)
        _vertices[:,:] += _weight_sum[:,None] * to_move
    # TODO end

    def trans_head(self, trans, coord='world'):
        if coord == 'world':
            _vertices[self.idx_vertices_for_bone] += self.weight_vertices_for_bone[:,None] * trans
            _T_global = np.eye(4)
            _T_global[:3,3] += trans
            self.update_local_coord(_T_global)
        # elif coord == 'local':

        if self.childs is not None:
            for child in self.childs:
                child.trans_head(trans, coord)
    
    def rot_update(self, mat_rot, center):
        _T_global1 = np.eye(4)
        _T_global1[:3,3] -= center

        _T_global2 = np.eye(4)
        _T_global2[:3,:3] = mat_rot

        _T_global3 = np.eye(4)
        _T_global3[:3,3] += center

        _T_global = _T_global3 @ _T_global2 @ _T_global1
        self.update_local_coord(_T_global)

        if self.childs is not None:
            for child in self.childs:
                child.rot_update(mat_rot, center)


    def rot(self, mat_rot):
        _weight_sum = np.zeros([vertices.shape[0]])
        self.concat_weights(_weight_sum)
        _vertices[:,:] -= self.head
        _vertices[:,:] += _weight_sum[:,None] * ((mat_rot @ _vertices.T).T - _vertices )
        _vertices[:,:] += self.head

        center = self.head
        # _T_global1 = np.eye(4)
        # _T_global1[:3,3] -= self.head

        # _T_global2 = np.eye(4)
        # _T_global2[:3,:3] = mat_rot

        # _T_global3 = np.eye(4)
        # _T_global3[:3,3] += self.head

        # _T_global = _T_global3 @ _T_global2 @ _T_global1
        self.rot_update(mat_rot, center)
        # if self.childs is not None:
        #     for child in self.childs:
        #         child.update_local_coord(_T_global)

        # self.rot_update(mat_rot)

    
    # def rot_rig_update(self, mat_rot, center):
    #     self.head = (mat_rot @ (self.head - center).T).T + center
    #     trans_tail = (mat_rot @ (self.tail - center).T).T + center - self.tail
    #     self.tail += trans_tail
    #     if self.childs is not None:
    #         for child in self.childs:
    #             child.rot_rig_update(mat_rot, center)

    def update(self):
        '''전부 head기준으로 움직임'''
        self.translate_head()

        _vertices[self.idx_vertices_for_bone] -= self.head
        _vertices[self.idx_vertices_for_bone] += _vertices[self.idx_vertices_for_bone]*self.disp_to_scale
        _vertices[self.idx_vertices_for_bone] += self.head

        if self.childs is not None:
            for child in self.childs:       
                child.update()

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

    # def o3d_color(self):
        


_vertices = copy.deepcopy(vertices)
# 디버깅 visualize용
tmp = rig_class(6)
# tmp = rig_class(4)
# tmp = rig_class(3)
tmp.rot(_rot_base('x', np.pi/3))
tmp.childs[0].childs[0].rot(_rot_base('x', np.pi/3))

# tmp.trans_head(np.array([-10,0,0]))
# tmp.scale_y_hier(2)
# tmp.scale_z_hier(2)
_list_joint = []
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(_vertices)
_color = np.ones([9067,3])*np.array([0.5,0.3,1])
pcd.colors = o3d.utility.Vector3dVector(_color)
_list_joint.append(pcd)


tmp.draw_joint(_list_joint)
o3d.visualization.draw_geometries(_list_joint)










_joint = []
coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=30)
_joint.append(coord)

for trans, ori, mat in zip(joint_pos, joint_ori, joint_mat):
    # trans = np.array([5.0813, 7.05, -2.182])
    joint_head = o3d.geometry.TriangleMesh.create_sphere(radius=1)
    # joint_head.translate(trans)
    joint_head.translate(mat[:3,3])

    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10)
    # _coord_R = _rot_base('z', 90/180*ori[2]) @ _rot_base('y', 90/180*ori[1]) @ _rot_base('x', 90/180*ori[0])
    # coord.rotate(_coord_R)
    coord.rotate(mat[:3,:3])
    coord.translate(trans)

    _joint.append(joint_head)
    _joint.append(coord)

_joint.append(pcd)
o3d.visualization.draw_geometries(_joint)







o3d.visualization.draw_geometries([pcd, coord, joint_head])



joint_ori




_rot_base('x', 90/180*np.pi)


coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10, R = _rot_base('x', 90/180*np.pi))
