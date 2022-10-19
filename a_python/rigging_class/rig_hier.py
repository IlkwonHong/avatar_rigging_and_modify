import pickle as pkl
from shutil import move
import numpy as np
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


# to np
vertices = np.asarray(vertices)
bone_ids = np.asarray(bone_ids)
vert_ids = np.asarray(vert_ids)
joint_leg = np.asarray(joint_leg)
weights = np.asarray(weights)
bone_head_tail = np.asarray(bone_head_tail)
# blender에서 찾은 trasnform inverse 반영
_rot = np.array([[1,0,0],[0,0,-1],[0,1,0]])
_scale = 100
_transform = np.array([0,0.022936*100, 0])
# rig와 mesh 스케일 차이
joint_leg_aligned = (_rot.T@joint_leg.T).T * _scale + _transform
bone_head_tail[:,:3] = (_rot.T@bone_head_tail[:,:3].T).T * _scale + _transform
bone_head_tail[:,3:6] = (_rot.T@bone_head_tail[:,3:6].T).T * _scale + _transform

idx_vertices_for_bone = []
weight_vertices_for_bone = []
for _i in range(len(bone_idx)):
    idx_vertices_for_bone.append(vert_ids[bone_ids == _i])
    weight_vertices_for_bone.append(weights[bone_ids == _i])

_vertices = copy.deepcopy(vertices)



weight_sum = np.zeros([vertices.shape[0]])
for _idx, _weight in zip(idx_vertices_for_bone,weight_vertices_for_bone):
     weight_sum[_idx] += _weight


# 본별로 인덱스만 가져와서 변위 스케일만 갖다주자! self.vertices[self.idx_vertices_for_bone]* self.weight_vertices_for_bone[:,None]* (v_trans)
class rig_base():
    def __init__(self, idx):
        self.head = bone_head_tail[idx][:3]
        self.tail = bone_head_tail[idx][3:6]
        self.idx = idx
        self.len = np.linalg.norm(self.head - self.tail)
        self.idx_child = idx_hier[idx]
        self.name = bone_idx[idx]

        self.idx_vertices_for_bone = idx_vertices_for_bone[idx]

        # weight unnormalized
        # self.weight_vertices_for_bone = weight_vertices_for_bone[idx] 
        # # weight normalized
        self.weight_vertices_for_bone = weight_vertices_for_bone[idx] /  weight_sum[self.idx_vertices_for_bone]
        
        self.n_verts = self.idx_vertices_for_bone.shape[0]

        # disp = vertices * disp_to_scale
        self.disp_to_scale = np.zeros([self.n_verts,3])
        self.trans_head = np.zeros(3)
        self.trans_tail = np.zeros(3)
        

class rig_class(rig_base):
    def __init__(self, idx):
        super().__init__(idx)
        if self.idx_child is not None:
            self.childs = []
            for idx in self.idx_child:
                self.childs.append(rig_class(idx))
        else :
            self.childs = None

    def concat_weights(self, weight_all):
        weight_all[self.idx_vertices_for_bone] += self.weight_vertices_for_bone
        if self.childs is not None:
            for child in self.childs:
                child.concat_weights(weight_all)

    def scale_all(self, scale):
        self.trans_tail = (self.tail - self.head) * scale + self.head - self.tail
        self.tail += self.trans_tail
        self.disp_to_scale = np.repeat(self.weight_vertices_for_bone[:,None]* (scale - 1), 3, axis=1)
        
        if self.childs is not None:
            for child in self.childs:       
                child.trans_head += self.trans_tail


    def translate_head(self):
        trans = copy.deepcopy(self.trans_head)
        self.trans_head *= 0
        _vertices[self.idx_vertices_for_bone] += self.weight_vertices_for_bone[:,None] @ trans[None,:]
        self.head += trans
        self.tail += trans

        if self.childs is not None:
            for child in self.childs:       
                child.trans_head += trans
                child.translate_head()

    def scale_longi_hier(self, scale):
        # dir_bone = (self.tail - self.head)/np.linalg.norm(self.tail - self.head)
        # verts_projected = (_vertices[:,:] - self.head )@ dir_bone[:,None] 
        # move_along_vec = (verts_projected * (scale - 1)) * dir_bone
        # _weight_sum = np.zeros([vertices.shape[0]])
        # self.concat_weights(_weight_sum)
        # _vertices[:,:] += _weight_sum[:,None] * move_along_vec


        dir_bone = (self.tail - self.head)/np.linalg.norm(self.tail - self.head)
        verts_projected = (_vertices[self.idx_vertices_for_bone] - self.head) @ dir_bone[:,None] 
        move_along_vec = (verts_projected * (scale - 1)) * dir_bone

        _vertices[self.idx_vertices_for_bone] += self.weight_vertices_for_bone[:,None] * move_along_vec
        self.tail += (self.tail - self.head) * (scale - 1) 

        if self.childs is not None:
            for child in self.childs:       
                child.trans_head += (self.tail - self.head) * (scale - 1)
                child.translate_head()



    
    def rot_rig_update(self, mat_rot, center):
        self.head = (mat_rot @ (self.head - center).T).T + center
        trans_tail = (mat_rot @ (self.tail - center).T).T + center - self.tail
        self.tail += trans_tail
        if self.childs is not None:
            for child in self.childs:
                child.rot_rig_update(mat_rot, center)

    # TODO : 관절별 로테이션으로 구현, local 좌표계 구현
    def rot(self, mat_rot, center):
        _weight_sum = np.zeros([vertices.shape[0]])
        self.concat_weights(_weight_sum)
        _vertices[:,:] -= center
        _vertices[:,:] += _weight_sum[:,None] * ((mat_rot @ _vertices.T).T - _vertices )
        _vertices[:,:] += center

        self.rot_rig_update(mat_rot, center)

    def update(self):
        '''전부 head기준으로 움직임'''
        self.translate_head()

        _vertices[self.idx_vertices_for_bone] -= self.head
        _vertices[self.idx_vertices_for_bone] += _vertices[self.idx_vertices_for_bone]*self.disp_to_scale
        _vertices[self.idx_vertices_for_bone] += self.head

        if self.childs is not None:
            for child in self.childs:       
                child.update()
    
    # def o3d_color(self):
        



# 디버깅 visualize용
tmp = rig_class(153)

tmp = rig_class(149)
tmp.scale_longi_hier(1.5)
tmp.childs[0].childs[0].rot(_rot_base('x', np.pi/3), tmp.childs[0].childs[0].head)
tmp.childs[0].rot(_rot_base('x', np.pi/2), tmp.childs[0].head)
tmp.rot(_rot_base('z', np.pi/5), tmp.head)
tmp.rot(_rot_base('x', -np.pi/3), tmp.head)


colors = np.ones([weight_sum.shape[0],3]) * 0.5
colors[:,0] = weight_sum
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(_vertices)
pcd.colors = o3d.utility.Vector3dVector(colors)
coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10)
joint_head = o3d.geometry.TriangleMesh.create_sphere(radius=1)
joint_head.translate(tmp.head)
joint_tail = o3d.geometry.TriangleMesh.create_sphere(radius=1)
joint_tail.translate(tmp.tail)

o3d.visualization.draw_geometries([pcd, coord, joint_head, joint_tail])









