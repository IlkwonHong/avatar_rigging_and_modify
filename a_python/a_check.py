import open3d as o3d
import numpy as np
import pickle as pkl
import math
import copy



# vertex metadata load
with open('/Users/ik/Desktop/zepeto/a_python/to_py.pkl', 'rb') as f:
    [vert_ids, bone_ids, weights, joint_leg] = pkl.load(f)
# vertex coordinate load
with open('/Users/ik/Desktop/zepeto/a_python/to_py_mesh.pkl', 'rb') as f:
    vertices = pkl.load(f)
# to np
vertices = np.asarray(vertices)
bone_ids = np.asarray(bone_ids)
vert_ids = np.asarray(vert_ids)
joint_leg = np.asarray(joint_leg)
weights = np.asarray(weights)

# blender에서 찾은 trasnform inverse 반영
_rot = np.array([[1,0,0],[0,0,-1],[0,1,0]])
_scale = 100
_transform = np.array([0,0.022936*100, 0])
joint_leg_aligned = (_rot.T@joint_leg.T).T * _scale + _transform


# visualize
## body mesh
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(vertices)
_color = np.ones([9067,3])*np.array([0.5,0.3,1])
idx = bone_ids == 154
_color[vert_ids[idx],:] = np.array([1.0,0.1,0.7])
pcd.colors = o3d.utility.Vector3dVector(_color)
## joint and others
coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10)
joint_ = o3d.geometry.TriangleMesh.create_sphere(radius=1)
joint_.translate(joint_leg_aligned[4,:])
o3d.visualization.draw_geometries([pcd,joint_,coord])





# left leg select
idx_leg_l = [149, 150, 151, 152, 153]
_v = vertices.shape[0]
idx_all = np.arange(_v)
to_select = np.zeros([_v, 5])
for i, _idx in enumerate(idx_leg_l):
    idx = bone_ids == _idx
    to_select[vert_ids[idx], i] = 1
to_select = to_select.sum(axis=1) >0

# _color = np.ones([9067,3])*np.array([0.5,0.3,1])
# _color[idx_all[to_select]] = np.array([1.0,0.1,0.7])
# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(vertices)
# pcd.colors = o3d.utility.Vector3dVector(_color)
# o3d.visualization.draw_geometries([pcd])

leg_all = idx_all[to_select]

idx_leg_l = [150, 151, 152, 153]
_v = vertices.shape[0]
idx_all = np.arange(_v)
to_select = np.zeros([_v, 5])
for i, _idx in enumerate(idx_leg_l):
    idx = bone_ids == _idx
    to_select[vert_ids[idx], i] = 1
to_select = to_select.sum(axis=1) >0

leg_part = idx_all[to_select]

_color = np.ones([9067,3])*np.array([0.5,0.3,1])
_color[idx_all[to_select]] = np.array([1.0,0.1,0.7])
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(vertices)
pcd.colors = o3d.utility.Vector3dVector(_color)
joint_ = o3d.geometry.TriangleMesh.create_sphere(radius=3)
joint_.translate(joint_leg_aligned[1,:])
o3d.visualization.draw_geometries([pcd, joint_])


idx = bone_ids == 149
tmp_vert = vert_ids[idx]
tmp_weight = weights[idx]


# scaling
_scale = 0.5
vertices_modi = copy.deepcopy(vertices)
vertices_modi -= joint_leg_aligned[1,:]
vertices_modi[tmp_vert] += vertices_modi[tmp_vert] * tmp_weight[:,None] * _scale
_leg_part = np.setdiff1d(leg_part, tmp_vert)
vertices_modi[_leg_part] += vertices_modi[_leg_part] * _scale

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(vertices_modi)
o3d.visualization.draw_geometries([pcd])






# def rotx(alpha):
#     rot = np.array([
#         [1, 0, 0],
#         [0, math.cos(alpha), -math.sin(alpha)],
#         [0, math.sin(alpha), math.cos(alpha)]
#     ])
#     return rot

'thigh.L', 'sheen.L', 'foot.L', 'toe.L', 'heel.02.L'











# left leg select
idx_leg_l = [149]
_v = vertices.shape[0]
idx_all = np.arange(_v)
to_select = np.zeros([_v, 5])
for i, _idx in enumerate(idx_leg_l):
    idx = bone_ids == _idx
    to_select[vert_ids[idx], i] = 1
to_select = to_select.sum(axis=1) >0
leg_all = idx_all[to_select]

idx_leg_l = [150]
_v = vertices.shape[0]
idx_all = np.arange(_v)
to_select = np.zeros([_v, 5])
for i, _idx in enumerate(idx_leg_l):
    idx = bone_ids == _idx
    to_select[vert_ids[idx], i] = 1
to_select = to_select.sum(axis=1) >0

leg_part = idx_all[to_select]


idx_leg_l = [151]
_v = vertices.shape[0]
idx_all = np.arange(_v)
to_select = np.zeros([_v, 5])
for i, _idx in enumerate(idx_leg_l):
    idx = bone_ids == _idx
    to_select[vert_ids[idx], i] = 1
to_select = to_select.sum(axis=1) >0

leg_shin = idx_all[to_select]

leg_all_ = np.setdiff1d(leg_all, leg_part)
_both = np.setdiff1d(leg_all, leg_all_)


only = np.setdiff1d(leg_part, idx_leg_l)
only = np.setdiff1d(only, leg_shin)
idx_leg_l

