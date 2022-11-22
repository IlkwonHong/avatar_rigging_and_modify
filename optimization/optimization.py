from a_python.rigging_class.rig_hier_maya_torch import *
import json
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import pickle as pkl
import scipy
import torch
import math
import time
import pytorch3d
from pytorch3d.io import load_objs_as_meshes, save_obj
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVOrthographicCameras,
    OpenGLOrthographicCameras,
    FoVPerspectiveCameras, 
    PointLights, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    SoftSilhouetteShader,
    SoftPhongShader
)

def img_upperbody_maya_joints(keypts):
    '''maya관절 길이비율대로 상체 조인트 추가 : chest upper, chest, spine'''
    img_hip = (keypts[9,:2] + keypts[12,:2])/2
    img_neck = (keypts[2,:2] + keypts[5,:2])/2
    img_spine = (img_neck - img_hip) * len_joint_upper[-1] + img_hip
    img_chest = (img_neck - img_hip) * (len_joint_upper[-1] + len_joint_upper[-2]) + img_hip
    img_chsetupper = (img_neck - img_hip) * (len_joint_upper[-1] + len_joint_upper[-2] + len_joint_upper[-3])  + img_hip
    return img_hip, img_neck, img_spine, img_chest, img_chsetupper

def fitting_based_on_upperbody_joints(body, img_spine, img_hip, _vertices):
    '''img_upperbody_maya_joints 함수 결과 사용해서 메쉬 스케일링으로 상체 조인트위치 맞춤'''
    # hip to spine 거리
    _len_spine_maya = np.linalg.norm(body.head - body.childs[0].head) 
    _len_spine_img = np.linalg.norm(img_spine - img_hip)

    # 상체 조인트 맞춤
    _spine_trans = -(_len_spine_maya - _len_spine_img) * \
        ((-body.head + body.childs[0].head) / np.linalg.norm(body.head - body.childs[0].head))
    body.childs[0].trans_head(_spine_trans, _vertices)
    _scale_spine_hier = (np.linalg.norm(img_hip - img_neck) * len_joint_upper[-1]) / \
        (np.linalg.norm(body.childs[0].head - body.childs[0].childs[0].head))
    body.childs[0].scale_iso_hier(_scale_spine_hier, _vertices)


idx_img = 8
folder = idx_img
path_img = "/Users/ik/Downloads/test_set_new/{}/test_resized.png".format(folder)
path_json = "/Users/ik/Downloads/test_set_new/{}/test_resized_keypoints.json".format(folder)
path_dp = "/Users/ik/Downloads/test_set_new/{}/dp_dump.pkl".format(folder)

with open(path_json, 'r') as f:
    data = json.load(f)
with open(path_dp, 'rb') as f:
    [img_seg, img_v, img_u,_] = pkl.load(f)
h,w = img_seg.shape
keypts = torch.tensor(data['people'][0]['pose_keypoints_2d']).reshape([-1,3])
img = cv.imread(path_img)

body = rig_class(65)


h, w = img_seg.shape
x = np.linspace(0, w-1, w)
y = np.linspace(0, h-1, h)
xx, yy = np.meshgrid(x,y, indexing='xy')

# h, w = img_seg.shape
x = torch.linspace(0, w-1, w)
y = torch.linspace(0, h-1, h)
# xx, yy = torch.meshgrid(x,y, indexing='xy')



# plt.imshow(img_seg)
_vertices = copy.deepcopy(vertices)
crotch = body.scaling_joint_len(_vertices, keypts, img_seg)


# 다리 optimization 용 마스킹
mask_body = img_seg==2
mask_body_ = mask_body * (xx >= np.asarray(crotch[0]))
mask_leg_l = img_seg==10
min(yy[mask_leg_l])
mask_body_ = mask_body_ * (yy > min(yy[mask_leg_l]))
mask_leg_l_for_optimization  = mask_leg_l + mask_body_

opt_x = torch.nn.Parameter(torch.tensor([0.,0.,1.,1.,1.,1.]))

img_grid = torch.from_numpy(mask_leg_l_for_optimization.astype(np.double))

optimizer = torch.optim.SGD([{'params' : opt_x}], lr=0.001, momentum=0.9)
optimizer = torch.optim.Adam([{'params' : opt_x}], lr=0.001)

# 최적화 함수 정의
def gaussian_2d(x, y, mx, my, sx, sy):
    return (2*math.pi*sx*sy) * torch.exp(-(
        (x[:,:,None] - mx[None, None, :])**2 / (2*sx**2) + (y[:,:,None] - my[None, None, :])**2 / (2*sy**2)
        ))

def gaussian_2d(x=0, y=0, mx=0, my=0, sx=1, sy=1):
    return 1 / (2*math.pi*sx*sy) * torch.exp(-((x - mx)**2 / (2*sx**2) + (y - my)**2 / (2*sy**2)))

def loss_opt_leg_l(opt_x, body, _vertices): # (opt_x, body, _vertices, xx, yy, w, h, mask):
    _vertices_opt = copy.deepcopy(_vertices)
    '''opt_x : uppperlegtrans_x, uppperlegtrans_y, upperlegscale, uppertwistscale, scale_iso_upper, scale_iso_upper_twist'''
    body.childs[1].childs[0].trans_head(torch.tensor([opt_x[0],opt_x[1],0]), _vertices_opt)
    body.childs[1].childs[0].scale_y(opt_x[2], _vertices_opt)
    body.childs[1].childs[0].scale_z(opt_x[2], _vertices_opt)
    body.childs[1].childs[0].childs[0].scale_y(opt_x[3], _vertices_opt)
    body.childs[1].childs[0].childs[0].scale_z(opt_x[3], _vertices_opt)
    
    _idx_proj = body.childs[1].childs[0].idx_vertices_for_bone
    verts_leg = (_vertices_opt[_idx_proj,:] - _vertices_opt[3106,:]).type(torch.double)
    verts_leg = verts_leg[:,:2]*torch.tensor([1,-1]) + crotch
    # normalize for grid_sample
    verts_leg_normalized = (verts_leg - torch.tensor([w-1,h-1])/2)/(torch.tensor([w-1,h-1])/2)
    verts_leg_normalized = verts_leg_normalized[None, None, :,:]
    output = torch.nn.functional.grid_sample(img_grid[None,None], verts_leg_normalized, align_corners=True) - 1
    
    # _mask = gaussian_2d(xx, yy, verts_leg[:,0], verts_leg[:,1], w/50, h/50)
    # _mask_pts = _mask > 1000


    loss = torch.linalg.norm(output).sum()
    return loss, _vertices_opt


# optimize process
loss_all = []
result_opt_x = []
n_iter = 2000
for i in range(n_iter):
    optimizer.zero_grad()
    loss, _vertices_opt = loss_opt_leg_l(opt_x, body, _vertices)
    # print('loss:', loss)
    loss_all.append(loss.detach())
    # print('opt_x:', opt_x)
    _result_opt_x = copy.deepcopy(opt_x)
    result_opt_x.append(np.asarray(_result_opt_x.detach()))
    loss.backward()
    optimizer.step()

result_opt_x = np.asarray(result_opt_x)
_xx = np.linspace(0,n_iter-1,n_iter)
plt.figure('loss')
plt.scatter(_xx, np.asarray(loss_all))
plt.figure('parameter2')
plt.scatter(_xx, result_opt_x[:,2])
plt.figure('parameter3')
plt.scatter(_xx, result_opt_x[:,3])



# 버텍스 프로젝션 # TODO: 현재는 orthogonal로 하는데 나중에 perspective로 바꾸려나?
_idx_proj = body.childs[1].childs[0].idx_vertices_for_bone
tmp = (_vertices_opt[_idx_proj,:] - _vertices_opt[3106,:])
tmp = tmp[:,:2]*torch.tensor([1,-1]) + crotch
tmp = np.asarray(tmp.detach())
tmp_origin = (_vertices[_idx_proj,:] - _vertices[3106,:])
tmp_origin = tmp_origin[:,:2]*torch.tensor([1,-1]) + crotch
tmp_origin = np.asarray(tmp_origin.detach())
plt.figure("img_idx : {}".format(idx_img))
plt.imshow(mask_leg_l_for_optimization)
plt.scatter(tmp[:,0], tmp[:,1],color='r')

plt.figure("img_idx : {} compare with origin".format(idx_img))
plt.imshow(mask_leg_l_for_optimization)
plt.scatter(tmp[:,0], tmp[:,1],color='r')
plt.scatter(tmp_origin[:,0], tmp_origin[:,1])

########################################################################
# Setup
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")


mesh = load_objs_as_meshes(['/Users/ik/Desktop/zepeto/blender/wip_find_weight.obj'], device)
mesh._verts_list[0] = _vertices - torch.tensor([crotch[0], crotch[1], 0])
# mesh._verts_list[0] *= 2 
# mesh._verts_padded *= 2 


camera = FoVOrthographicCameras(device=device, zfar=200, \
    T = torch.tensor([[0,0,10]]), znear=-100) 
lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])

sigma = 1e-5
raster_settings_soft = RasterizationSettings(
    image_size=512, 
    blur_radius=np.log(1. / sigma - 1.)*sigma, 
    faces_per_pixel=50, 
)

renderer_silhouette = MeshRenderer(rasterizer=MeshRasterizer(
    cameras=camera, raster_settings=raster_settings_soft),
    shader=SoftSilhouetteShader())

shader=SoftSilhouetteShader()
import time
pre =time.time()
_img_ = renderer_silhouette(mesh, cameras=camera, lights=lights)
print(time.time() - pre)
plt.imshow(np.asarray(_img_[0,:,:,3]))



fragments = renderer_silhouette.rasterizer(mesh)




# 8 , 7 , 3, 2



# _vertices = copy.deepcopy(vertices)
# body = rig_class(65)
# body.childs[1].childs[0].trans_head(torch.tensor([10,0,0]), _vertices)


# Visualizatioin
_list_joint = []
pcd = o3d.geometry.PointCloud()

_vertices_draw = np.asarray(copy.deepcopy(_vertices_opt.detach()))


# _vertices_draw = np.asarray(copy.deepcopy(_vertices.detach()))
pcd.points = o3d.utility.Vector3dVector(_vertices_draw)
_color = np.ones([9067,3])*np.array([0.5,0.3,1])
pcd.colors = o3d.utility.Vector3dVector(_color)
# _pcd = o3d.geometry.PointCloud()
# _pcd.points = o3d.utility.Vector3dVector(vertices)
# _list_joint.append(_pcd)
_list_joint.append(pcd)
body.draw_joint(_list_joint)
o3d.visualization.draw_geometries(_list_joint)















h, w = 200, 200
fmap = torch.zeros(h, w)


pts = torch.rand(20, 2)
pts *= torch.tensor([h, w])
x, y = pts.T.long()

fmap[x, y] = 1
sampler = torch.distributions.MultivariateNormal(pts.T, 10*torch.eye(len(pts)))
for x in range(10):
   x, y = sampler.sample()
   x, y = x.clip(0, h-1).long(), y.clip(0, w-1).long()
   fmap[x, y] = 1

plt.imshow(np.asarray(fmap))


h, w = 50, 50
x0, y0 = torch.rand(2, 20)
origins = torch.stack((x0*h, y0*w)).T
def gaussian_2d(x=0, y=0, mx=0, my=0, sx=1, sy=1):
    return 1 / (2*math.pi*sx*sy) * \
      torch.exp(-((x - mx)**2 / (2*sx**2) + (y - my)**2 / (2*sy**2)))

x = torch.linspace(0, h-1, h)
y = torch.linspace(0, w-1, w)
x, y = torch.meshgrid(x, y, indexing='xy')

z = torch.zeros(h, w)
for x0, y0 in origins:
  z += gaussian_2d(x, y, mx=x0, my=y0, sx=h/10, sy=w/10)

plt.imshow(np.asarray(z))
  







########################################################################
pytorch3d.renderer.cameras.FoVOrthographicCameras

# Setup
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

# Set paths
DATA_DIR = "./data"
obj_filename = os.path.join(DATA_DIR, "cow_mesh/cow.obj")

# Load obj file
mesh = load_objs_as_meshes([obj_filename], device=device)

plt.figure(figsize=(7,7))
texture_image=mesh.textures.maps_padded()
plt.imshow(texture_image.squeeze().cpu().numpy())
plt.axis("off");

plt.figure(figsize=(7,7))
texturesuv_image_matplotlib(mesh.textures, subsample=None)
plt.axis("off");







########################################################################



# Setup
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")


mesh = load_objs_as_meshes(['/Users/ik/Desktop/zepeto/blender/wip_find_weight.obj'], device)
mesh._verts_list[0] *= 2 
mesh._verts_padded *= 2 


camera = FoVOrthographicCameras(device=device, zfar=200, T = torch.tensor([[0,-0.51,10]]), znear=-100) 
# camera = OpenGLOrthographicCameras(device=device, zfar=200, T = torch.tensor([[0,-0.51,0]]), znear=-100) 
lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])

sigma = 1e-5
# sigma = 1e-6
raster_settings_soft = RasterizationSettings(
    image_size=256, 
    blur_radius=np.log(1. / sigma - 1.)*sigma, 
    faces_per_pixel=50, 
)

renderer_silhouette = MeshRenderer(
    rasterizer=MeshRasterizer(
        cameras=camera, 
        raster_settings=raster_settings_soft
    ),
    shader=SoftSilhouetteShader()
)

# sigma = 1e-4
# raster_settings_soft = RasterizationSettings(
#     image_size=128, 
#     blur_radius=np.log(1. / 1e-4 - 1.)*sigma, 
#     faces_per_pixel=50, 
#     perspective_correct=False, 
# )

# # Differentiable soft renderer using per vertex RGB colors for texture
# renderer_textured = MeshRenderer(
#     rasterizer=MeshRasterizer(
#         cameras=camera, 
#         raster_settings=raster_settings_soft
#     ),
#     shader=SoftPhongShader(device=device, 
#         cameras=camera,
#         lights=lights)
# )
import time
from pytorch3d.renderer.mesh import rasterize_meshes

pre =time.time()
_img_ = renderer_silhouette(mesh, cameras=camera, lights=lights)
print(time.time() - pre)
# _img_ = renderer_textured(mesh, cameras=camera, lights=lights)

plt.imshow(np.asarray(_img_[0,:,:,3]))

rasterizer=MeshRasterizer(cameras=camera, \
    raster_settings=RasterizationSettings(image_size=256, blur_radius=np.log(1. / sigma - 1.)*sigma, faces_per_pixel=50
))

fragments = rasterizer(mesh)



mesh = load_objs_as_meshes(['/Users/ik/Desktop/zepeto/blender/wip_find_weight.obj'], device)
tmp_vertices = _vertices - _vertices[3106,:]
scale = (torch.max(tmp_vertices, dim=0).values - torch.min(tmp_vertices, dim=0).values).max()
mesh._verts_list[0] = ((_vertices -_vertices[3106,:])/scale) * 2
mesh._verts_list[0] += torch.tensor([0, 0, 1])
mesh._verts_list[0] *= 3
# mesh._verts_padded *= 2 

pre =time.time()
fragments = rasterize_meshes(mesh, image_size = 128, blur_radius=0, faces_per_pixel=8,\
     perspective_correct= False, )
print(time.time() - pre)

plt.imshow(np.asarray(torch.sum(fragments[1][0], dim=2)/8+1))


