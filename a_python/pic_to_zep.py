from a_python.rigging_class.rig_hier_maya import *
import json
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import pickle as pkl


folder = 5
path_img = "/Users/ik/Desktop/zepeto/a_python/test_data/{}/test_resized.png".format(folder)
path_json = "/Users/ik/Desktop/zepeto/a_python/test_data/{}/test_resized_keypoints.json".format(folder)
path_dp = "/Users/ik/Desktop/zepeto/a_python/test_data/{}/dp_dump.pkl".format(folder)


with open(path_json, 'r') as f:
    data = json.load(f)

with open(path_dp, 'rb') as f:
    [img_seg, img_v, img_u,_] = pkl.load(f)
h,w = img_seg.shape

keypts = np.asarray(data['people'][0]['pose_keypoints_2d']).reshape([-1,3])
img = cv.imread(path_img)

if False:
    plt.imshow(img)
    plt.scatter(keypts[:,0], keypts[:,1])
    plt.show()

    plt.figure(1)
    plt.imshow(img_seg)
    plt.figure(2)
    plt.imshow(img_v)
    plt.figure(3)
    plt.imshow(img_u)
    plt.show()

if False:
    len_shoulder = [2,5]
    len_pelvis = [9,12]
    leg_L = [9,10,11]
    leg_R = [12,13,14]

    len_spine = np.linalg.norm(((keypts[2,:] + keypts[5,:]) - (keypts[9,:] + keypts[12,:]))/2)
    len_pelvis = np.linalg.norm((keypts[9,:] - keypts[12,:]))
    len_leg_upper = (np.linalg.norm(keypts[9,:] - keypts[10,:]) + np.linalg.norm(keypts[12,:] - keypts[13,:]))/2
    len_leg_lower = (np.linalg.norm(keypts[10,:] - keypts[11,:]) + np.linalg.norm(keypts[13,:] - keypts[14,:]))/2

    len_maya_pelvis = np.linalg.norm(joint_mat[6,:3,3] - joint_mat[15,:3,3])
    len_maya_leg_upper = np.linalg.norm(joint_mat[6,:3,3] - joint_mat[4,:3,3])
    len_maya_leg_lower = np.linalg.norm(joint_mat[4,:3,3] - joint_mat[1,:3,3])

    scale_img_to_maya = len_maya_pelvis / len_pelvis

    scale_to_align_leg_upper = len_leg_upper * scale_img_to_maya / len_maya_leg_upper
    scale_to_align_leg_lower = (len_leg_lower * scale_img_to_maya / len_maya_leg_lower)/scale_to_align_leg_upper


body = rig_class(65)
_vertices = copy.deepcopy(vertices)

# body.childs[0].trans_head(np.array([0,-10,0]), _vertices)
crotch = body.scaling_joint_len(_vertices, keypts, img_seg)
len_maya_thigh_crotch, len_maya_knee, len_maya_ankle = body.len_leg_seg(_vertices)


# TODO : 이부분 클래스 안으로 넣을 수 있으면 넣기
# upper key length
upper_body = [64, 63, 18, 17, 16, 65]
len_joint_upper = np.linalg.norm(joint_mat[upper_body[:-1],:3,3] - joint_mat[upper_body[1:],:3,3], axis=1)
len_joint_upper = len_joint_upper[1:] 
len_joint_upper = len_joint_upper / len_joint_upper.sum()
img_hip = (keypts[9,:2] + keypts[12,:2])/2
img_neck = (keypts[2,:2] + keypts[5,:2])/2
img_spine = (img_neck - img_hip) * len_joint_upper[-1] + img_hip
img_chest = (img_neck - img_hip) * (len_joint_upper[-1] + len_joint_upper[-2]) + img_hip
img_chsetupper = (img_neck - img_hip) * (len_joint_upper[-1] + len_joint_upper[-2] + len_joint_upper[-3])  + img_hip



body.childs[0].name
body.childs[0].childs[0].name

_len_spine_maya = np.linalg.norm(body.head - body.childs[0].head) 
_len_spine_img = np.linalg.norm(img_spine - img_hip)

# 상체 조인트 맞춤
_spine_trans = -(_len_spine_maya - _len_spine_img) * ((-body.head + body.childs[0].head) / np.linalg.norm(body.head - body.childs[0].head))
body.childs[0].trans_head(_spine_trans, _vertices)
_scale_spine_hier = (np.linalg.norm(img_hip - img_neck) * len_joint_upper[-1]) / (np.linalg.norm(body.childs[0].head - body.childs[0].childs[0].head))
body.childs[0].scale_iso_hier(_scale_spine_hier, _vertices)





# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(_vertices)
# joint_ = o3d.geometry.TriangleMesh.create_sphere(radius=1)
# joint_.translate(body.head)
# o3d.visualization.draw_geometries([pcd,joint_])







######

if True:
    # spine 두께
    thr = 0.015
    v_spine = img_v[round(img_spine[1]), round(img_spine[0])]
    _mask1 = img_seg==2
    _mask2 = (img_v < v_spine+thr) * (img_v > v_spine-thr) 
    _mask = _mask1 * _mask2
    plt.imshow(img_seg)
    plt.imshow(_mask)
    h, w = img_seg.shape

    x = np.linspace(0, w-1, w)
    y = np.linspace(0, h-1, h)
    xx, yy = np.meshgrid(x,y)

    _tmp_max = np.array([xx[_mask][np.argmax(xx[_mask])] , yy[_mask][np.argmax(xx[_mask])]])
    _tmp_min = np.array([xx[_mask][np.argmin(xx[_mask])] , yy[_mask][np.argmin(xx[_mask])]])
    
    # TODO : 이거 너비 반영 안 했음 wip
    img_spine_width = np.linalg.norm(_tmp_max - _tmp_min)



    # 허벅지 가랑이 부분 두께
    vec_thigh = (keypts[13,:2] - keypts[12,:2])/ np.linalg.norm((keypts[13,:] - keypts[12,:]))
    vec_thigh_perpend = np.array([vec_thigh[1], -vec_thigh[0]])
    tmp  = ((crotch - keypts[12,:2]).T @ vec_thigh) * vec_thigh + keypts[12,:2]

    val = []
    t = 0
    _tmp = tmp + vec_thigh_perpend * t
    while img_seg[round(_tmp[1]), round(_tmp[0])] == 10:
        _tmp = tmp + vec_thigh_perpend * t
        print(t)
        t+=1
    _max = t-1
    val.append([round(_tmp[0]), round(_tmp[1])])
    t = 0
    _tmp = tmp + vec_thigh_perpend * t
    while img_seg[round(_tmp[1]), round(_tmp[0])] == 10:
        _tmp = tmp + vec_thigh_perpend * t
        print(t)
        t-=1
    _min = t+1
    val.append([round(_tmp[0]), round(_tmp[1])])
    val = np.asarray(val)

    # left leg
    vec_thigh = (keypts[13,:2] - keypts[12,:2])/ np.linalg.norm((keypts[13,:] - keypts[12,:]))
    if vec_thigh[1]>0:
        vec_thigh_perpend = np.array([vec_thigh[1], -vec_thigh[0]])
    else:
        vec_thigh_perpend = np.array([vec_thigh[1], -vec_thigh[0]]) * -1

    vec_shin = (keypts[14,:2] - keypts[13,:2])/ np.linalg.norm((keypts[14,:] - keypts[13,:]))
    if vec_shin[1]>0:
        vec_shin_perpend = np.array([vec_shin[1], -vec_shin[0]])
    else:
        vec_shin_perpend = np.array([vec_shin[1], -vec_shin[0]]) * -1


    # 무릎 in 허벅지
    knee_width_wrt_thigh = []

    _tmp = [keypts[13,0], keypts[13,1]]
    while((img_seg[round(_tmp[1]), round(_tmp[0])] == 10) + \
        (img_seg[round(_tmp[1]), round(_tmp[0])] == 14) + \
            (img_seg[round(_tmp[1]), round(_tmp[0])] == 5 ) + \
                (img_seg[round(_tmp[1]), round(_tmp[0])] == 2)):
        _tmp += vec_thigh_perpend

    _tmp -= vec_thigh_perpend
    knee_width_wrt_thigh.append([round(_tmp[0]), round(_tmp[1])])

    _tmp = [keypts[13,0], keypts[13,1]]
    while((img_seg[round(_tmp[1]), round(_tmp[0])] == 10) + \
        (img_seg[round(_tmp[1]), round(_tmp[0])] == 14) + \
            (img_seg[round(_tmp[1]), round(_tmp[0])] == 5 ) + \
                (img_seg[round(_tmp[1]), round(_tmp[0])] == 2)):
        _tmp -= vec_thigh_perpend 

    _tmp += vec_thigh_perpend 
    knee_width_wrt_thigh.append([round(_tmp[0]), round(_tmp[1])])

    # 무릎 in 정강이
    knee_width_wrt_shin = []

    _tmp = [keypts[13,0], keypts[13,1]]
    while((img_seg[round(_tmp[1]), round(_tmp[0])] == 10) + \
        (img_seg[round(_tmp[1]), round(_tmp[0])] == 14) + \
            (img_seg[round(_tmp[1]), round(_tmp[0])] == 5 ) + \
                (img_seg[round(_tmp[1]), round(_tmp[0])] == 2)):
        _tmp += vec_shin_perpend

    _tmp -= vec_shin_perpend
    knee_width_wrt_shin.append([round(_tmp[0]), round(_tmp[1])])

    _tmp = [keypts[13,0], keypts[13,1]]

    while((img_seg[round(_tmp[1]), round(_tmp[0])] == 10) + \
        (img_seg[round(_tmp[1]), round(_tmp[0])] == 14) + \
            (img_seg[round(_tmp[1]), round(_tmp[0])] == 5 ) + \
                (img_seg[round(_tmp[1]), round(_tmp[0])] == 2)):
        _tmp -= vec_shin_perpend 

    _tmp += vec_shin_perpend 
    knee_width_wrt_shin.append([round(_tmp[0]), round(_tmp[1])])

    # 발목 in 정강이
    ankle = []

    _tmp = [keypts[14,0], keypts[14,1]]
    while((img_seg[round(_tmp[1]), round(_tmp[0])] == 10) + \
        (img_seg[round(_tmp[1]), round(_tmp[0])] == 14) + \
            (img_seg[round(_tmp[1]), round(_tmp[0])] == 5 ) + \
                (img_seg[round(_tmp[1]), round(_tmp[0])] == 2)):
        _tmp += vec_shin_perpend

    _tmp -= vec_shin_perpend
    ankle.append([round(_tmp[0]), round(_tmp[1])])

    _tmp = [keypts[14,0], keypts[14,1]]

    while((img_seg[round(_tmp[1]), round(_tmp[0])] == 10) + \
        (img_seg[round(_tmp[1]), round(_tmp[0])] == 14) + \
            (img_seg[round(_tmp[1]), round(_tmp[0])] == 5 ) + \
                (img_seg[round(_tmp[1]), round(_tmp[0])] == 2)):
        _tmp -= vec_shin_perpend 

    _tmp += vec_shin_perpend 
    ankle.append([round(_tmp[0]), round(_tmp[1])])

    knee_width_wrt_shin = np.asarray(knee_width_wrt_shin)
    knee_width_wrt_thigh = np.asarray(knee_width_wrt_thigh)
    ankle = np.asarray(ankle)

    # 길이
    len_thigh_crotch = np.linalg.norm(val[0,:] - val[1,:])
    len_knee = np.linalg.norm(knee_width_wrt_thigh[0,:] - knee_width_wrt_thigh[1,:])
    len_ankle = np.linalg.norm(ankle[0,:] - ankle[1,:])






    plt.figure(1)
    plt.imshow(img_seg)
    plt.imshow(img_v)
    plt.scatter(val[:,0], val[:,1])
    plt.scatter(knee_width_wrt_thigh[:,0], knee_width_wrt_thigh[:,1])
    plt.scatter(knee_width_wrt_shin[:,0], knee_width_wrt_shin[:,1])
    plt.scatter(ankle[:,0], ankle[:,1])
    plt.show()

scale_knee = len_knee / len_maya_knee
scale_thigh_crotch = len_thigh_crotch/(len_maya_thigh_crotch * scale_knee)
scale_ankle = len_ankle / (len_maya_ankle * scale_knee)




body.childs[1].childs[0].scale_y(scale_thigh_crotch, _vertices)
body.childs[1].childs[0].scale_z(scale_thigh_crotch, _vertices)
body.childs[1].childs[0].scale_y_hier(scale_knee, _vertices)
body.childs[1].childs[0].scale_z_hier(scale_knee, _vertices)
body.childs[1].childs[0].childs[0].scale_y_hier(scale_ankle, _vertices)
body.childs[1].childs[0].childs[0].scale_z_hier(scale_ankle, _vertices)

body.childs[1].childs[1].scale_y(scale_thigh_crotch, _vertices)
body.childs[1].childs[1].scale_z(scale_thigh_crotch, _vertices)
body.childs[1].childs[1].scale_y_hier(scale_knee, _vertices)
body.childs[1].childs[1].scale_z_hier(scale_knee, _vertices)
body.childs[1].childs[1].childs[0].scale_y_hier(scale_ankle, _vertices)
body.childs[1].childs[1].childs[0].scale_z_hier(scale_ankle, _vertices)


body.childs[1].childs[1].childs[0].childs[0].childs[0].childs[0].scale_iso_hier(0.75, _vertices)
body.childs[1].childs[0].childs[0].childs[0].childs[0].childs[0].scale_iso_hier(0.75, _vertices)


######

# plt.imshow(img_seg)
plt.imshow(img_v)
# plt.scatter(img_neck[0], img_neck[1])
plt.scatter(img_spine[0], img_spine[1])
# plt.scatter(img_chest[0], img_chest[1])
# plt.scatter(img_chsetupper[0], img_chsetupper[1])
# plt.scatter(img_hip[0], img_hip[1])
plt.show()










# visualize
if True:
    _list_joint = []
    _list_joint = []
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(_vertices)
    _color = np.ones([9067,3])*np.array([0.5,0.3,1])
    pcd.colors = o3d.utility.Vector3dVector(_color)
    # _pcd = o3d.geometry.PointCloud()
    # _pcd.points = o3d.utility.Vector3dVector(vertices)
    # _list_joint.append(_pcd)
    _list_joint.append(pcd)
    body.draw_joint(_list_joint)
    o3d.visualization.draw_geometries(_list_joint)



    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(_vertices)
    _color = np.ones([9067,3])*np.array([0.5,0.3,1])
    _color[3106,:] = np.array([0.1,0.9,1])
    pcd.colors = o3d.utility.Vector3dVector(_color)

    o3d.visualization.draw_geometries([pcd])

with open('/Users/ik/Desktop/zepeto/a_python/to_py_mesh_.pkl', 'wb') as f:
    pkl.dump(_vertices , f)

