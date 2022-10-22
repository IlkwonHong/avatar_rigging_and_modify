from a_python.rigging_class.rig_hier_maya import *
import json
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import pickle as pkl


path_img = "/Users/ik/Desktop/zepeto/a_python/test_data/1/test_resized.png"
path_json = "/Users/ik/Desktop/zepeto/a_python/test_data/1/test_resized_keypoints.json"
path_dp = "/Users/ik/Desktop/zepeto/a_python/test_data/dp_dump.pkl"


with open(path_json, 'r') as f:
    data = json.load(f)

with open(path_dp, 'rb') as f:
    data_dp = pkl.load(f)

keypts = np.asarray(data['people'][0]['pose_keypoints_2d']).reshape([-1,3])
img = cv.imread(path_img)

if False:
    plt.imshow(img)
    plt.scatter(keypts[:,0], keypts[:,1])
    plt.show()

    plt.figure(1)
    plt.imshow(data_dp[0])
    plt.figure(2)
    plt.imshow(data_dp[1])
    plt.figure(3)
    plt.imshow(data_dp[2])
    plt.figure(4)
    plt.imshow(data_dp[3])
    plt.show()

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

body.childs[1].childs[0]
body.childs[1].childs[0]

_list_joint = []

body.scale_iso_hier(1.5)

vertices

# body.reset_vertices()
_list_joint = []
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(_vertices)
_color = np.ones([9067,3])*np.array([0.5,0.3,1])
pcd.colors = o3d.utility.Vector3dVector(_color)

# _pcd = o3d.geometry.PointCloud()
# _pcd.points = o3d.utility.Vector3dVector(vertices)
# _list_joint.append(_pcd)

_list_joint.append(pcd)


tmp.draw_joint(_list_joint)
o3d.visualization.draw_geometries(_list_joint)
