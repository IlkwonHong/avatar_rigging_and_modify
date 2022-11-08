from matplotlib.cbook import to_filehandle
from a_python.rigging_class.rig_hier_maya import *
import json
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import pickle as pkl
import scipy



def find_body_silhouette(img_seg, img_u, img_v):
    # '''몸통 가장자리 찾기, densepose 결과 사용해서 -> 커브 피팅에 사용'''
    # densepose uv 값으로 가장자리 가져올거라서 threshold
    v_thr = 0.01
    u_thr = 0.01
    samp_v = 0.65
    u_low = 0.25
    u_high = 1 - u_low

    h, w = img_seg.shape
    x = np.linspace(0, w-1, w)
    y = np.linspace(0, h-1, h)
    xx, yy = np.meshgrid(x,y)
    
    # 몸통
    mask_body = img_seg == 2
    mask_leg_l = img_seg == 10
    mask_leg_r = img_seg == 9
    

    # u, v 맵 마스킹해서 몸통 바운딩박스 -> for문 돌리기 위한 범위 찾기용
    masked_u = img_u * mask_body[:,:]
    masked_v = img_v * mask_body[:,:]
    masked_u_leg_l = img_u * mask_leg_l[:,:]
    masked_u_leg_r = img_u * mask_leg_r[:,:]

    masked_u_mid = np.abs(masked_u - 0.5) < u_thr
    bb_y_max = yy[mask_body][np.argmax(yy[mask_body])]
    bb_y_min = yy[mask_body][np.argmin(yy[mask_body])]

    masked_u_thr_low = masked_u * (masked_u < u_low)
    masked_u_thr_high = masked_u * (masked_u > u_high)

    masked_u_thr_low_w_val = masked_u_thr_low > 0
    masked_u_thr_high_w_val = masked_u_thr_high > 0

    # 몸통 가장자리, 중앙(평균) 찾기
    pts_l = []
    pts_r = []
    pts_mid = []
    for i in range(round(bb_y_min), round(bb_y_max)):
        # 실루엣 왼쪽
        if masked_u_thr_low_w_val[i,:].sum() > 0:
            x_max = xx[i, masked_u_thr_low_w_val[i,:]][np.argmax(xx[i,masked_u_thr_low_w_val[i,:]])]
            if masked_u_leg_l[i,:].sum() > 0:
                x_max_leg = max(xx[i,mask_leg_l[i,:]])
                x_max = max(x_max, x_max_leg)
            pts_l.append([x_max, i])

        # 실루엣 오른쪽
        if masked_u_thr_high_w_val[i,:].sum() > 0:
            x_min = xx[i, masked_u_thr_high_w_val[i,:]][np.argmin(xx[i, masked_u_thr_high_w_val[i,:]])]
            if masked_u_leg_r[i,:].sum() > 0:
                x_min_leg = min(xx[i,mask_leg_r[i,:]])
                x_min = min(x_min, x_min_leg)
            pts_r.append([x_min, i])

        # u=0.5
        if masked_u_mid[i,:].sum() > 0:
            x_mid = xx[i, masked_u_mid[i,:]][np.argmin(xx[i, masked_u_mid[i,:]])]
            pts_mid.append([x_mid, i])

    pts_l = np.asarray(pts_l)
    pts_r = np.asarray(pts_r)
    pts_mid = np.asarray(pts_mid)
    
    return pts_l, pts_r, pts_mid


def find_upper_zepeto_joint(keypts):
    '''상체 척추선 일직선으로 잡아서 비율로 zepeto joint 찾기, 
    TODO: 다른 뭐가 가능할지 생각, 목 관절 위치 조금 다름(오픈포즈 어깨 중앙으로 했음)'''
    # upper body joint 일단 척추 일직선으로 했음 
    upper_body = [64, 63, 18, 17, 16, 65]
    len_joint_upper = np.linalg.norm(joint_mat[upper_body[:-1],:3,3] - joint_mat[upper_body[1:],:3,3], axis=1)
    len_joint_upper = len_joint_upper[1:] 
    len_joint_upper = len_joint_upper / len_joint_upper.sum()
    img_hip = (keypts[9,:2] + keypts[12,:2])/2
    img_neck = (keypts[2,:2] + keypts[5,:2])/2
    img_spine = (img_neck - img_hip) * len_joint_upper[-1] + img_hip
    img_chest = (img_neck - img_hip) * (len_joint_upper[-1] + len_joint_upper[-2]) + img_hip
    img_chsetupper = (img_neck - img_hip) * (len_joint_upper[-1] + len_joint_upper[-2] + len_joint_upper[-3])  + img_hip
    # 척추선
    vec_spine = img_neck - img_hip
    vec_spine = vec_spine/np.linalg.norm(vec_spine)
    _c = -(vec_spine[1]*img_hip[0] - vec_spine[0]*img_hip[1])
    line_spine = np.array([vec_spine[1], -vec_spine[0], _c])

    zepeto_joints_upperbody = [img_neck, img_chsetupper, img_chest, img_spine, img_hip]
    return np.asarray(zepeto_joints_upperbody), line_spine
    

def to_homogeneous(vec):
    return np.array([vec[0], vec[1], 1])

def pt_lines_intersect(line_1, line_2):
    x = (line_2[2]*line_1[1] - line_2[1]*line_1[2])/(line_2[1]*line_1[0] - line_1[1]*line_2[0])
    y = (line_2[2]*line_1[0] - line_2[0]*line_1[2])/(-line_2[1]*line_1[0] + line_1[1]*line_2[0])
    # z = -(line_1[0] * x + line_1[1] * y)
    return np.array([x,y,1])

def silhouette_projected_to_spine(line_spine, pts, zepeto_joints_upperbody):
    '''result: (교점x, 교점y, 거리), result_proj: 척추 축에 투영했을때의 x축, 0:hip'''
    result = []
    for i in range(pts.shape[0]):
        line_perepend = np.array([-line_spine[1], line_spine[0],\
            -(-line_spine[1]*pts[i,0] + line_spine[0]*pts[i,1])])
        line_perepend[2] = -(line_spine[1]*pts[i,0] + line_spine[0]*pts[i,1])
        pt_intersect = pt_lines_intersect(line_perepend, line_spine)
        dist = np.linalg.norm(pts[i] - pt_intersect[:2])
        result.append([pt_intersect[0], pt_intersect[1], dist])
    result = np.asarray(result)

    vec_hip_neck = zepeto_joints_upperbody[0] - zepeto_joints_upperbody[-1]
    vec_hip_neck = vec_hip_neck/np.linalg.norm(vec_hip_neck)
    # project to vec_hip_neck
    result_proj = result[:,:2] - zepeto_joints_upperbody[-1]
    result_proj = result_proj@vec_hip_neck.T

    zepeto_joints_proj = zepeto_joints_upperbody - zepeto_joints_upperbody[-1]
    zepeto_joints_proj = zepeto_joints_proj@vec_hip_neck.T
    return result, result_proj, zepeto_joints_proj


def upperbody_curve_fitting(img_seg, img_u, img_v, keypts, Debug=False):
    '''chestupper, chest, spine, 폭 '''
    pts_l, pts_r, pts_mid = find_body_silhouette(img_seg, img_u, img_v)
    # line_spine : hip 에서 neck으로 향하는 직선, zepeto_joints_upperbody 는 목부터
    zepeto_joints_upperbody, line_spine = find_upper_zepeto_joint(keypts)


    # curve fitting
    result_l, result_l_proj, zepeto_joints_proj = silhouette_projected_to_spine(line_spine, pts_l, zepeto_joints_upperbody)
    result_r, result_r_proj, _ = silhouette_projected_to_spine(line_spine, pts_r, zepeto_joints_upperbody)
    def func(x, a, b, c, d):
        return a*x**3 + b*x**2 + c*x + d
    def func_(x, popt):
        [a, b, c, d] = popt
        return a*x**3 + b*x**2 + c*x + d
    for_curve_x = np.concatenate([result_l_proj, result_r_proj])
    for_curve_y = np.concatenate([result_l[:,2],result_r[:,2]])
    popt, pcov = scipy.optimize.curve_fit(func,for_curve_x, for_curve_y)
    res_curvefitting = func_(for_curve_x, popt)

    # 상체 사이즈 키포인트들 
    upper_kpts = func_(zepeto_joints_proj[1:-1], popt)

    if Debug:
        plt.figure('curve_fitting_result')
        plt.scatter(for_curve_x, for_curve_y)
        plt.scatter(for_curve_x, res_curvefitting)
        plt.scatter(for_curve_x, -res_curvefitting)
        plt.scatter(zepeto_joints_proj, np.zeros_like(zepeto_joints_proj))
        plt.scatter(zepeto_joints_proj[1:-1], upper_kpts)

        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')
        
        plt.figure('important points')
        plt.imshow(img_seg)
        plt.scatter(pts_l[:,0], pts_l[:,1])
        plt.scatter(pts_r[:,0], pts_r[:,1])
        plt.scatter(pts_mid[:,0], pts_mid[:,1])
        plt.scatter(zepeto_joints_upperbody[:,0], zepeto_joints_upperbody[:,1])

        plt.figure('segmentation')
        plt.imshow(img_u)
        
        plt.show()
    return upper_kpts

#####################################################################################
#####################################################################################
#####################################################################################


# folder = 11
# path_img = "/Users/ik/Downloads/test_set_new/{}/test_resized.png".format(folder)
# path_json = "/Users/ik/Downloads/test_set_new/{}/test_resized_keypoints.json".format(folder)
# path_dp = "/Users/ik/Downloads/test_set_new/{}/dp_dump.pkl".format(folder)

# with open(path_json, 'r') as f:
#     data = json.load(f)
# with open(path_dp, 'rb') as f:
#     [img_seg, img_v, img_u,_] = pkl.load(f)
# keypts = np.asarray(data['people'][0]['pose_keypoints_2d']).reshape([-1,3])
# img = cv.imread(path_img)



# pts_l, pts_r, pts_mid = find_body_silhouette(img_seg, img_u, img_v)
# # line_spine : hip 에서 neck으로 향하는 직선, zepeto_joints_upperbody 는 목부터
# zepeto_joints_upperbody, line_spine = find_upper_zepeto_joint(keypts)


# # curve fitting
# result_l, result_l_proj, zepeto_joints_proj = silhouette_projected_to_spine(line_spine, pts_l, zepeto_joints_upperbody)
# result_r, result_r_proj, _ = silhouette_projected_to_spine(line_spine, pts_r, zepeto_joints_upperbody)
# def func(x, a, b, c, d):
#     return a*x**3 + b*x**2 + c*x + d
# def func_(x, popt):
#     [a, b, c, d] = popt
#     return a*x**3 + b*x**2 + c*x + d
# for_curve_x = np.concatenate([result_l_proj, result_r_proj])
# for_curve_y = np.concatenate([result_l[:,2],result_r[:,2]])
# popt, pcov = scipy.optimize.curve_fit(func,for_curve_x, for_curve_y)
# res_curvefitting = func_(for_curve_x, popt)

# # 상체 사이즈 키포인트들 
# upper_kpts = func_(zepeto_joints_proj[1:-1], popt)






