import xml.etree.ElementTree as ET
import numpy as np
import pickle as pkl

tree = ET.parse("./data/maya_weight_modi.xml")
root = tree.getroot()

joint_name = []
joint_weight = []
joint_group_verts = []
for i in range(2,len(root)):
    joint_name.append(root[i].attrib['source'])
    _joint_weight = []
    _joint_group_verts = []
    for child in root[i]:
        _joint_group_verts.append(int(child.attrib['index']))
        _joint_weight.append(float(child.attrib['value']))
    joint_group_verts.append(_joint_group_verts)
    joint_weight.append(_joint_weight)
print('end')

with open('./data/maya_weight.pkl', 'wb') as f:
    pkl.dump([joint_name, joint_weight, joint_group_verts], f)