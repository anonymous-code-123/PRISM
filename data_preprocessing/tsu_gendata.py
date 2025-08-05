import argparse
import math
import pickle
from tqdm import tqdm
import sys
import json
sys.path.extend(['../'])
from data_gen.preprocess import pre_normalization2d


# joints distrubution
joints = ['head', 'nose' ,'Neck' ,'Chest' ,'Mhip' ,'Lsho' ,'Rsho', 'Lelb', 'Relb', 'Lwri', 'Rwri', 'Lhip', 'Rhip', 'Lkne', 'Rkne', 'Lank' ,'Rank']
lcrnet = {'nose': 12, 'head':12, 'Lsho':11, 'Rsho':10, 'Lelb':9, 'Relb':8, 'Lwri':7, 'Rwri':6, 'Lhip':5, 'Rhip':4, 'Lkne':3, 'Rkne':2, 'Lank':1 ,'Rank':0}

action_classes = {
'Cook.Cleandishes':0,'Cook.Cleanup':1,'Cook.Cut':2, 'Cook.Stir':3, 'Cook.Usestove':4, 
'Cutbread':5, 'Drink.Frombottle':6, 'Drink.Fromcan':7, 'Drink.Fromcup':8, 'Drink.Fromglass':9,
'Eat.Attable':10, 'Eat.Snack':11, 'Enter':12,'Getup':13, 'Laydown':14, 'Leave':15, 
'Makecoffee.Pourgrains':16, 'Makecoffee.Pourwater':17, 'Maketea.Boilwater':18, 'Maketea.Insertteabag':19, 'Pour.Frombottle':20, 
'Pour.Fromcan':21, 'Pour.Fromkettle':22, 'Readbook':23, 'Sitdown':24, 'Takepills':25, 
'Uselaptop':26, 'Usetelephone':27, 'Usetablet':28, 'Walk':29, 'WatchTV':30
}

action_cv = {
'Cutbread':0, 'Drink.Frombottle':1, 'Drink.Fromcan':2, 'Drink.Fromcup':3, 'Drink.Fromglass':4, 'Eat.Attable':5, 
'Eat.Snack':6, 'Enter':7, 'Getup':8, 'Leave':9, 'Pour.Frombottle':10, 'Pour.Fromcan':11, 
'Readbook':12, 'Sitdown':13, 'Takepills':14, 'Uselaptop':15, 'Usetablet':16,
'Usetelephone':17, 'Walk':18
}

nb_videos = {
'Cook.Cleandishes': 378, 'Cook.Cleanup': 380, 'Cook.Cut': 178, 'Cook.Stir': 579, 'Cook.Usestove': 96, 'Cutbread': 45, 'Drink.Frombottle': 341, 'Drink.Fromcan': 325, 'Drink.Fromcup': 2241, 'Drink.Fromglass': 65, 'Eat.Attable': 617, 'Eat.Snack': 216, 'Enter': 444, 'Getup': 833, 'Laydown': 181, 'Leave': 416, 'Makecoffee.Pourgrains': 64, 'Makecoffee.Pourwater': 76, 'Maketea.Boilwater': 62, 'Maketea.Insertteabag': 56, 'Pour.Frombottle': 276, 'Pour.Fromcan': 59, 'Pour.Fromkettle': 107, 'Readbook': 942, 'Sitdown': 1116, 'Takepills': 344, 'Uselaptop': 396, 'Usetelephone': 451, 'Usetablet': 49, 'Walk': 4070, 'WatchTV': 712
}


#CS
training_subjects = [3,4,6,7,9,12,13,15,17,19,25] 

#CV
training_cameras1 = [1]
training_cameras2 = [1,3,4,6,7]
testing_cameras=[2]

max_body_true = 2
max_body = 2
num_joint = 17
max_frame = 51000
num_classes = 51

import numpy as np
import os


def read_skeleton_filter(file):
    with open(file, 'r') as json_data:
        skeleton_sequence = json.load(json_data)

    return skeleton_sequence


def get_nonzero_std(s): 
    index = s.sum(-1).sum(-1) != 0  
    s = s[index]
    if len(s) != 0:
        s = s[:, :, 0].std() + s[:, :, 1].std()
    else:
        s = 0
    return s
    
def normalize_screen_coordinates( X, w, h):
    assert X.shape[-1] == 2
    zeros=np.where(X==0)
    # Normalize so that [0, w] is mapped to [-1, 1], while preserving the aspect ratio
    center= X/w*2 - [1, h/w]
    center[zeros]=0
    return center


def read_xyz(file, max_body=2, num_joint=17):  
    seq_info = read_skeleton_filter(file)
    data = np.zeros((max_body, len(seq_info['frames']), num_joint, 2))
    nb_frames = len(seq_info['frames'])
    for n, f in enumerate(seq_info['frames']):
        if len(f) != 0:
            for m, b in enumerate(f):
                if m < max_body:
                    for j,k in enumerate(joints):
                        if k == 'Mhip':
                            data[m, n, j, :] = [ (b['pose2d'][4] + b['pose2d'][5])/2, (b['pose2d'][17] + b['pose2d'][18])/2 ]
                        elif k == 'Neck':
                            data[m, n, j, :] = [ (b['pose2d'][10] + b['pose2d'][11])/2, (b['pose2d'][23] + b['pose2d'][24])/2 ]  
                        elif k == 'Chest':
                            data[m, n, j, :] = [ (b['pose2d'][4] + b['pose2d'][5] + b['pose2d'][10] + b['pose2d'][11])/4, (b['pose2d'][17] + b['pose2d'][18] + b['pose2d'][23] + b['pose2d'][24])/4 ]  
                        else:
                            data[m, n, j, :] = [ b['pose2d'][lcrnet[k]], b['pose2d'][lcrnet[k] + 13] ]
                    data[m, n, 1, :]=(data[m, n, 0, :]+data[m, n, 2, :])/2
                else:
                    pass

    # select the max energy body
    energy = np.array([get_nonzero_std(x) for x in data])
    index = energy.argsort()[::-1][0:max_body_true]
    data = data[index]
    
    
    # centralization
    for i in range(data.shape[0]):
        keypoints=data[i,:,:,:2]
        keypoints = normalize_screen_coordinates(keypoints[..., :2], w=640, h=480)
        data[i,:,:,:2]=keypoints


    data = data.transpose(3, 1, 2, 0)
    return data, nb_frames


def gendata(data_path, out_path, split_path, benchmark='xview', part='eval'):
    with open(split_path, 'r') as f:
        data_split = json.load(f)
    ignored_samples = []
    sample_name = []
    sample_label = []
    for filename in os.listdir(data_path):
        vid = filename[:-5]
        istraining = data_split[vid]['subset'] == 'training'
        istesting = data_split[vid]['subset'] == 'testing'
        if part == 'train':
            issample = istraining
        elif part == 'val':
            issample = istesting
        else:
            raise ValueError()

        if issample:
            sample_name.append(filename) 
    '''
    fp = np.zeros((len(sample_name), 2, max_frame, num_joint, max_body_true), dtype=np.float32)

    for i, s in enumerate(tqdm(sample_name[:])):
        vid = s[:-5]
        data, nb_frame = read_xyz(os.path.join(data_path, s), max_body = max_body, num_joint = num_joint)
        fp[i, :, 0:data.shape[1], :, :] = data
        fp[i, -1, -1, -1, -1] = nb_frame

        label = np.zeros((max_frame, num_classes), np.float32)
        fps = float(nb_frame/float(data_split[vid]['duration']))
        for ann in data_split[vid]['actions']:
            for fr in range(0, nb_frame, 1):
                # print (fr,num_feat,fps)
                if fr/fps > ann[1] and fr/fps < ann[2]:
                    label[fr, ann[0]] = 1 # bi
        sample_label.append((label, data_split[vid]['duration'])) 

    with open('{}/{}_label.pkl'.format(out_path, part), 'wb') as f:
        pickle.dump((sample_name, list(sample_label)), f)

    #fp = pre_normalization2d(fp)
  
    np.save('{}/{}_data_joint.npy'.format(out_path, part), fp)
    '''
    # window
    ws = 300
    sample_name_new = []
    for i, s in enumerate(tqdm(sample_name[:])):
        vid = s[:-5]
        data, nb_frame = read_xyz(os.path.join(data_path, s), max_body = max_body, num_joint = num_joint)
        nb_split = math.ceil(nb_frame/ws)
        fp = np.zeros((nb_split, 2, ws+1, num_joint, max_body_true), dtype=np.float32)
        labelp = np.zeros((nb_split, ws, num_classes), np.float32)
        
        label = np.zeros((nb_frame, num_classes), np.float32)
        fps = float(nb_frame/float(data_split[vid]['duration']))
        for ann in data_split[vid]['actions']:
            for fr in range(0, nb_frame, 1):
                # print (fr,num_feat,fps)
                if fr/fps > ann[1] and fr/fps < ann[2]:
                    label[fr, ann[0]] = 1 # bi


        for f in range(nb_split):
        
            fp[f, :, 0:min(ws, data.shape[1]-ws*f), :, :] = data[:,ws*f:min(ws*f+ws, data.shape[1]),:,:]
            fp[f, -1, -1, -1, -1] = min(ws, data.shape[1]-ws*f)
            labelp[f, 0:min(ws, data.shape[1]-ws*f), :] = label[ws*f:min(ws*f+ws, data.shape[1]),:]
            sample_label.append((labelp[f], data_split[vid]['duration']))
            sample_name_new.append(sample_name[i])
        if i == 0:
            out = fp
        else:
            out = np.concatenate((out, fp), axis=0)
    with open('{}/{}_label.pkl'.format(out_path, part), 'wb') as f:
        pickle.dump((sample_name_new, list(sample_label)), f)

    #fp = pre_normalization2d(fp)
  
    np.save('{}/{}_data_joint.npy'.format(out_path, part), out)
    print(len(sample_label), out.shape[0])


        


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Smarthome Data Converter.')
    parser.add_argument('--data_path', default='/data/stars/user/rdai/smarthome_untrimmed/untrimmed_json_new/')
    parser.add_argument('--split_path',
                        default='../../Toyota_Smarthome/pipline/data/smarthome_CV_51.json')

                        #default='/data/stars/user/dyang/project2022/unbalanced_data_unsynchronised_footage/smarthome_CV_51.json')
    parser.add_argument('--out_folder', default='../data/tsu-w300/')

    benchmark = ['xview']#, 'xview']
    part = ['train', 'val']
    arg = parser.parse_args()
    print('skeleton path: ', arg.data_path)
    for b in benchmark:
        for p in part:
            out_path = os.path.join(arg.out_folder, b)
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            print(b, p)
            gendata(
                arg.data_path,
                out_path,
                arg.split_path,
                benchmark=b,
                part=p)
