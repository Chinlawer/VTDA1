import os, sys, math, random
import numpy as np
import scipy.io as sio
from torch.utils.data import Dataset
from path import Path
import torch
import torchvision.transforms as Tr
import pandas as pd
from tqdm import tqdm
from collections import Counter
import random
from .utils import read_from_path, offread_uniformed
from .shapenet import ShapeNet
from .transforms import default_pc_transform
from pytorch3d.transforms import axis_angle_to_matrix
from collections import Counter

import plyfile
import h5py
import transforms3d
import open3d
import torchnet as tnt
# import pcl


class SessionDataset(Dataset):
    def __init__(self, _data, transform=default_pc_transform):
        super().__init__()
        self.paths = []
        self.labels = []
        self.load_method = []
        for i, path_list in enumerate(_data):
            #print(i, len(path_list))
            for path, load_method in path_list:
                self.paths.append(path)
                self.labels.append(i)
                self.load_method.append(load_method)
        self.transform = transform
        
    def save(self, save_path):
        with open(save_path, 'w') as f:
            for path in self.paths:
                print(path, file=f)
                
    def check(self, cmp_path):
        cmp_set = set()
        self_set = set()
        with open(cmp_path, 'r') as f:
            for path in f.readlines():
                cmp_set.add(path.strip().split('/')[-1])
        for path in self.paths:
            self_set.add(path.strip().split('/')[-1])
        if cmp_set != self_set:
            print(len(cmp_set|self_set)-len(cmp_set&self_set))
        else:
            print('ok!')
            
    def get_cat_num(self): 
        return len(dict(Counter(self.labels)))
    
    def set_transform(self, transform):
        self.transform = transform
    
    def __getitem__(self, idx):
        if self.load_method[idx] is None:
            point_cloud = read_from_path(self.paths[idx])
        else:
            point_cloud = self.load_method[idx](self.paths[idx])
        if self.transform is not None:
            point_cloud = self.transform(point_cloud)
        #if point_cloud.shape[0] < 2000:
        #    return None
        #vertex = []
        #for i in range(point_cloud.shape[0]):
        #    vertex.append((point_cloud[i, 0], point_cloud[i, 1], point_cloud[i, 2]))
        #vertex = np.array(vertex, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
        #pp = "/".join(self.paths[idx].split('/')[:-1])
        #pp = pp + f'/{point_cloud.shape[0]}.ply'
        #print(pp)
        #el = plyfile.PlyElement.describe(vertex, 'vertex')
        #plyfile.PlyData([el]).write(pp)
        return point_cloud, self.labels[idx]
        
    def __len__(self):
        return len(self.paths)


class SessionMaker:
    """
    input: dataset(data_path, label_id), label-idx map
    store: paths of instances(train, test) 
    memory: examplar set
    session config: [base sets, incremental sets] -- [equal new classes per session, session list]
    output: dataset(train, test) for each session
    """
    def __init__(self):
        self.id2name = []
        self.name2id = {}
        self.data_train = [] # [label_0: [...], label_1: [...], ...] (path, load_method)
        self.data_test = [] # [label_0: [...], label_1: [...], ...] (path, load_method)
        self.cat_tot = 0 # total number of categories
        self.cat_cnt_train = [] # [0: cnt0, 1: cnt1, ...]
        self.cat_cnt_test = [] # [0: cnt0, 1: cnt1, ...]
        self.session_cfg = [] # [session_0: [], session_1: [], ...]  example: [[0, 1, 2, 3, 4, 5], [6, 7], [8, 9]]
        self.base_few_shot = 0
        self.inc_few_shot = 0
        # TODO: support tensor memory  (data, target)
        self.memory = [] # [((path, load_method)/data, label), ...]
    
    def update_memory(self, examplar):
        # print(examplar['path'])
        self.memory.append(examplar) 
    
    def tot_session(self):
        return len(self.session_cfg)
    
    def make_session(self, session_id, update_memory=0):
        """ return train_dataset & test dataset of session_id """
        # train data [session_i + memory]
        data_train = [[] for i in range(self.cat_tot)]
        tmp_new_mem = []
        for label in self.session_cfg[session_id]:
            if session_id == 0:
                data_train[label] = self.data_train[label]
                if self.base_few_shot > 0:
                    data_train[label] = data_train[label][:self.base_few_shot]
            else:
                data_train[label] = self.data_train[label]
                if self.inc_few_shot > 0:
                    data_train[label] = data_train[label][:self.inc_few_shot]
            # new memory examplar
            for path, load_method in data_train[label][:update_memory]:
                tmp_new_mem.append({'path': path, 'load_method': load_method, 'label': label})
        # memory
        for examplar in self.memory:
            data_train[examplar['label']].append((examplar['path'], examplar['load_method']))
        for new_examplar in tmp_new_mem:
            self.update_memory(new_examplar)
        
        # test data [session_0 + session_1 + ... + session_i]
        data_test = [[] for i in range(self.cat_tot)]
        for session in range(session_id + 1):
            for label in self.session_cfg[session]:
                data_test[label] = self.data_test[label]
        return SessionDataset(data_train), SessionDataset(data_test)

    def get_id2name(self):
        """ may used for prompt """
        return self.id2name
    
    def set_session_list(self, session_list, base_few_shot=0, inc_few_shot=5):      # default: inc_few_shot=5
        """ session list contains categories` name instead of id """
        self.session_cfg = [[self.name2id[name] for name in session] for session in session_list]
        self.base_few_shot = base_few_shot
        self.inc_few_shot = inc_few_shot
    
    def set_session(self, num_base_cat, num_inc_cat, base_few_shot=0, inc_few_shot=5):      # default: inc_few_shot=5
        """ in the order of appending datasets(categories) """
        res = num_base_cat
        self.session_cfg = [[i for i in range(num_base_cat)]]
        while res < self.cat_tot:
            self.session_cfg.append([i for i in range(res, min(res + num_inc_cat, self.cat_tot))])
            res += num_inc_cat
        self.base_few_shot = base_few_shot
        self.inc_few_shot = inc_few_shot
        
    def merge_new_data(self, 
                       new_data_train,
                       new_data_test,
                       new_cat_cnt_train,
                       new_cat_cnt_test,
                       new_id2name,
                       new_cat_tot,
                       new_dataset_name
                       ):
        """ assign new category idx, empty categories will be ignore """
        merged_cat_num = 0
        for i in range(new_cat_tot):
            if new_cat_cnt_train[i] == 0 and new_cat_cnt_test[i] == 0:
                continue
            self.data_train.append(new_data_train[i])
            self.data_test.append(new_data_test[i])
            self.cat_cnt_train.append(new_cat_cnt_train[i])
            self.cat_cnt_test.append(new_cat_cnt_test[i])
            self.id2name.append(new_id2name[i])
            self.name2id[new_id2name[i]] = self.cat_tot
            self.cat_tot += 1
            merged_cat_num += 1
        print(f"{merged_cat_num} categories has been merged from '{new_dataset_name}'.")
        
        
    def append_dataset(self, new_dataset, new_id2name, load_method=None, split_ratio=0.8):
        """
        dataset before splitted to (train, test)
        Args:
            new_dataset (iterable): dataset that return (path, label_id) when iterate
            new_id2name (list): label name list 
            split_ratio (float): the ratio of train/test after split
        """
        new_cat_tot = len(new_id2name)
        new_cat_cnt = [0 for i in range(new_cat_tot)]
        new_data = [[] for i in range(new_cat_tot)]  # before split
        # load paths
        for path, label in new_dataset:
            new_data[label].append((path, load_method))
            new_cat_cnt[label] += 1
        # split train/test
        new_data_train = [[] for i in range(new_cat_tot)]
        new_data_test = [[] for i in range(new_cat_tot)]
        new_cat_cnt_train = [0 for i in range(new_cat_tot)] 
        new_cat_cnt_test = [0 for i in range(new_cat_tot)] 
        for label, path_list in enumerate(new_data):
            num_train = int(new_cat_cnt[label] * split_ratio) # round
            num_test = new_cat_cnt[label] - num_train
            new_data_train[label] = path_list[:num_train + 1]
            new_data_test[label] = path_list[num_train + 1:]
            new_cat_cnt_train[label] = num_train
            new_cat_cnt_test[label] = num_test
        for name in new_id2name:
            if new_cat_cnt[new_id2name.index(name)] > 0 and name in self.id2name:
                print('duplicated category:', name)
        # merge
        self.merge_new_data(new_data_train, new_data_test, new_cat_cnt_train, new_cat_cnt_test, new_id2name, new_cat_tot, type(new_dataset).__name__)
        
    def append_dataset_train_test(self, new_dataset_train, new_dataset_test, new_id2name, load_method=None):
        new_cat_tot = len(new_id2name)
        new_data_train = [[] for i in range(new_cat_tot)] 
        new_data_test = [[] for i in range(new_cat_tot)]
        new_cat_cnt_train = [0 for i in range(new_cat_tot)] 
        new_cat_cnt_test = [0 for i in range(new_cat_tot)] 
        # load paths
        for path, label in new_dataset_train:
            new_data_train[label].append((path, load_method))
            new_cat_cnt_train[label] += 1
        for path, label in new_dataset_test:
            new_data_test[label].append((path, load_method))
            new_cat_cnt_test[label] += 1
        for name in new_id2name:
            if (new_cat_cnt_test[new_id2name.index(name)] > 0 or new_cat_cnt_train[new_id2name.index(name)] > 0) and name in self.id2name:
                print('duplicated category:', name)
        # merge
        self.merge_new_data(new_data_train, new_data_test, new_cat_cnt_train, new_cat_cnt_test, new_id2name, new_cat_tot, type(new_dataset_train).__name__)
    
    def info(self):
        info_dict = {}
        info_dict['category_num'] = self.cat_tot
        info_dict['categories'] = {i: name for i, name in enumerate(self.id2name)}
        info_dict['train_instance_num'] = sum(self.cat_cnt_train)
        info_dict['train_cat_cnt'] = {i: (self.id2name[i], num) for i, num in enumerate(self.cat_cnt_train)}
        info_dict['test_instance_num'] = sum(self.cat_cnt_test)
        info_dict['test_cat_cnt'] = {i: (self.id2name[i], num) for i, num in enumerate(self.cat_cnt_test)}
        info_dict['session_num'] = len(self.session_cfg)
        info_dict['session_cfg'] = {i: s for i, s in enumerate(self.session_cfg)}
        info_dict['base_few_shot'] = self.base_few_shot
        info_dict['inc_few_shot'] = self.inc_few_shot
        return info_dict
        

class ShapeNetCIL(Dataset):
    cat_labels = {'02691156': 0, '02747177': 1, '02773838': 2, '02801938': 3, '02808440': 4, '02818832': 5, '02828884': 6, '02843684': 7, '02871439': 8, '02876657': 9, '02880940': 10, '02924116': 11, '02933112': 12, '02942699': 13, '02946921': 14, '02954340': 15, '02958343': 16, 
                  '02992529': 17, '03001627': 18, '03046257': 19, '03085013': 20, '03207941': 21, '03211117': 22, '03261776': 23, '03325088': 24, '03337140': 25, '03467517': 26, '03513137': 27, '03593526': 28, '03624134': 29, '03636649': 30, '03642806': 31, '03691459': 32, '03710193': 33, 
                  '03759954': 34, '03761084': 35, '03790512': 36, '03797390': 37, '03928116': 38, '03938244': 39, '03948459': 40, '03991062': 41, '04004475': 42, '04074963': 43, '04090263': 44, '04099429': 45, '04225987': 46, '04256520': 47, '04330267': 48, '04379243': 49, '04401088': 50, 
                  '04460130': 51, '04468005': 52, '04530566': 53, '04554684': 54}
    id2name = ['airplane', 'ashcan', 'bag', 'basket', 'bathtub', 'bed', 'bench', 'birdhouse', 'bookshelf', 'bottle', 'bowl', 'bus', 'cabinet', 'camera', 'can', 'cap', 'car', 'cellular telephone', 'chair', 'clock', 'computer keyboard', 'dishwasher', 'display', 'earphone', 'faucet', 'file', 'guitar', 'helmet', 'jar', 'knife', 'lamp', 'laptop', 'loudspeaker', 'mailbox', 'microphone', 'microwave', 'motorcycle', 'mug', 'piano', 'pillow', 'pistol', 'pot', 'printer', 'remote control', 'rifle', 'rocket', 'skateboard', 'sofa', 'stove', 'table', 'telephone', 'tower', 'train', 'vessel', 'washer']
    def __init__(self, root='/data/qly/FILP-3D-main/data/ShapeNet55', partition='train', banlist=[], whole=False):
        assert partition in ['train', 'test']
        self.data_root = root + '/ShapeNet-55'
        self.pc_path = root + '/shapenet_pc'
        self.subset = partition
        
        self.data_list_file = os.path.join(self.data_root, f'{self.subset}.txt')
        test_data_list_file = os.path.join(self.data_root, 'test.txt')
        self.whole = whole

        with open(self.data_list_file, 'r') as f:
            lines = f.readlines()
        if self.whole:
            with open(test_data_list_file, 'r') as f:
                test_lines = f.readlines()
            lines = test_lines + lines
        self.file_list = []
        check_list = ['03001627-udf068a6b', '03001627-u6028f63e', '03001627-uca24feec', '04379243-', '02747177-', '03001627-u481ebf18', '03001627-u45c7b89f', '03001627-ub5d972a1', '03001627-u1e22cc04', '03001627-ue639c33f']
        
        # flag = False
        for line in lines:
            line = line.strip()
            taxonomy_id = line.split('-')[0]
            model_id = line.split('-')[1].split('.')[0]
            if ShapeNetCIL.id2name[ShapeNetCIL.cat_labels[taxonomy_id]] in banlist:
                continue
            if taxonomy_id + '-' + model_id not in check_list:
                self.file_list.append({
                    'taxonomy_id': taxonomy_id,
                    'model_id': model_id,
                    'file_path': line
                })
        self.partition = partition
    
    def __getitem__(self, idx):
        sample = self.file_list[idx]
        path = os.path.join(self.pc_path, sample['file_path'])
        return path, ShapeNetCIL.cat_labels[sample['taxonomy_id']]


class ModelNet40AlignCIL(Dataset):
    cats = {'airplane': 0, 'bathtub': 1, 'bed': 2, 'bench': 3, 'bookshelf': 4, 'bottle': 5, 'bowl': 6, 'car': 7, 'chair': 8, 'cone': 9, 'cup': 10, 'curtain': 11, 'desk': 12, 'door': 13, 'dresser': 14, 'flower_pot': 15, 'glass_box': 16, 'guitar': 17, 'keyboard': 18, 'lamp': 19, 'laptop': 20, 'mantel': 21, 'monitor': 22, 'night_stand': 23, 'person': 24, 'piano': 25, 'plant': 26, 'radio': 27, 'range_hood': 28, 'sink': 29, 'sofa': 30, 'stairs': 31, 'stool': 32, 'table': 33, 'tent': 34, 'toilet': 35, 'tv_stand': 36, 'vase': 37, 'wardrobe': 38, 'xbox': 39}
    id2name = list(cats.keys())
    '''
        points are randomly sampled from .off file, so the results of this dataset may be better or wrose than our claim results
    '''
    def __init__(self, root='/data/qly/FILP-3D-main/data/ModelNet40_manually_aligned', partition='train', banlist=[]): 
        assert partition in ('test', 'train')
        super().__init__()
        self.root = root
        self.partition = partition
        self._load_data(banlist)

    def _load_data(self, banlist):
        self.paths = []
        self.labels = []
       # from ipdb import set_trace
       # set_trace()

        for cat in os.listdir(self.root):
            if cat in banlist:
                continue
            cat_path = os.path.join(self.root, cat, self.partition)
            for case in os.listdir(cat_path):
                if case.endswith('.off'):
                # if case.endswith('.pt'):
                    self.paths.append(os.path.join(cat_path, case))
                    self.labels.append(ModelNet40AlignCIL.cats[cat])
    
    def get_load_method(self):
        def load(path, pt_num=1024):
            points = torch.Tensor(offread_uniformed(path, sampled_pt_num=pt_num)).type(torch.FloatTensor)
            rota1 = axis_angle_to_matrix(torch.tensor([0.5 * np.pi, 0, 0]))
            rota2 = axis_angle_to_matrix(torch.tensor([0, -0.5 * np.pi, 0]))
            points = points @ rota1 @ rota2
            return points.numpy()
        return load
        
    def __getitem__(self, index):      
        return self.paths[index], self.labels[index]
    
    def __len__(self):
        return len(self.labels)


# miss = ['orange', ]
class CO3DCIL(Dataset):
    def __init__(self, root='/data/CO3D', banlist=[]):
        self.data_root = root
        self.npoints = '2000'
        
        self.label2name = []
        self.name2label = {}
        cat_cnt = []
        self.file_list = []
        label = 0
        for cat in os.listdir(self.data_root):
            if cat in banlist:
                continue
            cat_cnt.append(0)
            self.name2label[cat] = label
            self.label2name.append(cat)
            for tax in [f for f in os.listdir(os.path.join(self.data_root, cat)) if not os.path.isfile(os.path.join(self.data_root, cat, f))]:
                if os.path.exists(os.path.join(self.data_root, cat, tax, f"{self.npoints}.ply")):
                    cat_cnt[-1] += 1
                    self.file_list.append({
                        'label': label,
                        'model_id': cat_cnt[-1],
                        'file_path': os.path.join(self.data_root, cat, tax, f"{self.npoints}.ply")
                    })
            label += 1

    def get_label2name(self):
        return self.label2name
        
    def __getitem__(self, idx):
        sample = self.file_list[idx]
        return sample['file_path'], sample['label']

    def __len__(self):
        return len(self.file_list)

class ScanObjectNN(Dataset):
    cats = {'Bag': 0, 'Bed': 1, 'Bin': 2, 'Box': 3, 'Cabinets': 4, 'Chair': 5, 'Desk': 6, 'Display': 7, 'Door': 8, 'Pillow': 9, 'Shelves': 10, 'Sink': 11, 'Sofa': 12, 'Table': 13, 'Toilet': 14}
    id2name = list(cats.keys())
    def __init__(self, partition='test', num_points=1024):
        assert partition in ('test', 'training')
        self.num_points = num_points
        points, labels = self._load_ScanObjectNN(partition)
        self._save_data2path(partition, points, labels)

    def _load_ScanObjectNN(self, partition):
        BASE_DIR = '/data/qly/RAL/DATASETS/h5_files/'
        DATA_DIR = os.path.join(BASE_DIR, 'main_split')
        h5_name = os.path.join(DATA_DIR, f'{partition}_objectdataset.h5')
        f = h5py.File(h5_name)
        # self.points = torch.from_numpy(f['data'][:].astype('float32')) 
        # self.labels = torch.from_numpy(f['label'][:].astype('int64')) 
        data = f['data'][:]
        label = f['label'][:]
        sampled, labels = self._get_current_data_h5(data, label, self.num_points)
        labels = np.squeeze(labels)
        points = self._rotate_and_jitter(sampled)
        return points, labels

    def _get_current_data_h5(self, pcs, labels, num_points):
        #shuffle points to sample
        idx_pts = np.arange(pcs.shape[1])
        np.random.shuffle(idx_pts)

        sampled = pcs[:,idx_pts[:num_points],:]
        #sampled = pcs[:,:num_points,:]

        #shuffle point clouds per epoch
        idx = np.arange(len(labels))
        np.random.shuffle(idx)

        sampled = sampled[idx]
        labels = labels[idx]
        labels = np.squeeze(labels)

        return sampled, labels
    
    def _rotate_and_jitter(self, data):
        """ Randomly rotate the point clouds to augument the dataset
            rotation is per shape based along up direction
            Input:
            BxNx3 array, original batch of point clouds
            Return:
            BxNx3 array, rotated batch of point clouds
        """
        rotated_data = np.zeros(data.shape, dtype=np.float32)
        for k in range(data.shape[0]):
            rotation_angle = np.random.uniform() * 2 * np.pi
            cosval = np.cos(rotation_angle)
            sinval = np.sin(rotation_angle)
            rotation_matrix = np.array([[cosval, 0, sinval],
                                        [0, 1, 0],
                                        [-sinval, 0, cosval]])
            shape_pc = data[k, ...]
            rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
        B, N, C = rotated_data.shape
        clip, sigma = 0.05, 0.01
        jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1*clip, clip)
        jittered_data += rotated_data
        return jittered_data

    def _save_data2path(self, partition, data, labels):
        cats = {'Bag': 0, 'Bed': 1, 'Bin': 2, 'Box': 3, 'Cabinets': 4, 'Chair': 5, 'Desk': 6, 'Display': 7, 'Door': 8, 'Pillow': 9, 'Shelves': 10, 'Sink': 11, 'Sofa': 12, 'Table': 13, 'Toilet': 14}
        self.paths = []
        self.labels = []
        save_dir = '/data/qly/RAL/DATASETS/scanobjectnn'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # 按照类别将点云数据分开并存储为.npz文件
        for category_name, category_id in cats.items():
            category_data = data[labels == category_id]  # 获取该类别下的所有点云数据
            category_folder = os.path.join(save_dir, category_name)
            if not os.path.exists(category_folder):
                os.makedirs(category_folder)
            category_folder_s = os.path.join(category_folder, f'{partition}')
            if not os.path.exists(category_folder_s):
                os.makedirs(category_folder_s)
            
            # 保存为npy文件，每个类别下的点云按文件保存
            for idx, point_cloud in enumerate(category_data):
                file_path = os.path.join(category_folder_s, f"{category_name}_{idx}.npy")
                np.save(file_path, point_cloud)
                self.paths.append(file_path)
                self.labels.append(self.cats[category_name])



    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        return self.paths[index], self.labels[index]

class SydneyUrbanObjects(Dataset):
    cats = {'4wd':0, 'building':1, 'bus':2, 'car':3, 'pedestrian':4, 'pillar':5, 'pole':6, 'traffic_lights':7, 
                'traffic_sign':8, 'tree':9, 'truck':10, 'trunk':11, 'ute':12, 'van':13}
    id2name = list(cats.keys())

    def __init__(self, partition ='train', banlist=[]): 

        assert partition in ('test', 'train')
        self.partition = partition
        self.banlist = banlist
        self.root = "/data/qly/RAL/DATASETS/sydney-urban-objects-dataset"
        self.npy_dir = "/data/qly/RAL/DATASETS/SUOD_generated"
        os.makedirs(self.npy_dir, exist_ok=True)

        self.paths = []
        self.labels = []

        if partition == "train":
            for num in range(1, 4):        # 0,1,2,3
                load_data = self._load_data(num)        
        else:
            load_data = self._load_data(0)
        
        return load_data
        
    def _load_data(self, num):
        dataset_file = os.path.join(self.root, f'folds/fold{num}.txt')
        with open(dataset_file, 'r') as f:
            
            data_dir = os.path.join(self.root, 'objects')
            file_list = f.readlines()

            for file in file_list:
                file_name = file.strip()
                file_path = os.path.join(data_dir, file_name)       # .bin

                cat = file_name.split('.')[0]    # category            

                # Load point cloud
                cloud = self._load_points_from_bin(file_path)       # (190, 3)

                # Data augmentation
                aug_steps = 12
                cloud_list = self._aug_data(cloud, aug_steps)       # (13, 190, 3)  13个点云样本，每个样本有190个点，每个点有三个坐标

                # Ensure save directory exists   
                save_dir = os.path.join(self.npy_dir, cat, self.partition)
                os.makedirs(save_dir, exist_ok=True)

                # Process each augmented point cloud
                idx = 0
                for points in cloud_list:
                    voxels, inside_points = self._voxelize(
                        points, voxel_size=(24, 24, 24), padding_size=(32, 32, 32), resolution=0.1
                        )
                    # Save voxelized data as .npy

                    if inside_points.shape[0] > 0:
                        # pc = pcl.PointCloud(points)
                        if points.shape[0] < 1024:
                            trimed_points = np.pad(points, ((0, 1024 - points.shape[0]), (0, 0)), mode='constant')
                        elif points.shape[0] > 1024:
                            indices = np.random.choice(points.shape[0], 1024, replace=False)  # 不放回抽样
                            trimed_points = points[indices] 
                        else:
                            trimed_points = points

                        save_name = os.path.join(save_dir, f"{file_name.split('.bin')[0]}_{idx}.npy")
                        np.save(save_name, trimed_points)
                        # pcd_name = os.path.join(save_dir, f"{file_name.split('.bin')[0]}_{idx}.pcd")
                        # pcl.save(pc, pcd_name)
                        self.paths.append(save_name)
                        self.labels.append(SydneyUrbanObjects.cats[cat])
                        idx += 1
                        # print(f'Saved PCD: {save_name}')



                    # if inside_points.shape[0] > 0:
                    #     save_name = os.path.join(save_dir, f"{file_name.split('.bin')[0]}_{idx}.npy")
                    #     np.save(save_name, voxels)
                    #     self.paths.append(save_name)
                    #     self.labels.append(SydneyUrbanObjects.cats[cat])
                    #     idx += 1

    def __getitem__(self, index):      
        return self.paths[index], self.labels[index]
    
    def __len__(self):
        return len(self.labels)
        
    def _load_points_from_bin(self, bin_file, with_intensity=False):
        """
        :param bin_file:
        :param with_intensity:
        :return: (N, 3) or (N, 4)
        """
        fields = ['t', 'intensity', 'id', 'x', 'y', 'z', 'azimuth', 'range', 'pid']
        types = ['int64', 'uint8', 'uint8', 'float32', 'float32', 'float32', 'float32', 'float32', 'int32']

        binType = np.dtype(dict(names=fields, formats=types))
        data = np.fromfile(bin_file, binType)

        # 3D points, one per row
        if with_intensity:
            points = np.vstack([data['x'], data['y'], data['z'], data['intensity']]).T
        else:
            points = np.vstack([data['x'], data['y'], data['z']]).T

        return points
    
    def _aug_data(self, points, aug_size, uniform_rotate_only=False):
        """
        Object segments data augmentation, translation as well as rotation refer to VoxelNet
        :param points:
        :param aug_size: creating n copies of each input instance, n is 12 or 18 refer VoxNet.
        :param uniform_rotate_only: just uniform rotate by "2π/aug_size"
        :return:
        """
        np.random.seed()
        rot_interval = 2 * np.pi / (aug_size+1)

        points_list = [points]
        for idx in range(1, aug_size+1):
            # rotate by a uniformally distributed random variable
            r_z = np.random.uniform(-np.pi / 10, np.pi / 10)
            t_x = np.random.normal()
            t_y = np.random.normal()
            t_z = np.random.normal()

            if uniform_rotate_only:
                # creating n copies of each input instance, each rotated 360◦/n intervals around the z axis.
                r_z = rot_interval * idx
                t_x = t_y = t_z = 0.

            # translation and rotation
            points_list.append(self._point_transform(points, t_x, t_y, t_z, rz=r_z))

        return np.float32(points_list)

    def _point_transform(self, points, tx, ty, tz, rx=0, ry=0, rz=0):
        """
        P(x, y, z) transform operation with translation(tx, ty, tz) and rotation(rx, ry, rz)
        :param original points: (N, 3)
        :param tx/y/z: in meter
        :param rx/y/z: in radian
        :return: transformed points: (N, 3)
        """

        N = points.shape[0]
        points = np.hstack([points, np.ones((N, 1))])

        mat1 = np.eye(4)
        mat1[3, 0:3] = tx, ty, tz
        points = np.matmul(points, mat1)

        if rx != 0:
            mat = np.zeros((4, 4))
            mat[0, 0] = 1
            mat[3, 3] = 1
            mat[1, 1] = np.cos(rx)
            mat[1, 2] = -np.sin(rx)
            mat[2, 1] = np.sin(rx)
            mat[2, 2] = np.cos(rx)
            points = np.matmul(points, mat)

        if ry != 0:
            mat = np.zeros((4, 4))
            mat[1, 1] = 1
            mat[3, 3] = 1
            mat[0, 0] = np.cos(ry)
            mat[0, 2] = np.sin(ry)
            mat[2, 0] = -np.sin(ry)
            mat[2, 2] = np.cos(ry)
            points = np.matmul(points, mat)

        if rz != 0:
            mat = np.zeros((4, 4))
            mat[2, 2] = 1
            mat[3, 3] = 1
            mat[0, 0] = np.cos(rz)
            mat[0, 1] = -np.sin(rz)
            mat[1, 0] = np.sin(rz)
            mat[1, 1] = np.cos(rz)
            points = np.matmul(points, mat)

        return points[:, 0:3]
    
    def _voxelize(self, points, voxel_size=(24, 24, 24), padding_size=(32, 32, 32), resolution=0.1):
        """
        Convert `points` to centerlized voxel with size `voxel_size` and `resolution`, then padding zero to
        `padding_to_size`. The outside part is cut, rather than scaling the points.

        Args:
        `points`: pointcloud in 3D numpy.ndarray
        `voxel_size`: the centerlized voxel size, default (24,24,24)
        `padding_to_size`: the size after zero-padding, default (32,32,32)
        `resolution`: the resolution of voxel, in meters

        Ret:
        `voxel`:32*32*32 voxel occupany grid
        `inside_box_points`:pointcloud inside voxel grid
        """

        if abs(resolution) < sys.float_info.epsilon:
            print('error input, resolution should not be zero')
            return None, None

        # remove all non-numeric elements of the said array
        points = points[np.logical_not(np.isnan(points).any(axis=1))]

        # filter outside voxel_box by using passthrough filter
        # TODO Origin, better use centroid?
        origin = (np.min(points[:, 0]), np.min(points[:, 1]), np.min(points[:, 2]))
        # set the nearest point as (0,0,0)
        points[:, 0] -= origin[0]
        points[:, 1] -= origin[1]
        points[:, 2] -= origin[2]
        # logical condition index
        x_logical = np.logical_and((points[:, 0] < voxel_size[0] * resolution), (points[:, 0] >= 0))
        y_logical = np.logical_and((points[:, 1] < voxel_size[1] * resolution), (points[:, 1] >= 0))
        z_logical = np.logical_and((points[:, 2] < voxel_size[2] * resolution), (points[:, 2] >= 0))
        xyz_logical = np.logical_and(x_logical, np.logical_and(y_logical, z_logical))
        inside_box_points = points[xyz_logical]

        # init voxel grid with zero padding_to_size=(32*32*32) and set the occupany grid
        voxels = np.zeros(padding_size)
        # centerlize to padding box
        center_points = inside_box_points + (padding_size[0] - voxel_size[0]) * resolution / 2  # 平移居中到新的体素网格中
        # TODO currently just use the binary hit grid
        x_idx = (center_points[:, 0] / resolution).astype(int)
        y_idx = (center_points[:, 1] / resolution).astype(int)
        z_idx = (center_points[:, 2] / resolution).astype(int)
        OCCUPIED = 1
        voxels[x_idx, y_idx, z_idx] = OCCUPIED

        return voxels, inside_box_points
        
