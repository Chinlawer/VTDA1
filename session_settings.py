from datasets.CILdataset import *

def shapenet2co3d():
    session_maker = SessionMaker()
    shapenet_train = ShapeNetCIL(partition='train')
    shapenet_test = ShapeNetCIL(partition='test')
    shapenet_id2name = ShapeNetCIL.id2name
    co3d = CO3DCIL(banlist=['car', 'microwave', 'bowl', 'bottle', 'skateboard', 'bench', 'motorcycle', 'laptop', 'chair'])
    co3d_id2name = co3d.get_label2name()
    session_maker.append_dataset_train_test(shapenet_train, shapenet_test, shapenet_id2name)
    session_maker.append_dataset(co3d, co3d_id2name, split_ratio=0.8)
    session_maker.set_session(num_base_cat=55, num_inc_cat=4)
    return session_maker
    
def shapenet2co3d_joint():
    session_maker = SessionMaker()
    shapenet_train = ShapeNetCIL(partition='train')
    shapenet_test = ShapeNetCIL(partition='test')
    shapenet_id2name = ShapeNetCIL.id2name
    co3d = CO3DCIL(banlist=['car', 'microwave', 'bowl', 'bottle', 'skateboard', 'bench', 'motorcycle', 'laptop', 'chair'])
    co3d_id2name = co3d.get_label2name()
    session_maker.append_dataset_train_test(shapenet_train, shapenet_test, shapenet_id2name)
    session_maker.append_dataset(co3d, co3d_id2name, split_ratio=0.8)
    session_maker.set_session_joint(num_base_cat=55, num_inc_cat=4)
    return session_maker

def shapenet2scanobjectnn():
    session_maker = SessionMaker()
    shapenet_train = ShapeNetCIL(partition='train')
    shapenet_test = ShapeNetCIL(partition='test')
    shapenet_id2name = ShapeNetCIL.id2name
    # banlist=['car', 'microwave', 'bowl', 'bottle', 'skateboard', 'bench', 'motorcycle', 'laptop', 'chair']
    scanobjectnn_train = ScanObjectNN(partition='training', num_points=1024)
    scanobjectnn_test = ScanObjectNN(partition='test', num_points=1024)
    scanobjectnn_id2name = ScanObjectNN.id2name
    session_maker.append_dataset_train_test(shapenet_train, shapenet_test, shapenet_id2name)
    session_maker.append_dataset_train_test(scanobjectnn_train, scanobjectnn_test, scanobjectnn_id2name)
    session_maker.set_session(num_base_cat=55, num_inc_cat=4)
    return session_maker

def shapenet2sydney():
    session_maker = SessionMaker()
    shapenet_train = ShapeNetCIL(partition='train')
    shapenet_test = ShapeNetCIL(partition='test')
    shapenet_id2name = ShapeNetCIL.id2name
    # banlist=['car', 'microwave', 'bowl', 'bottle', 'skateboard', 'bench', 'motorcycle', 'laptop', 'chair']
    sydney_train = SydneyUrbanObjects(partition='train')
    sydney_test = SydneyUrbanObjects(partition='test')
    sydney_test_id2name = SydneyUrbanObjects.id2name
    session_maker.append_dataset_train_test(shapenet_train, shapenet_test, shapenet_id2name)
    session_maker.append_dataset_train_test(sydney_train, sydney_test, sydney_test_id2name)
    session_maker.set_session(num_base_cat=55, num_inc_cat=4)
    return session_maker

def shapenet2modelnet():        
    """
    load .npy(numpy) and .off mesh(voxel) documents网格/体素文件
    not point clouds, voxel can be sampled to points doc
    doubt here to project to maps.
    """
    session_maker = SessionMaker()
    shapenet_train = ShapeNetCIL(partition='train')
    shapenet_test = ShapeNetCIL(partition='test')
    shapenet_id2name = ShapeNetCIL.id2name
    modelnet_banlist = ['airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car', 'chair', 'guitar', 'lamp', 'laptop', 'piano', 'sofa', 'table', 'keyboard']
    # modelnet_banlist = []
    modelnet_train = ModelNet40AlignCIL(partition='train', banlist=modelnet_banlist)
    modelnet_test = ModelNet40AlignCIL(partition='test', banlist=modelnet_banlist)
    modelnet_id2name = ModelNet40AlignCIL.id2name
    session_maker.append_dataset_train_test(shapenet_train, shapenet_test, shapenet_id2name)
    session_maker.append_dataset_train_test(modelnet_train, modelnet_test, modelnet_id2name, modelnet_train.get_load_method())
    # .get_load_method()  used to sample points from mesh
    session_maker.set_session(num_base_cat=55, num_inc_cat=4)
    return session_maker

def shapenet2modelnet2scan():        
    session_maker = SessionMaker()
    shapenet_train = ShapeNetCIL(partition='train')
    shapenet_test = ShapeNetCIL(partition='test')
    shapenet_id2name = ShapeNetCIL.id2name
    modelnet_banlist = ['airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car', 'chair', 'guitar', 'lamp', 'laptop', 'piano', 'sofa', 'table', 'keyboard']
    # modelnet_banlist = []
    modelnet_train = ModelNet40AlignCIL(partition='train', banlist=modelnet_banlist)
    modelnet_test = ModelNet40AlignCIL(partition='test', banlist=modelnet_banlist)
    modelnet_id2name = ModelNet40AlignCIL.id2name
    scanobjectnn_train = ScanObjectNN(partition='training', num_points=1024)
    scanobjectnn_test = ScanObjectNN(partition='test', num_points=1024)
    scanobjectnn_id2name = ScanObjectNN.id2name
    session_maker.append_dataset_train_test(shapenet_train, shapenet_test, shapenet_id2name)
    session_maker.append_dataset_train_test(modelnet_train, modelnet_test, modelnet_id2name, modelnet_train.get_load_method())
    session_maker.append_dataset_train_test(scanobjectnn_train, scanobjectnn_test, scanobjectnn_id2name)
    # .get_load_method()  used to sample points from mesh
    session_maker.set_session(num_base_cat=55, num_inc_cat=4)
    return session_maker

def shapenet2modelnet2sydney():        
    session_maker = SessionMaker()
    shapenet_train = ShapeNetCIL(partition='train')
    shapenet_test = ShapeNetCIL(partition='test')
    shapenet_id2name = ShapeNetCIL.id2name
    modelnet_banlist = ['airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car', 'chair', 'guitar', 'lamp', 'laptop', 'piano', 'sofa', 'table', 'keyboard']
    # modelnet_banlist = []
    modelnet_train = ModelNet40AlignCIL(partition='train', banlist=modelnet_banlist)
    modelnet_test = ModelNet40AlignCIL(partition='test', banlist=modelnet_banlist)
    modelnet_id2name = ModelNet40AlignCIL.id2name
    sydney_train = SydneyUrbanObjects(partition='train')
    sydney_test = SydneyUrbanObjects(partition='test')
    sydney_test_id2name = SydneyUrbanObjects.id2name
    session_maker.append_dataset_train_test(shapenet_train, shapenet_test, shapenet_id2name)
    session_maker.append_dataset_train_test(modelnet_train, modelnet_test, modelnet_id2name, modelnet_train.get_load_method())
    session_maker.append_dataset_train_test(sydney_train, sydney_test, sydney_test_id2name)
    session_maker.set_session(num_base_cat=55, num_inc_cat=4)
    return session_maker

def shapenet2scan2sydney():        
    session_maker = SessionMaker()
    shapenet_train = ShapeNetCIL(partition='train')
    shapenet_test = ShapeNetCIL(partition='test')
    shapenet_id2name = ShapeNetCIL.id2name
    scanobjectnn_train = ScanObjectNN(partition='training', num_points=1024)
    scanobjectnn_test = ScanObjectNN(partition='test', num_points=1024)
    scanobjectnn_id2name = ScanObjectNN.id2name

    sydney_train = SydneyUrbanObjects(partition='train')
    sydney_test = SydneyUrbanObjects(partition='test')
    sydney_test_id2name = SydneyUrbanObjects.id2name
    
    session_maker.append_dataset_train_test(shapenet_train, shapenet_test, shapenet_id2name)
    session_maker.append_dataset_train_test(scanobjectnn_train, scanobjectnn_test, scanobjectnn_id2name)
    session_maker.append_dataset_train_test(sydney_train, sydney_test, sydney_test_id2name)
    session_maker.set_session(num_base_cat=55, num_inc_cat=4)
    return session_maker

def shapenet2sydney2scan():        
    session_maker = SessionMaker()
    shapenet_train = ShapeNetCIL(partition='train')
    shapenet_test = ShapeNetCIL(partition='test')
    shapenet_id2name = ShapeNetCIL.id2name
    sydney_train = SydneyUrbanObjects(partition='train')
    sydney_test = SydneyUrbanObjects(partition='test')
    sydney_test_id2name = SydneyUrbanObjects.id2name
    
    scanobjectnn_train = ScanObjectNN(partition='training', num_points=1024)
    scanobjectnn_test = ScanObjectNN(partition='test', num_points=1024)
    scanobjectnn_id2name = ScanObjectNN.id2name
    
    session_maker.append_dataset_train_test(shapenet_train, shapenet_test, shapenet_id2name)
    
    session_maker.append_dataset_train_test(sydney_train, sydney_test, sydney_test_id2name)
    session_maker.append_dataset_train_test(scanobjectnn_train, scanobjectnn_test, scanobjectnn_id2name)

    session_maker.set_session(num_base_cat=55, num_inc_cat=4)
    return session_maker


def shapenet2model2scan2sydney():        
    session_maker = SessionMaker()
    shapenet_train = ShapeNetCIL(partition='train')
    shapenet_test = ShapeNetCIL(partition='test')
    shapenet_id2name = ShapeNetCIL.id2name
    modelnet_banlist = ['airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car', 'chair', 'guitar', 'lamp', 'laptop', 'piano', 'sofa', 'table', 'keyboard']
    # modelnet_banlist = []
    modelnet_train = ModelNet40AlignCIL(partition='train', banlist=modelnet_banlist)
    modelnet_test = ModelNet40AlignCIL(partition='test', banlist=modelnet_banlist)
    modelnet_id2name = ModelNet40AlignCIL.id2name

    scanobjectnn_train = ScanObjectNN(partition='training', num_points=1024)
    scanobjectnn_test = ScanObjectNN(partition='test', num_points=1024)
    scanobjectnn_id2name = ScanObjectNN.id2name

    sydney_train = SydneyUrbanObjects(partition='train')
    sydney_test = SydneyUrbanObjects(partition='test')
    sydney_test_id2name = SydneyUrbanObjects.id2name
    
    session_maker.append_dataset_train_test(shapenet_train, shapenet_test, shapenet_id2name)
    session_maker.append_dataset_train_test(modelnet_train, modelnet_test, modelnet_id2name, modelnet_train.get_load_method())
    session_maker.append_dataset_train_test(scanobjectnn_train, scanobjectnn_test, scanobjectnn_id2name)
    session_maker.append_dataset_train_test(sydney_train, sydney_test, sydney_test_id2name)

    # .get_load_method()  used to sample points from mesh
    session_maker.set_session(num_base_cat=55, num_inc_cat=4)
    return session_maker

def null2modelnet():
    session_maker = SessionMaker()
    modelnet_banlist = ['airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car', 'chair', 'guitar', 'lamp', 'laptop', 'piano', 'sofa', 'table', 'keyboard']
    modelnet_train = ModelNet40AlignCIL(partition='train', banlist=modelnet_banlist)
    modelnet_test = ModelNet40AlignCIL(partition='test', banlist=modelnet_banlist)
    modelnet_id2name = ModelNet40AlignCIL.id2name
    session_maker.append_dataset_train_test(modelnet_train, modelnet_test, modelnet_id2name, modelnet_train.get_load_method())
    session_maker.set_session(num_base_cat=0, num_inc_cat=4)
    return session_maker

def save_settings(maker):
    for t in range(0, maker.tot_session()):
        train, test = maker.make_session(t, update_memory=0)
        d1 = './data/index_files/shapenet_modelnet/'+str(t)
        d2 = './data/index_files/shapenet_modelnet/'+str(t)
        if not os.path.exists(d1):
            os.makedirs(d1)
        if not os.path.exists(d2):
            os.makedirs(d2)
        train.save(d1+'/train.txt')
        test.save(d2+'/test.txt')

if __name__ == '__main__':
    #################maker = shapenet2co3d()
    maker = shapenet2modelnet()
    #print(maker.info())
    #save_settings(maker)
    train, test = maker.make_session(maker.tot_session()-1, update_memory=0)
    for data in train:
        pass
    for data in test:
        pass
    """
    for t in range(0, maker.tot_session()):
        train, test = maker.make_session(t, update_memory=0)
        d = './data/index_files/check/'+ str(t)
        print(f't:{t}')
        print('train:')
        train.check(d+'/train.txt')
        print('test:')
        test.check(d+'/test.txt')
        
    """
    