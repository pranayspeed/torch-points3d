import os
from glob import glob
import numpy as np
import multiprocessing
import logging
import torch

from torch_geometric.data import Dataset, Data

from torch_points3d.datasets.base_dataset import BaseDataset
#from torch_points3d.datasets.segmentation.kitti_config import *
from torch_points3d.datasets.segmentation import IGNORE_LABEL
from torch_points3d.metrics.segmentation_tracker import SegmentationTracker

from torch_points3d.datasets.segmentation.bagsfit_config import *



log = logging.getLogger(__name__)


class SampledBagsfit(Dataset):
    r"""SemanticKITTI: A Dataset for Semantic Scene Understanding of LiDAR Sequences"
    from the <https://arxiv.org/pdf/1904.01416.pdf> paper, 
    containing about 21 lidar scan sequences with dense point-wise annotations.
    
    root dir should be structured as
    rootdir
        └── sequences/
            ├── 00/
            │   ├── labels/
            │   │     ├ 000000.label
            │   │     └ 000001.label
            │   └── velodyne/
            │         ├ 000000.bin
            │         └ 000001.bin
            ├── 01/
            ├── 02/
            .
            .
            .
            └── 21/
    Args:
        root (string): Root directory of the dataset.
        split (string, optional): If :obj:`"train"`, loads the training
            dataset.
            If :obj:`"val"`, loads the validation dataset.
            If :obj:`"trainval"`, loads the training and validation dataset.
            If :obj:`"test"`, loads the test dataset.
            (default: :obj:`"trainval"`)
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
    """

    LABELS = LABELS
    COLOR_MAP = COLOR_MAP
    #CONTENT = CONTENT
    REMAPPING_MAP = REMAPPING_MAP
    LEARNING_MAP_INV = LEARNING_MAP_INV
    SPLIT = SPLIT
    AVAILABLE_SPLITS = ["train", "val", "test", "trainval"]

    def __init__(self, root, split="trainval", transform=None, process_workers=1, pre_transform=None):
        print("ignore label: ", IGNORE_LABEL)
        print("self.REMAPPING_MAP[0]: ", self.REMAPPING_MAP[0])
        #assert self.REMAPPING_MAP[0] == IGNORE_LABEL  # Make sure we have the same convention for unlabelled data
        self.use_multiprocessing = process_workers > 1
        self.process_workers = process_workers

        super().__init__(root, transform=transform, pre_transform=pre_transform)
        
        if split == "train":
            self._scans = glob(os.path.join(self.processed_paths[0], "*.pt"))
        elif split == "val":
            self._scans = glob(os.path.join(self.processed_paths[1], "*.pt"))
        elif split == "test":
            self._scans = glob(os.path.join(self.processed_paths[2], "*.pt"))
        elif split == "trainval":
            self._scans = glob(os.path.join(self.processed_paths[0], "*.pt")) + glob(
                os.path.join(self.processed_paths[1], "*.pt")
            )
        else:
            raise ValueError("Split %s not recognised" % split)

    @property
    def raw_file_names(self):
        return [""]

    @property
    def processed_file_names(self):
        return [s for s in self.AVAILABLE_SPLITS[:-1]]

    def _load_paths(self, seqs, dir_name):
        scan_paths = []
        label_path = []
        print("self.raw_paths[0]: ", self.raw_paths[0])
        print(self.raw_paths)
        print(seqs)

        if dir_name == "test":
            dir_name_str = "TEST-20s"
        else:
            dir_name_str = "TRAIN-20s"
        print(os.path.join(self.raw_paths[0], dir_name_str))

        for seq in seqs:
            # scan_paths.extend(
            #     sorted(glob(os.path.join(self.raw_paths[0], "{0:02d}".format(int(seq)), "velodyne", "*.bin")))
            # )
            # label_path.extend(
            #     sorted(glob(os.path.join(self.raw_paths[0], "{0:02d}".format(int(seq)), "labels", "*.label")))
            # )
            scan_paths.extend(
                 sorted(glob(os.path.join(self.raw_paths[0], dir_name_str, "{0:03d}".format(int(seq)), "*.npz"))))

        if len(label_path) == 0:
            label_path = [None for i in range(len(scan_paths))]
        if len(label_path) > 0 and len(scan_paths) != len(label_path):
            raise ValueError((f"number of scans {len(scan_paths)} not equal to number of labels {len(label_path)}"))

        return scan_paths, label_path

    # @staticmethod
    # def read_raw(scan_file, label_file=None):
    #     scan = np.fromfile(scan_file, dtype=np.float32).reshape(-1, 4)
    #     data = Data(pos=torch.tensor(scan[:, :3]), x=torch.tensor(scan[:, 3]).reshape(-1, 1),)
    #     if label_file:
    #         label = np.fromfile(label_file, dtype=np.uint32).astype(np.int32)
    #         assert scan.shape[0] == label.shape[0]
    #         semantic_label = label & 0xFFFF
    #         instance_label = label >> 16
    #         data.y = torch.tensor(semantic_label).long()
    #         data.instance_labels = torch.tensor(instance_label).long()
    #     return data

    @staticmethod
    def read_raw(scan_file, label_file=None):
        full_path = scan_file
        geom_data={}
        with np.load(full_path) as data:
            geom_data['data'] = data['data'].reshape(3,-1).T  # with noise
            geom_data['ins'] = np.array(data['ins']).reshape(-1,1).flatten()
            geom_data['cls']= np.array(data['cls']).reshape(-1,1).flatten()
            geom_data['scan'] = data['scan'].reshape(3,-1).T  # without noise
            if 'normals' in data:
                geom_data['normals'] = data['normals'].reshape(3,-1).T

            data = Data(pos=torch.tensor(geom_data['data']), x=torch.ones((len(geom_data['data']), 1)))#torch.tensor(geom_data['data']).reshape(-1, 1),)
            
            data.y = torch.tensor(geom_data['cls']).long()
            data.instance_labels = torch.tensor(geom_data['ins']).long()

        #TBD: fill primitive data as well
        #geom_data['prim'] = get_prim_data(bagsfit_files_path, sequence, test_case)
        return data

    @staticmethod
    def process_one(scan_file, label_file, transform, out_file):


        data = SampledBagsfit.read_raw(scan_file, label_file)
        if transform:
            data = transform(data)
        log.info("Processed file %s, nb points = %i", os.path.basename(out_file), data.pos.shape[0])
        torch.save(data, out_file)

    def get(self, idx):
        data = torch.load(self._scans[idx])
        if data.y is not None:
            data.y = self._remap_labels(data.y)
        return data

    def process(self):
        for i, split in enumerate(self.AVAILABLE_SPLITS[:-1]):
            if os.path.exists(self.processed_paths[i]):
                continue
            os.makedirs(self.processed_paths[i])

            print(split)
            seqs = self.SPLIT[split]
            scan_paths, label_paths = self._load_paths(seqs, split)
            scan_names = []
            for scan in scan_paths:
                scan = os.path.splitext(scan)[0]
                #print(scan)
                _, seq, scan_id = scan.split(os.path.sep)[-3:]
                #print(seq, scan_id)
                scan_names.append("{}_{}".format(seq, scan_id))
                #print(scan_names)
            #exit(0)

            out_files = [os.path.join(self.processed_paths[i], "{}.pt".format(scan_name)) for scan_name in scan_names]
            args = zip(scan_paths, label_paths, [self.pre_transform for i in range(len(scan_paths))], out_files)
            if self.use_multiprocessing:
                with multiprocessing.Pool(processes=self.process_workers) as pool:
                    pool.starmap(self.process_one, args)
            else:
                for arg in args:
                    self.process_one(*arg)

    def len(self):
        return len(self._scans)

    def download(self):
        print(self.raw_dir)
        # if len(os.listdir(self.raw_dir)) == 0:
        #     url = "http://semantic-kitti.org/"
        #     print(f"please download the dataset from {url} with the following folder structure")
        #     print(
        #         """
        #             rootdir
        #                 └── sequences/
        #                     ├── 00/
        #                     │   ├── labels/
        #                     │   │     ├ 000000.label
        #                     │   │     └ 000001.label
        #                     │   └── velodyne/
        #                     │         ├ 000000.bin
        #                     │         └ 000001.bin
        #                     ├── 01/
        #                     ├── 02/
        #                     .
        #                     .
        #                     .
                          
        #                     └── 21/
        #         """
        #     )

    def _remap_labels(self, semantic_label):
        """ Remaps labels to [0 ; num_labels -1]. Can be overriden.
        """
        new_labels = semantic_label.clone()
        for source, target in self.REMAPPING_MAP.items():
            mask = semantic_label == source
            new_labels[mask] = target

        return new_labels

    @property
    def num_classes(self):
        return 5


class BagsfitDataset(BaseDataset):
    """ Wrapper around Semantic Kitti that creates train and test datasets.
    Parameters
    ----------
    dataset_opt: omegaconf.DictConfig
        Config dictionary that should contain
            - root,
            - split,
            - transform,
            - pre_transform
            - process_workers
    """

    def __init__(self, dataset_opt):
        super().__init__(dataset_opt)
        process_workers: int = dataset_opt.process_workers if dataset_opt.process_workers else 0

        print(self._data_path)
        self.train_dataset = SampledBagsfit(
            self._data_path,
            split="train",
            transform=self.train_transform,
            pre_transform=self.pre_transform,
            process_workers=process_workers,
        )

        self.val_dataset = SampledBagsfit(
            self._data_path,
            split="val",
            transform=self.val_transform,
            pre_transform=self.pre_transform,
            process_workers=process_workers,
        )

        self.test_dataset = SampledBagsfit(
            self._data_path,
            split="test",
            transform=self.test_transform,
            pre_transform=self.pre_transform,
            process_workers=process_workers,
        )

    def get_tracker(self, wandb_log: bool, tensorboard_log: bool):
        """Factory method for the tracker
        Arguments:
            wandb_log - Log using weight and biases
            tensorboard_log - Log using tensorboard
        Returns:
            [BaseTracker] -- tracker
        """
        return SegmentationTracker(self, wandb_log=wandb_log, use_tensorboard=tensorboard_log)





#####################UTILS #########################

def get_prim_data(bagsfit_files_path, sequence, test_case):
    full_path_prim = bagsfit_files_path + f"{sequence:03d}" + "/" +f"{test_case:05d}" +".prim"
    prim_data = []
    with open(full_path_prim, "r") as data:        
        for line in data:
            curr_prim = {}
            curr_data = line.split(" ")
            #print(curr_data[0] )
            curr_prim['type'] = curr_data[0] 
            if curr_prim['type'] == "Plane":
                curr_prim['point'] = np.array([float(curr_data[1]), float(curr_data[2]), float(curr_data[3])])
                curr_prim['norm'] = np.array([float(curr_data[4]), float(curr_data[5]), float(curr_data[6])])
            elif curr_prim['type'] == "Sphere":
                curr_prim['point'] = np.array([float(curr_data[1]), float(curr_data[2]), float(curr_data[3])])
                curr_prim['radius'] = float(curr_data[4])
            elif curr_prim['type'] == "Cylinder":
                curr_prim['point'] = np.array([float(curr_data[1]), float(curr_data[2]), float(curr_data[3])])
                curr_prim['axis'] = np.array([float(curr_data[4]), float(curr_data[5]), float(curr_data[6])])
                curr_prim['radius'] = float(curr_data[7])
            elif curr_prim['type'] == "Cone":
                curr_prim['point'] = np.array([float(curr_data[1]), float(curr_data[2]), float(curr_data[3])])
                curr_prim['axis'] = np.array([float(curr_data[4]), float(curr_data[5]), float(curr_data[6])])
                curr_prim['angle'] = float(curr_data[7])
            prim_data.append(curr_prim)
    return prim_data

def get_test_case_data(bagsfit_files_path, sequence, test_case):
    full_path = bagsfit_files_path + f"{sequence:03d}" + "/" +f"{test_case:05d}" +".npz"
    geom_data={}
    with np.load(full_path) as data:
        geom_data['data'] = data['data'].reshape(3,-1).T
        geom_data['ins'] = np.array(data['ins']).reshape(-1,1).flatten()
        geom_data['cls']= np.array(data['cls']).reshape(-1,1).flatten()
        geom_data['scan'] = data['scan'].reshape(3,-1).T
        if 'normals' in data:
            geom_data['normals'] = data['normals'].reshape(3,-1).T

    geom_data['prim'] = get_prim_data(bagsfit_files_path, sequence, test_case)
    return geom_data

import os

def get_all_test_cases(test_cases_path):
    test_cases = []

    for seq_index in range(len(os.listdir(test_cases_path))):
        cases_list = os.listdir(test_cases_path+"/"+ f"{seq_index:03d}")
        subtest_count = sum('.npz' in s for s in cases_list)
        for i in range(subtest_count):
            try:
                #test_cases.append(get_test_case_data(test_cases_path, seq_index, i))
                test_cases.append([test_cases_path, seq_index, i])
            except:
                break
    return test_cases

####################################


import torch
import torch.utils.data as data

class PointCloudDataset(data.Dataset):
    def __init__(self, datapath):

        self.datapath = datapath
        self.test_data_path = get_all_test_cases(datapath)

        #self.points = points
        #self.labels = labels

    def __len__(self):
        return len(self.test_data_path)

    def __getitem__(self, idx):
        #print(self.test_data_path[idx])
        test_data = get_test_case_data(self.test_data_path[idx][0], self.test_data_path[idx][1], self.test_data_path[idx][2])
        #self.test_data[idx]['data'] = torch.from_numpy(self.test_data[idx]['data']).float()
        #print("Data loader ", test_data['data'].shape)
        return torch.from_numpy(test_data['data']).float(), torch.from_numpy(test_data['cls']).float()







if __name__ == "__main__":
    DIR = os.path.dirname(os.path.realpath(__file__))
    #dataroot = os.path.join(DIR, "..", "..", "data", "kitti")
    dataroot = "/home/pranayspeed/Work/git_repo/datasets/Bagsfit/"
    SampledBagsfit(
        dataroot, split="train", process_workers=1,
    )





