from builtins import breakpoint
import os
import torch
import torch.nn.functional as F
import torchvision.transforms.functional_tensor as F_t
import torchvision.transforms.functional as TF
from torchvision import transforms
import glob
import imageio
import numpy as np
import random
from .util import detect_black, get_image_to_tensor_balanced, load_pfm


class ColorJitter(object):
    """
    Color jitter module for argumentation.
    """
    def __init__(
        self,
        hue_range=0.15,
        saturation_range=0.15,
        brightness_range=0.15,
        contrast_range=0.15,
    ):
        self.hue_range = [-hue_range, hue_range]
        self.saturation_range = [1 - saturation_range, 1 + saturation_range]
        self.brightness_range = [1 - brightness_range, 1 + brightness_range]
        self.contrast_range = [1 - contrast_range, 1 + contrast_range]

    def apply_color_jitter(self, images):
        # apply the same color jitter over batch of images
        hue_factor = np.random.uniform(*self.hue_range)
        saturation_factor = np.random.uniform(*self.saturation_range)
        brightness_factor = np.random.uniform(*self.brightness_range)
        contrast_factor = np.random.uniform(*self.contrast_range)
        for i in range(len(images)):
            tmp = (images[i] + 1.0) * 0.5
            tmp = F_t.adjust_saturation(tmp, saturation_factor)
            tmp = F_t.adjust_hue(tmp, hue_factor)
            tmp = F_t.adjust_contrast(tmp, contrast_factor)
            tmp = F_t.adjust_brightness(tmp, brightness_factor)
            images[i] = tmp * 2.0 - 1.0
        return images

class DTU_Dataset(torch.utils.data.Dataset):
    """
    Dataset for DTU MVS dataset.
    """
    def __init__(
        self,
        stage="train",
        opts=None
    ):
        super().__init__()

        path = opts.path
        list_prefix = opts.list_prefix
        self.input_view_num = opts.input_view_num
        self.num_views = opts.num_views
        self.input_suv_ratio = opts.input_suv_ratio
        self.base_path = path
        self.stage = stage
        self.image_size = opts.image_size
        assert os.path.exists(self.base_path)
        
        all_objs  = []
        self.train_path = os.path.join(path, list_prefix + "train.lst")
        self.val_path = os.path.join(path, list_prefix + "val.lst")
        self.test_path = os.path.join(path, list_prefix + "test.lst")
        if stage == "train":
            self.train = True
            file = os.path.join(path, list_prefix + "train.lst")
        elif stage == "val":
            self.train = False
            file = os.path.join(path, list_prefix + "val.lst")
        elif stage == "test": # there are only  training and validataion dataset in this dataset.
            self.train = False
            file = os.path.join(path, list_prefix + "val.lst")
        with open(file, "r") as f:
            objs = [ x.strip() for x in f.readlines()]
        f.close()
        all_objs.extend(objs)
        self.all_objs = all_objs

        self.image_to_tensor = get_image_to_tensor_balanced()
        print(
            "Loading DVR dataset",
            self.base_path,
            "stage",
            stage,
            len(self.all_objs),
            "objs",
        )

        self._coord_trans_world = torch.tensor(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            dtype=torch.float32,
        )
        self._coord_trans_cam = torch.tensor(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            dtype=torch.float32,
        )

        self.scale_factor = opts.scale_factor
        self.max_imgs = opts.max_imgs

        self.z_near = opts.z_near
        self.z_far = opts.z_far
        self.lindisp = False
        self.use_depth = opts.use_gt_depth
        self.camera_path = opts.camera_path
        self.depth_path = opts.depth_path
        self.camera = np.load(self.camera_path, allow_pickle=True)

        self.suffix = opts.suffix
        self.colorjitter = ColorJitter()
        if self.stage == "test":
            self.test_view = list(map(int,opts.test_view.split()))

    def __len__(self):
        return len(self.all_objs)

    def __getitem__(self, index):
        scan_index = self.all_objs[index]
        ## randomly sample one light conditions
        suffix = random.choice(self.suffix)

        ## read RGB image
        root_dir = os.path.join(self.base_path, "Rectified", scan_index)

        rgb_paths = [
            x
            for x in glob.glob(os.path.join(self.base_path, "Rectified", scan_index, 'image', "*"))
            if (x.endswith( suffix + ".jpg") or x.endswith(suffix + ".png"))
        ]

        if self.use_depth is True:
            depth_path = os.path.join(self.depth_path, scan_index, '*')

            depth_paths = [
                x
                for x in glob.glob(depth_path)
                if x.endswith(".pfm")
            ]

            depth_paths = sorted(depth_paths)
        rgb_paths = sorted(rgb_paths)

        if self.stage == "test":
            ## for test, we use the test_view as inputs and generate all the other views.
            all_indices = list(np.arange(len(rgb_paths)))
            sel_indices = []
            for i in self.test_view:
                sel_indices.append(i)
            all_indices = [x for x in all_indices if x not in sel_indices]
            sel_indices = sel_indices + (all_indices)
        else:
            ## We randomly select num_views images, whhile the first input_view_num image for input.
            sample_num = min(self.num_views, self.max_imgs)
            sel_indices = torch.randperm(len(rgb_paths))[0:sample_num]
            if self.input_suv_ratio > 0.0:
                pros = np.random.uniform(0, 1, min(self.input_view_num, sample_num - self.input_view_num))
                for i in range(pros.shape[0]):
                    if pros[i] <= self.input_suv_ratio:
                        sel_indices[self.input_view_num+i] = sel_indices[i]
        rgb_paths = [rgb_paths[i] for i in sel_indices]

        all_imgs = []
        all_poses = []
        all_poses_inverse = []
        all_depths = []
        focal = None
        depth_max = 0
        fx, fy, cx, cy = 0.0, 0.0, 0.0, 0.0

        for idx, rgb_path in enumerate(rgb_paths):
            i = sel_indices[idx]
            img = imageio.imread(rgb_path)[..., :3]
            H, W = img.shape[0], img.shape[1]

            if self.use_depth:
                depth, scale = load_pfm(depth_paths[i])
                depth_tensor = torch.tensor(depth.copy())
                all_depths.append(depth_tensor)
                if depth_tensor.max() > depth_max:
                    depth_max = depth_tensor.max()

            R = self.camera[i][0][:3, :3]
            T = self.camera[i][0][:3, 3]
            T = T/self.scale_factor ## we scale the word coordinate.
            K = self.camera[i][1]

            pose = np.eye(4, dtype=np.float32)
            pose[:3, :3] = R

            pose[:3, 3] = T
            pose = (
                self._coord_trans_world
                @ torch.tensor(pose, dtype=torch.float32)
                @ self._coord_trans_cam
            )

            fx += K[0, 0]
            fy += K[1, 1]
            cx += K[0, 2]
            cy += K[1, 2]
            img_tensor = self.image_to_tensor(img)
            pose_inverse = torch.inverse(pose)
            all_imgs += [img_tensor]
            all_poses += [pose]
            all_poses_inverse += [pose_inverse]

        if self.use_depth:
            all_depths = [depth/self.scale_factor for depth in all_depths]

        fx /= len(rgb_paths)
        fy /= len(rgb_paths)
        cx /= len(rgb_paths)
        cy /= len(rgb_paths)

        focal = torch.tensor((fx, fy), dtype=torch.float32)
        c = torch.tensor((cx, cy), dtype=torch.float32)
        all_imgs = torch.stack(all_imgs)

        K = torch.zeros((4,4) , dtype=torch.float32)
        K[0, 0] = fx
        K[1, 1] = fy
        K[0, 2] = cx
        K[1, 2] = cy
        K[2, 2] = 1.0
        K[3, 3] = 1.0
        inverse_K = torch.inverse(K)

        if self.image_size is not None and all_imgs.shape[-2:] != self.image_size:
            scale = self.image_size[0] / all_imgs.shape[-2]
            K[0:2, 0:3] *= scale
            all_imgs = F.interpolate(all_imgs, size=self.image_size, mode="area")
        mask = ~detect_black(all_imgs,-0.95)

        if self.use_depth:
            H, W = all_imgs.shape[-2:]
            all_depths = torch.stack(all_depths, dim=0)
            all_depths = F.interpolate(all_depths.unsqueeze(0), size=all_imgs.shape[-2:], mode="nearest")
            all_depths = all_depths[0]
            all_depths = [all_depths[i].unsqueeze(0) for i in range(all_depths.shape[0])]
        
        if self.train:
            all_imgs = self.colorjitter.apply_color_jitter(all_imgs)
        all_imgs = [all_imgs[i] for i in range(all_imgs.shape[0])]
        all_mask = [mask[i] for i in range(mask.shape[0])]
        cameras = []
        for i in range(len(all_imgs)):
            cameras += [
                {
                    'focal': focal,
                    'P': all_poses[i],
                    'Pinv': all_poses_inverse[i],
                    'K': K,
                    'Kinv': inverse_K,
                    'c': c
                }
            ]
        if self.use_depth:
            result = {
                "path": root_dir,
                "img_id": index,
                "images": all_imgs,
                "masks": all_mask,
                "cameras": cameras,
                "depths": all_depths
            }
        else:
            result = {
                "path": root_dir,
                "img_id": index,
                "images": all_imgs,
                "masks": all_mask,
                "cameras": cameras,
            }
        return result

    def totrain(self, epoch=0):
        self.train = True
        all_objs  = []
        with open(self.train_path, "r") as f:
            objs = [x.strip() for x in f.readlines()]
        f.close()
        all_objs.extend(objs)
        self.all_objs = all_objs
        self.stage = "train"

    def toval(self, epoch=0):
        self.train = False
        all_objs  = []
        with open(self.val_path, "r") as f:
            objs = [x.strip() for x in f.readlines()]
        f.close()
        all_objs.extend(objs)
        self.all_objs = all_objs
        self.stage = "val"

    def totest(self, epoch=0):
        self.train = False
        all_objs  = []
        with open(self.val_path, "r") as f:
            objs = [x.strip() for x in f.readlines()]
        f.close()
        all_objs.extend(objs)
        self.all_objs = all_objs
        self.stage = "test"
