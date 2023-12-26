# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it
# under the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import json
import os
import os.path as osp
from collections import defaultdict

import cv2
import numpy as np
import torch
from constants import AUGMENTED_VERTICES_INDEX_DICT
from inference import Simple3DMeshInferencer, detect_all_persons
from omegaconf import OmegaConf
from smplx import build_layer
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from virtualmarker.core.config import cfg, init_experiment_dir, update_config
from virtualmarker.utils.coord_utils import pixel2cam
from virtualpose.core.config import config as det_cfg
from virtualpose.core.config import update_config as det_update_config
from xtcocotools.coco import COCO

from mmpose.evaluation.metrics.infinity_metric import InfinityAnatomicalMetric

CKPT_PATH = "data/ckpt/hr48-PA43.0_MJE69.0_MVE81.2_3dpw.pt"
BACKBONE = "hr48"
BATCH_SIZE = 10000

used_data_keys=[
        "nose",
        "left_eye",
        "right_eye",
        "left_ear",
        "right_ear",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
        "sternum",
        "rshoulder",
        "lshoulder",
        "r_lelbow",
        "l_lelbow",
        "r_melbow",
        "l_melbow",
        "r_lwrist",
        "l_lwrist",
        "r_mwrist",
        "l_mwrist",
        "r_ASIS",
        "l_ASIS",
        "r_PSIS",
        "l_PSIS",
        "r_knee",
        "l_knee",
        "r_mknee",
        "l_mknee",
        "r_ankle",
        "l_ankle",
        "r_mankle",
        "l_mankle",
        "r_5meta",
        "l_5meta",
        "r_toe",
        "l_toe",
        "r_big_toe",
        "l_big_toe",
        "l_calc",
        "r_calc",
        "C7",
        "L2",
        "T11",
        "T6",
    ]

AUGMENTED_VERTICES_INDEX_DICT = {
    key: value for key, value in AUGMENTED_VERTICES_INDEX_DICT.items() if key in used_data_keys
}

def eval_dataset(root_dir, annotation_path):
    cfg.data_dir = osp.join(".", 'data')
    infinity_metric = InfinityAnatomicalMetric(
        osp.join(root_dir, annotation_path), use_area=False, used_data_keys=used_data_keys
    )
    coco = COCO(osp.join(root_dir, annotation_path))
    # ann_ids = coco.getAnnIds(imgIds=img_id)
    infinity_metric.dataset_meta = {"CLASSES": coco.loadCats(coco.getCatIds())}
    infinity_metric.dataset_meta["num_keypoints"] = len(used_data_keys)
    infinity_metric.dataset_meta["sigmas"] = np.array(
        [
            0.026,
            0.025,
            0.025,
            0.035,
            0.035,
            0.079,
            0.079,
            0.072,
            0.072,
            0.062,
            0.062,
            0.107,
            0.107,
            0.087,
            0.087,
            0.089,
            0.089,
        ] + [0.05 for _ in range(len(used_data_keys) - 17)]
    )
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # exp_cfg, def_matrix, body_model = get_smplx_tools(device)

    print("--------------------------- 3D HPS estimation ---------------------------")

    pose_dataset = PoseDataset(root_dir, annotation_path)
    pose_data_loader = DataLoader(pose_dataset, batch_size=BATCH_SIZE, num_workers=0)
    for batch in tqdm(pose_data_loader):
        # norm_img = batch["norm_img"].to(device).float()
        # center = batch["center"].to(device).float()
        # scale = batch["scale"].to(device).float()
        img_h = batch["img_h"].to(device).float()
        img_w = batch["img_w"].to(device).float()
        # focal_length = batch["focal_length"].to(device).float()
        ann_ids = coco.getAnnIds(imgIds=batch["id"].numpy())
        anns = coco.loadAnns(ann_ids)
        # cx, cy, b = center[:, 0], center[:, 1], scale * 200
        # bbox_info = torch.stack([cx - img_w / 2.0, cy - img_h / 2.0, b], dim=-1)
        # # The constants below are used for normalization, and calculated from H36M data.
        # # It should be fine if you use the plain Equation (5) in the paper.
        # bbox_info[:, :2] = (
        #     bbox_info[:, :2] / focal_length.unsqueeze(-1) * 2.8
        # )  # [-1, 1]
        # bbox_info[:, 2] = (bbox_info[:, 2] - 0.24 * focal_length) / (
        #     0.06 * focal_length
        # )  # [-1, 1]

        # with torch.no_grad():
        #     pred_rotmat, pred_betas, pred_cam_crop = cliff_model(norm_img, bbox_info)

        # # convert the camera parameters from the crop camera to the full camera
        # full_img_shape = torch.stack((img_h, img_w), dim=-1)
        # pred_cam_full = cam_crop2full(
        #     pred_cam_crop, center, scale, full_img_shape, focal_length
        # )
        # pred_output = smpl_model(
        #     betas=pred_betas,
        #     body_pose=pred_rotmat[:, 1:],
        #     global_orient=pred_rotmat[:, [0]],
        #     pose2rot=False,
        #     transl=pred_cam_full,
        # )
        # pred_vertices = pred_output.vertices

        # var_dict = smpl_to_smplx(
        #     pred_vertices,
        #     torch.tensor(
        #         smpl_model.faces.astype(np.int32), dtype=torch.long, device=device
        #     ),
        #     exp_cfg,
        #     body_model,
        #     def_matrix,
        #     device,
        # )
        # detection_all, max_person, valid_frame_idx_all = detect_all_persons(batch["img_path"])
        max_person = 1
        load_path_test = 'experiment/simple3dmesh_train/baseline_mix/final.pth.tar'
        detection_all = np.array([[i, batch["bbox"][0][i], batch["bbox"][1][i], batch["bbox"][2][i], batch["bbox"][3][i], 100, 100, 10000] for i in range(len(batch["bbox"][0]))])

        assert cfg.model.name == 'simple3dmesh', 'check cfg of the model name'
        inferencer = Simple3DMeshInferencer(load_path=load_path_test, img_path_list=batch["img_path"], detection_all=detection_all, max_person=max_person)
        inferencer.model.eval()

        results = defaultdict(list)
        with torch.no_grad():
            for i, meta in enumerate(tqdm(inferencer.demo_dataloader, dynamic_ncols=True)):
                for k, _ in meta.items():
                    meta[k] = meta[k].cuda()

                imgs = meta['img'].cuda()
                inv_trans, intrinsic_param = meta['inv_trans'].cuda(), meta['intrinsic_param'].cuda()
                pose_root = meta['root_cam'].cuda()
                depth_factor = meta['depth_factor'].cuda()

                _, _, _, _, pred_mesh, _, pred_root_xy_img = inferencer.model(imgs, inv_trans, intrinsic_param, pose_root, depth_factor, flip_item=None, flip_mask=None)
                results['pred_mesh'].append(pred_mesh.detach().cpu().numpy())
                results['pose_root'].append(pose_root.detach().cpu().numpy())
                results['pred_root_xy_img'].append(pred_root_xy_img.squeeze(1).squeeze(-1).detach().cpu().numpy())
                results['focal_l'].append(meta['focal_l'].detach().cpu().numpy())
                results['center_pt'].append(meta['center_pt'].detach().cpu().numpy())

        for term in results.keys():
            results[term] = np.concatenate(results[term])

        pred_mesh = results['pred_mesh']  # (N*T, V, 3)
        pred_root_xy_img = results['pred_root_xy_img']  # (N*T, J, 2)
        pose_root = results['pose_root']  # (N*T, 3)
        focal_l = results['focal_l']
        center_pt = results['center_pt']
        # root modification (differenct root definition betwee VM & VirtualPose)
        new_pose_root = []
        for root_xy, root_cam, focal, center in zip(pred_root_xy_img, pose_root, focal_l, center_pt):
            root_img = np.array([root_xy[0], root_xy[1], root_cam[-1]])
            new_root_cam = pixel2cam(root_img[None,:], center, focal)
            new_pose_root.append(new_root_cam)
        pose_root = np.array(new_pose_root)  # (N*T, 1, 3)
        pred_mesh = pred_mesh + pose_root
        data_samples = []

        for i in range(len(batch["img_path"])):
            data_sample = {}
            data_sample["ori_shape"] = (
                torch.stack((img_h, img_w), dim=-1)[0].cpu().numpy()
            )
            data_sample["id"] = int(batch["id"][i].cpu().numpy())
            data_sample["img_id"] = int(batch["id"][i].cpu().numpy())
            data_sample["raw_ann_info"] = anns[i]
            data_sample["gt_instances"] = {}
            img_ori_path = batch["img_path"][i]
            img_ori = cv2.imread(img_ori_path)



            chosen_mask = detection_all[:, 0] == i
            pred_mesh_T = pred_mesh[chosen_mask]  # (N, V, 3)
            focal_T = focal_l[chosen_mask]  # (N, ...)
            center_pt_T = center_pt[chosen_mask]  # (N, ...)
            intrinsic_matrix = torch.tensor(
                [
                    [focal_T[0][0], 0, center_pt_T[0][0]],
                    [0, focal_T[0][1], center_pt_T[0][1]],
                    [0, 0, 1],
                ]
            )
            pred_vertices = pred_mesh_T[0]
            anatomical_vertices = pred_vertices[
                list(AUGMENTED_VERTICES_INDEX_DICT.values())
            ]
            projected_vertices = np.matmul(anatomical_vertices, intrinsic_matrix.cpu().detach().numpy().T)
            projected_vertices = projected_vertices[:, :2] / projected_vertices[:, 2:]
            projected_vertices = projected_vertices[:, :2]
            # print("projected_vertices:", projected_vertices)

            # add dimension to axis 0:
            projected_vertices = np.expand_dims(projected_vertices, axis=0)
            data_sample["pred_instances"] = {}
            data_sample["pred_instances"]["bbox_scores"] = np.ones(
                len(projected_vertices)
            )
            coco_kps = np.zeros((len(projected_vertices), 17, 2))
            keypoints = np.concatenate((coco_kps, projected_vertices), axis=1)
            data_sample["pred_instances"]["keypoints"] = keypoints
            data_sample["pred_instances"]["keypoint_scores"] = np.ones(
                (1, len(projected_vertices[0]) + 17)
            )
            data_sample["raw_ann_info"]["keypoints"] = {
                key: value for key, value in data_sample["raw_ann_info"]["keypoints"].items() if key in used_data_keys
            }
            data_sample["category_id"] = data_sample["raw_ann_info"]["category_id"]
            # print(
            #     "keypoint_scores shape:",
            #     data_sample["pred_instances"]["keypoint_scores"].shape,
            # )
            # render vertices on image and save it
            for x, y in keypoints[0, 17:, :]:
                cv2.circle(img_ori, (int(x), int(y)), 1, (0, 0, 255))
            for name in data_sample["raw_ann_info"]["keypoints"]:
                cv2.circle(
                    img_ori,
                    (
                        int(data_sample["raw_ann_info"]["keypoints"][name]["x"]),
                        int(data_sample["raw_ann_info"]["keypoints"][name]["y"]),
                    ),
                    10,
                    (255, 0, 0),
                )
            filename = osp.basename(img_ori_path).split(".")[0]
            filename = filename + "_vertices_cliff_%s.jpg" % BACKBONE
            # create folder if not exists
            if not osp.exists("eval_test"):
                os.makedirs("eval_test")
            vertices_path = osp.join("eval_test", filename)
            cv2.imwrite(vertices_path, img_ori)
            data_samples.append(data_sample)
        infinity_metric.process([], data_samples)
        torch.cuda.empty_cache()
        del inferencer
        # results = infinity_metric.compute_metrics(infinity_metric.results)
    # print("results:", infinity_metric.results)
    infinity_metric.evaluate(size=len(infinity_metric.results))


class PoseDataset(Dataset):
    def __init__(self, root_dir, annotation_path):
        self.root_dir = root_dir
        self.annotations = json.load(open(osp.join(root_dir, annotation_path), "r"))

    def __len__(self):
        return len(self.annotations["images"])

    def __getitem__(self, idx):
        """
        bbox: [batch_id, min_x, min_y, max_x, max_y, det_conf, nms_conf, category_id]
        :param idx:
        :return:
        """

        item = {}
        img_bgr = cv2.imread(
            osp.join(self.root_dir, self.annotations["images"][idx]["img_path"])
        )
        img_rgb = img_bgr[:, :, ::-1]
        img_h, img_w, _ = img_rgb.shape
        bbox = self.annotations["annotations"][idx]["bbox"]
        bbox[2] = bbox[0] + bbox[2]
        bbox[3] = bbox[1] + bbox[3]
        item["bbox"] = bbox

        item["img_h"] = img_h
        item["img_w"] = img_w
        item["img_path"] = osp.join("../", self.annotations["images"][idx]["img_path"])
        item["id"] = self.annotations["annotations"][idx]["id"]

        item["keypoints"] = self.annotations["annotations"][idx]["keypoints"]
        item["coco_keypoints"] = self.annotations["annotations"][idx]["coco_keypoints"]

        return item

if __name__ == "__main__":
    # eval_dataset("../", "combined_dataset_15fps/test/annotations.json")
    eval_dataset("/scratch/users/yonigoz/RICH/downsampled/", "val_annotations.json")
    # eval_dataset("/scratch/users/yonigoz/BEDLAM/data/", "val_annotations.json")
