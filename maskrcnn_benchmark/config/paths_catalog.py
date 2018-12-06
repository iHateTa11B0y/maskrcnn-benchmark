# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""Centralized catalog of paths."""

import os


class DatasetCatalog(object):
    DATA_DIR = ""

    DATASETS = {
        "coco_2014_train": (
            "coco/train2014",
            "coco/annotations/instances_train2014.json",
        ),
        "coco_2014_val": ("coco/val2014", "coco/annotations/instances_val2014.json"),
        "coco_2014_minival": (
            "coco/val2014",
            "coco/annotations/instances_minival2014.json",
        ),
        "coco_2014_valminusminival": (
            "coco/val2014",
            "coco/annotations/instances_valminusminival2014.json",
        ),
        "coco_gbox_soft_package_train": (
            "",
            "/core1/data/home/niuwenhao/soft_package/segs/coco_gbox_soft_package_train.json",
        ),
        "coco_gbox_soft_package_test": (
            "",
            "/core1/data/home/niuwenhao/soft_package/segs/coco_gbox_soft_package_test.json",
        ),
        "coco_gbox_soft_package_ordered_train": (
            "",
            "/core1/data/home/niuwenhao/soft_package/segs/coco_gbox_soft_package_ordered_train.json",
        ),
        "coco_gbox_soft_package_ordered_test": (
            "",
            "/core1/data/home/niuwenhao/soft_package/segs/coco_gbox_soft_package_ordered_test.json",
        ),
        "coco_brain_hole_v1": (
            "",
            "/core1/data/home/liuhuawei/data-manager/data/cx_brain_hole/coco_cx_brain_hole_v1_train_new.json",
        ),
        "coco_brain_hole_v1_val": (
            "",
            "/core1/data/home/liuhuawei/data-manager/data/cx_brain_hole/coco_cx_brain_hole_v1_test_new.json",
        ),
        "coco_all_detection": (
            "",
            "/core1/data/home/liuhuawei/data-manager/data/all_detection/coco_all_detection_train_new.json",
        ),
        "coco_all_detection_etra": (
            "",
            "/core1/data/home/liuhuawei/data-manager/data/all_detection/coco_all_detection_extra_train_new.json",
        ),
        "coco_iuu_all_train_new": (
            "",
            "/core1/data/home/liuhuawei/data-manager/data/all_detection/iuu_all_train_new.json",
        ),
        "coco_all_detection_extra_2": (
            "",
            "/core1/data/home/liuhuawei/data-manager/data/all_detection/coco_all_detection_extra_2_train_new.json",
        ),
        
        "coco_mryx_testset_val": (
            "",
            "/core1/data/home/liuhuawei/data-manager/data/testset/coco_mryx_testset_test_new.json",
        ),
        "coco_iuu_all_val": (
            "",
            "/core1/data/home/liuhuawei/data-manager/data/0724/iuu_all_test_new_clean.json",
        ),
        "coco_midea_testset_val": (
            "",
            "/core1/data/home/liuhuawei/data-manager/data/testset/coco_midea_testset_test_new.json",
        ),
        "coco_gbox_testset_val": (
            "",
            "/core1/data/home/liuhuawei/data-manager/data/testset/coco_gbox_testset_test_new.json",
        ),
        "coco_infer_val": (
            "",
            "/data/wenhao/data/seg/utils/test.json",
        ),
        
    }

    @staticmethod
    def get(name):
        if "coco" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs[0]),
                ann_file=os.path.join(data_dir, attrs[1]),
            )
            return dict(
                factory="COCODataset",
                args=args,
            )
        raise RuntimeError("Dataset not available: {}".format(name))


class ModelCatalog(object):
    S3_C2_DETECTRON_URL = "https://s3-us-west-2.amazonaws.com/detectron"
    C2_IMAGENET_MODELS = {
        "MSRA/R-50": "ImageNetPretrained/MSRA/R-50.pkl",
        "MSRA/R-101": "ImageNetPretrained/MSRA/R-101.pkl",
        "FAIR/20171220/X-101-32x8d": "ImageNetPretrained/20171220/X-101-32x8d.pkl",
    }

    C2_DETECTRON_SUFFIX = "output/train/coco_2014_train%3Acoco_2014_valminusminival/generalized_rcnn/model_final.pkl"
    C2_DETECTRON_MODELS = {
        "35857197/e2e_faster_rcnn_R-50-C4_1x": "01_33_49.iAX0mXvW",
        "35857345/e2e_faster_rcnn_R-50-FPN_1x": "01_36_30.cUF7QR7I",
        "35857890/e2e_faster_rcnn_R-101-FPN_1x": "01_38_50.sNxI7sX7",
        "36761737/e2e_faster_rcnn_X-101-32x8d-FPN_1x": "06_31_39.5MIHi1fZ",
        "35858791/e2e_mask_rcnn_R-50-C4_1x": "01_45_57.ZgkA7hPB",
        "35858933/e2e_mask_rcnn_R-50-FPN_1x": "01_48_14.DzEQe4wC",
        "35861795/e2e_mask_rcnn_R-101-FPN_1x": "02_31_37.KqyEK4tT",
        "36761843/e2e_mask_rcnn_X-101-32x8d-FPN_1x": "06_35_59.RZotkLKI",
    }

    @staticmethod
    def get(name):
        if name.startswith("Caffe2Detectron/COCO"):
            return ModelCatalog.get_c2_detectron_12_2017_baselines(name)
        if name.startswith("ImageNetPretrained"):
            return ModelCatalog.get_c2_imagenet_pretrained(name)
        raise RuntimeError("model not present in the catalog {}".format(name))

    @staticmethod
    def get_c2_imagenet_pretrained(name):
        prefix = ModelCatalog.S3_C2_DETECTRON_URL
        name = name[len("ImageNetPretrained/"):]
        name = ModelCatalog.C2_IMAGENET_MODELS[name]
        url = "/".join([prefix, name])
        return url

    @staticmethod
    def get_c2_detectron_12_2017_baselines(name):
        # Detectron C2 models are stored following the structure
        # prefix/<model_id>/2012_2017_baselines/<model_name>.yaml.<signature>/suffix
        # we use as identifiers in the catalog Caffe2Detectron/COCO/<model_id>/<model_name>
        prefix = ModelCatalog.S3_C2_DETECTRON_URL
        suffix = ModelCatalog.C2_DETECTRON_SUFFIX
        # remove identification prefix
        name = name[len("Caffe2Detectron/COCO/"):]
        # split in <model_id> and <model_name>
        model_id, model_name = name.split("/")
        # parsing to make it match the url address from the Caffe2 models
        model_name = "{}.yaml".format(model_name)
        signature = ModelCatalog.C2_DETECTRON_MODELS[name]
        unique_name = ".".join([model_name, signature])
        url = "/".join([prefix, model_id, "12_2017_baselines", unique_name, suffix])
        return url
