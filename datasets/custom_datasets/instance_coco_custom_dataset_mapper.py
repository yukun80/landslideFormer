# ------------------------------------------------------------------------------
# Reference: https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/data/dataset_mappers/coco_instance_new_baseline_dataset_mapper.py
# Modified by Jitesh Jain (https://github.com/praeclarumjj3)
# ------------------------------------------------------------------------------

import copy
import logging

import numpy as np
import torch

from detectron2.data import MetadataCatalog
from detectron2.config import configurable
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from oneformer.data.tokenizer import SimpleTokenizer, Tokenize
from pycocotools import mask as coco_mask

__all__ = ["InstanceCOCOCustomNewBaselineDatasetMapper"]


def convert_coco_poly_to_mask(segmentations, height, width):
    """
    args:
        segmentations: list[list[list[float]]] 多边形分割列表
        height, width: the size of the resulting mask.
    """
    masks = []
    for polygons in segmentations:
        """对于每个多边形分割，函数首先使用coco_mask.frPyObjects方法将其转换为RLE
        （Run-Length Encoding，行程长度编码）格式，
        然后使用coco_mask.decode方法将RLE格式的分割解码为二进制掩码。"""
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        """接下来，函数检查解码后的掩码的维度。
        如果掩码的维度少于3（即，掩码是一个二维数组），函数将在最后一个维度上添加一个新的维度。
        这是因为我们需要一个三维的掩码，其中第一维表示不同的分割，第二和第三维表示图像的高度和宽度。"""
        if len(mask.shape) < 3:
            mask = mask[..., None]
        """然后，函数将掩码转换为torch.Tensor类型，并将其数据类型设置为torch.uint8。
        这是因为我们需要一个由0和1组成的二进制掩码，而torch.uint8类型可以有效地存储这样的数据。"""
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        """函数使用any方法将掩码在第三个维度上进行逻辑或运算，得到一个二维的掩码。
        这是因为在COCO数据集中，一个物体可能有多个分割，我们需要将这些分割合并为一个掩码。"""
        mask = mask.any(dim=2)
        masks.append(mask)
    """如果masks列表不为空，函数使用torch.stack方法将列表中的所有掩码堆叠在一起，得到一个三维的Tensor；
    如果masks列表为空，函数使用torch.zeros方法创建一个形状为(0, height, width)的零Tensor。"""
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


def build_transform_gen(cfg, is_train):
    """
    该函数从配置对象 cfg 中创建一个默认的 Augmentation 列表。
    这个函数主要用于图像的预处理，包括缩放和翻转等操作。
    Returns:
        list[Augmentation]
    """
    assert is_train, "Only support training augmentation"
    image_size = cfg.INPUT.IMAGE_SIZE
    min_scale = cfg.INPUT.MIN_SCALE
    max_scale = cfg.INPUT.MAX_SCALE

    augmentation = []

    if cfg.INPUT.RANDOM_FLIP != "none":
        augmentation.append(
            T.RandomFlip(
                horizontal=cfg.INPUT.RANDOM_FLIP == "horizontal",
                vertical=cfg.INPUT.RANDOM_FLIP == "vertical",
            )
        )

    augmentation.extend(
        [
            T.ResizeScale(min_scale=min_scale, max_scale=max_scale, target_height=image_size, target_width=image_size),
            T.FixedSizeCrop(crop_size=(image_size, image_size)),
        ]
    )

    return augmentation


# This is specifically designed for the COCO Instance Segmentation dataset.
class InstanceCOCOCustomNewBaselineDatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by OneFormer for custom instance segmentation using COCO format.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies geometric transforms to the image and annotation
    3. Find and applies suitable cropping to the image and annotation
    4. Prepare image and annotation to Tensors
    """

    @configurable
    def __init__(
        self,
        is_train=True,
        *,
        num_queries,
        tfm_gens,
        meta,
        image_format,
        max_seq_len,
        task_seq_len,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            is_train: for training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            crop_gen: crop augmentation
            tfm_gens: data augmentation
            image_format: an image format supported by :func:`detection_utils.read_image`.
        """
        self.tfm_gens = tfm_gens
        logging.getLogger(__name__).info(
            "[InstanceCOCOCustomNewBaselineDatasetMapper] Full TransformGens used in training: {}".format(str(self.tfm_gens))
        )

        self.img_format = image_format
        self.is_train = is_train
        self.meta = meta
        self.num_queries = num_queries

        self.things = []
        for k, v in self.meta.thing_dataset_id_to_contiguous_id.items():
            self.things.append(v)
        # 类别
        self.class_names = self.meta.thing_classes
        self.text_tokenizer = Tokenize(SimpleTokenizer(), max_seq_len=max_seq_len)
        self.task_tokenizer = Tokenize(SimpleTokenizer(), max_seq_len=task_seq_len)

    @classmethod
    def from_config(cls, cfg, is_train=True):
        """根据配置文件创建一个实例。构建图像增广操作，并获取数据集元数据。"""
        # Build augmentation
        tfm_gens = build_transform_gen(cfg, is_train)
        dataset_names = cfg.DATASETS.TRAIN
        meta = MetadataCatalog.get(dataset_names[0])

        ret = {
            "is_train": is_train,
            "meta": meta,
            "tfm_gens": tfm_gens,
            "image_format": cfg.INPUT.FORMAT,
            "num_queries": cfg.MODEL.ONE_FORMER.NUM_OBJECT_QUERIES - cfg.MODEL.TEXT_ENCODER.N_CTX,
            "task_seq_len": cfg.INPUT.TASK_SEQ_LEN,
            "max_seq_len": cfg.INPUT.MAX_SEQ_LEN,
        }
        return ret

    # 实例分割任务的描述文本生成函数
    def _get_texts(self, classes, num_class_obj):
        """根据类ID生成描述文本。返回包含描述文本的列表。"""
        classes = list(np.array(classes))
        texts = ["an instance photo"] * self.num_queries

        for class_id in classes:
            cls_name = self.class_names[class_id]
            num_class_obj[cls_name] += 1

        num = 0
        for i, cls_name in enumerate(self.class_names):
            if num_class_obj[cls_name] > 0:
                for _ in range(num_class_obj[cls_name]):
                    if num >= len(texts):
                        break
                    texts[num] = f"a photo with a {cls_name}"
                    num += 1

        return texts

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        # TODO: get padding mask
        # by feeding a "segmentation mask" to the same transforms
        padding_mask = np.ones(image.shape[:2])

        image, transforms = T.apply_transform_gens(self.tfm_gens, image)
        # the crop transformation has default padding value 0 for segmentation
        padding_mask = transforms.apply_segmentation(padding_mask)
        padding_mask = ~padding_mask.astype(bool)
        """转换图像格式：获取图像的形状（高度和宽度）。"""
        image_shape = image.shape[:2]  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        dataset_dict["padding_mask"] = torch.as_tensor(np.ascontiguousarray(padding_mask))

        """如果不是训练模式，移除注释并返回处理后的数据集字典。"""
        if not self.is_train:
            # 遍历注释，移除关键点信息
            # USER: Modify this if you want to keep them for some reason.
            dataset_dict.pop("annotations", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                anno.pop("keypoints", None)

            # USER: Implement additional transformations if you have other types of data
            # 对每个注释应用变换，并过滤掉iscrowd属性为1的注释。
            annos = [
                utils.transform_instance_annotations(obj, transforms, image_shape)
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            # 将注释转换为实例对象。
            instances = utils.annotations_to_instances(annos, image_shape)
            # 从实例掩码中获取边界框。
            instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            # Need to filter empty instances first (due to augmentation)
            instances = utils.filter_empty_instances(instances)
            # Generate masks from polygon
            h, w = instances.image_size
            # image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float)
            if hasattr(instances, "gt_masks"):
                gt_masks = instances.gt_masks
                gt_masks = convert_coco_poly_to_mask(gt_masks.polygons, h, w)  # 将COCO格式的多边形转换为掩码。
                instances.gt_masks = gt_masks

            num_class_obj = {}
            for name in self.class_names:
                num_class_obj[name] = 0

            task = "The task is instance"
            text = self._get_texts(instances.gt_classes, num_class_obj)

            dataset_dict["instances"] = instances
            dataset_dict["orig_shape"] = image_shape
            dataset_dict["task"] = task
            dataset_dict["text"] = text
            dataset_dict["thing_ids"] = self.things

        return dataset_dict
