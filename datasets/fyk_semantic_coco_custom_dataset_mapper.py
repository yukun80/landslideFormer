import os
import cv2
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, build_detection_train_loader, DatasetCatalog
from detectron2.data.datasets import load_coco_json, register_coco_instances


class Register:
    """用于注册自己的数据集"""

    CLASS_NAMES = ["__background__", "landslide"]  # 类名233333
    ROOT = "./datasets/coco/"  #  填自己的2333

    def __init__(self):
        self.CLASS_NAMES = Register.CLASS_NAMES
        # 数据集路径233333
        self.ANN_ROOT = "./datasets/coco/annotations/"  # 填写自己的annotations路径

        self.TRAIN_PATH = os.path.join(Register.ROOT, "CAS_img_jpg")
        self.VAL_PATH = os.path.join(Register.ROOT, "CAS_img_jpg")

        self.TRAIN_JSON = os.path.join(self.ANN_ROOT, "instances_train2017.json")
        self.VAL_JSON = os.path.join(self.ANN_ROOT, "instances_val2017.json")

        # 声明数据集的子集 23333333
        self.PREDEFINED_SPLITS_DATASET = {
            "coco_landslide_train": (self.TRAIN_PATH, self.TRAIN_JSON),
            "coco_landslide_val": (self.VAL_PATH, self.VAL_JSON),
        }

    def register_dataset(self):
        """
        purpose: register all splits of datasets with PREDEFINED_SPLITS_DATASET
        注册数据集（这一步就是将自定义数据集注册进Detectron2）
        """
        for key, (image_root, json_file) in self.PREDEFINED_SPLITS_DATASET.items():
            self.register_dataset_instances(name=key, json_file=json_file, image_root=image_root)

    @staticmethod
    def register_dataset_instances(name, json_file, image_root):
        """复现了register_coco_instances函数
        purpose: register datasets to DatasetCatalog,
                 register metadata to MetadataCatalog and set attribute
        注册数据集实例，加载数据集中的对象实例
        """
        DatasetCatalog.register(name, lambda: load_coco_json(json_file, image_root, name))
        MetadataCatalog.get(name).set(json_file=json_file, image_root=image_root, evaluator_type="coco")

    def plain_register_dataset(self):
        """修改注册数据集和元数据"""
        # 训练集 233333333
        DatasetCatalog.register("coco_Jiuzhai_train", lambda: load_coco_json(self.TRAIN_JSON, self.TRAIN_PATH))
        MetadataCatalog.get("coco_Jiuzhai_train").set(
            thing_classes=self.CLASS_NAMES,  # 可以选择开启，但是不能显示中文，这里需要注意，中文的话最好关闭
            evaluator_type="coco",  # 指定评估方式
            json_file=self.TRAIN_JSON,
            image_root=self.TRAIN_PATH,
        )

        # DatasetCatalog.register("coco_my_val", lambda: load_coco_json(VAL_JSON, VAL_PATH, "coco_2017_val"))
        # 验证/测试集 23333333
        DatasetCatalog.register("coco_Jiuzhai_val", lambda: load_coco_json(self.VAL_JSON, self.VAL_PATH))
        MetadataCatalog.get("coco_Jiuzhai_val").set(
            thing_classes=self.CLASS_NAMES,  # 可以选择开启，但是不能显示中文，这里需要注意，中文的话最好关闭
            evaluator_type="coco",  # 指定评估方式
            json_file=self.VAL_JSON,
            image_root=self.VAL_PATH,
        )


if __name__ == "__main__":
    register_coco_instances("my_dataset_train", {}, "tool_data/train_coco_format.json", "tool_data/train")
    register_coco_instances("my_dataset_val", {}, "tool_data/val_coco_format.json", "tool_data/val")
