import os
import json
import argparse
from detectron2.config import get_cfg, CfgNode as CN
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator
import build_timm_backbone

class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)

def main(args):
    register_coco_instances("my_dataset_train", {}, args.train_json, args.train_images)
    register_coco_instances("my_dataset_val", {}, args.val_json, args.val_images)
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.OUTPUT_DIR = args.output_dir
    cfg.MODEL.BACKBONE.NAME = "build_timm_backbone"
    cfg.MODEL.TIMM = CN()
    cfg.MODEL.TIMM.NAME = args.model_name
    cfg.MODEL.TIMM.PRETRAINED = True
    cfg.MODEL.TIMM.OUT_INDICES = tuple(args.out_indices)
    cfg.MODEL.FPN.IN_FEATURES = ["res2", "res3", "res4", "res5"]
    cfg.MODEL.FPN.OUT_CHANNELS = 256
    cfg.DATASETS.TRAIN = ("my_dataset_train",)
    cfg.DATASETS.TEST = ("my_dataset_val",)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = args.num_classes
    cfg.SOLVER.IMS_PER_BATCH = args.ims_per_batch
    cfg.SOLVER.BASE_LR = args.base_lr
    cfg.SOLVER.MAX_ITER = args.max_iter
    cfg.SOLVER.STEPS = tuple(args.steps)
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    if args.weights:
        cfg.MODEL.WEIGHTS = args.weights
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = Trainer(cfg)
    if args.eval_only:
        trainer.resume_or_load(resume=False)
        results = trainer.test(cfg, trainer.model)
        with open(os.path.join(cfg.OUTPUT_DIR, "eval_results.json"), "w") as f:
            json.dump(results, f, indent=2)
        return
    trainer.resume_or_load(resume=True)
    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-json", required=True)
    parser.add_argument("--val-json", required=True)
    parser.add_argument("--train-images", required=True)
    parser.add_argument("--val-images", required=True)
    parser.add_argument("--num-classes", type=int, required=True)
    parser.add_argument("--output-dir", default="./output_timm")
    parser.add_argument("--model-name", default="tf_efficientnet_b3_ns")
    parser.add_argument("--out-indices", nargs="+", type=int, default=[1, 2, 3, 4])
    parser.add_argument("--ims-per-batch", type=int, default=2)
    parser.add_argument("--base-lr", type=float, default=0.00025)
    parser.add_argument("--max-iter", type=int, default=20000)
    parser.add_argument("--steps", nargs="+", type=int, default=[12000, 16000])
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--weights", default=None)
    args = parser.parse_args()
    main(args)
