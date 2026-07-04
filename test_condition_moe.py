import argparse
import os

import torch
import yaml
from tqdm import tqdm

from datasets.custom_dataset import get_dataloader, get_dataset
from datasets.custom_transforms import get_transformations
from datasets.utils.configs import INPUT_SIZE
from evaluation.evaluate_utils import PerformanceMeter, predict, save_input_img,save_gt
from models.build_models import build_model
from utils import create_pred_dir, get_output, to_cuda, load_checkpoint_state

BASELINES = {
    # 对应 PASCAL-Context (Table 5, ViT-L based)
    'pascal': {
        'semseg_mIoU': 80.25,       # ↑
        'human_parts_mIoU': 70.54,  # ↑
        'sal_maxF': 84.54,          # ↑
        'normals_mErr': 13.57,       # ↓
        'edge_odsF': 74.22
    },
    # 对应 NYUD-v2 (Table 14, ViT-B based)
    'nyud': {
        'semseg_mIoU': 51.15,       # ↑
        'depth_RMSE': 0.5792,       # ↓
        'normals_mErr': 19.77,       # ↓
        'edge_odsF': 77.35
    }
}
def eval_metric(tasks, dataname, test_dl, model, evaluate, save_predictions, save_input, save_gts, pred_dir):
    if evaluate:
        performance_meter = PerformanceMeter(dataname, tasks)

    if save_predictions:
        # Save all tasks
        tasks_to_save = tasks
    else:
        # Save only edge
        tasks_to_save = ["edge"] if "edge" in tasks else []

    assert evaluate or len(tasks_to_save) > 0

    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_dl, desc="Evaluating"):
            batch = to_cuda(batch)
            images = batch["image"]
            task_gts = batch["label"]

            if model.backbone.__class__.__name__ == "Condition_MoE_PRISM":
                outputs = model(images, vfm_training=False)
                outputs = outputs["output_for_tasks"]
            else:
                raise NotImplementedError(f"Unsupported backbone: {model.backbone.__class__.__name__}")
            if evaluate:
                # print(f"outputs keys: {outputs.keys()}")
                performance_meter.update({t: get_output(outputs[t], t) for t in tasks}, task_gts)

            for task in tasks_to_save:
                predict(dataname, batch["meta"], outputs, task, pred_dir)
                if save_gts and task == "human_parts":
                    save_gt(dataname, batch["meta"], task_gts, task, pred_dir)

            if save_input:
                save_input_img(
                    batch["meta"],
                    images,
                    pred_dir,
                    mean=[0.485, 0.456, 0.406],  # 如果你没做 normalize，就改成 None
                    std=[0.229, 0.224, 0.225]
                )


    if evaluate:
        # Get evaluation results
        eval_results = performance_meter.get_score()

        results_dict = {}
        for t in tasks:
            for key in eval_results[t]:
                results_dict[t + "_" + key] = eval_results[t][key]

        return results_dict


def test(exp, config_path, checkpoint_file,  results_dir, evaluate, save_predictions, save_input, save_gts, gpu_id):
    print("Evaluate %s" % exp)

    with open(config_path, "r") as stream:
        configs = yaml.safe_load(stream)
    configs["student"]["backbone"]["vfm_training"] = False
    torch.cuda.set_device(gpu_id)

    # Get dataset and tasks
    dataname = configs["dataset"]
    task_dict = configs["task_dict"]
    task_list = []
    for task_name in task_dict:
        task_list += [task_name] * task_dict[task_name]

    test_transforms = get_transformations(dataname, INPUT_SIZE[dataname], train=False)
    test_ds = get_dataset(dataname, train=False, tasks=task_list, transform=test_transforms)
    test_dl = get_dataloader(train=False, configs=configs, dataset=test_ds)

    # Setup output folders
    exp_dir, pred_dir = create_pred_dir(results_dir, exp, task_list)

    # Setup model
    if "student" in configs:
        # Get teacher names
        tea_dims = {}
        for tea_name in configs["teachers"]:
            tea_dims[tea_name] = 0  # dummy value

        # Build student
        stu_config = configs["student"]
        stu_config["backbone"]["tea_dims"] = tea_dims
        stu_config["backbone"]["aligner"] = False  # remove aligners
    else:
        stu_config = configs

    model = build_model(
        arch="mt",
        img_size=INPUT_SIZE[dataname],
        backbone_args=stu_config["backbone"],
        dataname=dataname,
        tasks=task_list,
    ).cuda()

    # load model from checkpoint
    # checkpoint_file = checkpoint
    if not os.path.exists(checkpoint_file):
        raise ValueError("Checkpoint %s not found!" % (checkpoint_file))

    _, state_dict, _, _ = load_checkpoint_state(checkpoint_file, weights_only=True, print_fn=print)

    # Remove aligners in student model
    if "student" in configs:
        out_dict = {}
        for k, v in state_dict.items():
            if "ts_aligner" in k:
                continue
            out_dict[k] = v
        state_dict = out_dict

    model.load_state_dict(state_dict, strict=False)

    res = eval_metric(task_list, dataname, test_dl, model, evaluate, save_predictions, save_input, save_gts, pred_dir)
    # Print and log results
    if evaluate:
        test_results = {key: "%.5f" % res[key] for key in res}
        print(test_results)
        results_file = os.path.join(results_dir, exp, "test_results.txt")
        with open(results_file, "w") as f:
            f.write(str(test_results))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", nargs="+", required=True, help="experiment name")
    parser.add_argument("--config_path", type=str, required=True, help="Config file path")
    parser.add_argument("--checkpoint", default=None, help="Load checkpoint")
    parser.add_argument("--results_dir", type=str, default="results", help="directory of results")
    parser.add_argument("--evaluate", action="store_true", help="evaluate models")
    parser.add_argument("--save_predictions", action="store_true",default=False,  help="save predictions")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU id")
    parser.add_argument("--save_input",action="store_true",default=False, help="Whether to save input images and GTs")
    parser.add_argument("--save_gts", action="store_true",default=False,  help="Whether to save input images and GTs")

    args = parser.parse_args()

    for exp in args.exp:
        test(exp, args.config_path, args.checkpoint, args.results_dir, args.evaluate, args.save_predictions, args.save_input, args.save_gts, args.gpu_id)
