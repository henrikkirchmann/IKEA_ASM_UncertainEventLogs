# Author: Yizhak Ben-Shabat (Itzik), 2020
# test I3D on the ikea ASM dataset

import os
# Do not hard-force CUDA settings; respect the user's environment.
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import argparse
import i3d_utils
import sys
from pathlib import Path
# Ensure imports work regardless of current working directory:
# add the repo's `action/` directory to sys.path (this file is action/clip_based/i3d/test_i3d.py).
sys.path.append(str(Path(__file__).resolve().parents[2]))
import utils
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms
import videotransforms
import numpy as np
from pytorch_i3d import InceptionI3d
from IKEAActionDataset import IKEAActionVideoClipDataset as Dataset


parser = argparse.ArgumentParser()
parser.add_argument('-mode', type=str, default='rgb', help='rgb | depth, indicating which data to load')
parser.add_argument('-frame_skip', type=int, default=1, help='reduce fps by skipping frames')
parser.add_argument('-batch_size', type=int, default=8, help='number of clips per batch')
parser.add_argument('-db_filename', type=str,
                    default='/mnt/IronWolf/Datasets/ANU_ikea_dataset_smaller/ikea_annotation_db_full',
                    help='database file')
parser.add_argument('-model_path', type=str, default='./log/dev3/',
                    help='path to model save dir')
parser.add_argument('-device', default='dev3', help='which camera to load')
parser.add_argument('-model', type=str, default='best_classifier.pt', help='path to model save dir')
parser.add_argument('-dataset_path', type=str,
                    default='/mnt/IronWolf/Datasets/ANU_ikea_dataset_smaller/', help='path to dataset')
parser.add_argument('--force_cpu', action='store_true', help='Force CPU inference (slow).')
parser.add_argument(
    "--dataset_mode",
    type=str,
    default="vid",
    choices=["vid", "img"],
    help="How to read inputs. Use 'vid' for scan_video.avi (recommended for depth zip), 'img' for extracted frames.",
)
parser.add_argument(
    "--compute_device",
    type=str,
    default="auto",
    choices=["auto", "cuda", "mps", "cpu"],
    help="Compute backend selection. auto prefers: cuda -> mps -> cpu.",
)
parser.add_argument(
    "--num_workers",
    type=int,
    default=0,
    help=(
        "DataLoader workers. Default 0 for macOS compatibility (spawn + sqlite connections are not picklable). "
        "Increase only if you know your dataset is multiprocessing-safe."
    ),
)
args = parser.parse_args()


def _pick_device(force_cpu: bool, compute_device: str) -> torch.device:
    if force_cpu or compute_device == "cpu":
        return torch.device("cpu")
    if compute_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested compute_device=cuda but torch.cuda.is_available() is False")
        return torch.device("cuda")
    if compute_device == "mps":
        mps_ok = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
        if not mps_ok:
            raise RuntimeError("Requested compute_device=mps but torch.backends.mps.is_available() is False")
        return torch.device("mps")
    # auto
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps_ok = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
    if mps_ok:
        return torch.device("mps")
    return torch.device("cpu")


def run(dataset_path, db_filename, model_path, output_path, frames_per_clip=64, mode='rgb',
        testset_filename='test_cross_env.txt', trainset_filename='train_cross_env.txt', frame_skip=1,
        batch_size=8, device='dev3', force_cpu: bool = False, compute_device: str = "auto", num_workers: int = 0,
        dataset_mode: str = "vid"):

    pred_output_filename = os.path.join(output_path, 'pred.npy')
    json_output_filename = os.path.join(output_path, 'action_segments.json')

    # setup dataset
    test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])

    test_dataset = Dataset(dataset_path, db_filename=db_filename, test_filename=testset_filename,
                           train_filename=trainset_filename, transform=test_transforms, set='test', camera=device,
                           frame_skip=frame_skip, frames_per_clip=frames_per_clip, resize=None, mode=dataset_mode,
                           input_type=mode)

    compute = _pick_device(force_cpu=force_cpu, compute_device=compute_device)
    pin_memory = compute.type == "cuda"
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=pin_memory,
    )

    # setup the model
    if mode == 'flow':
        i3d = InceptionI3d(400, in_channels=2)
    else:
        i3d = InceptionI3d(157, in_channels=3)
    num_classes = test_dataset.num_classes
    i3d.replace_logits(num_classes)
    map_location = None if compute.type == "cuda" else "cpu"
    checkpoints = torch.load(model_path, map_location=map_location)
    i3d.load_state_dict(checkpoints["model_state_dict"]) # load trained model
    i3d = i3d.to(compute)
    if compute.type == "cuda":
        i3d = nn.DataParallel(i3d)

    n_examples = 0

    # Iterate over data.
    avg_acc = []
    pred_labels_per_video = [[] for i in range(len(test_dataset.video_list))]
    logits_per_video = [[] for i in range(len(test_dataset.video_list)) ]
    gt_labels_per_video = [[] for i in range(len(test_dataset.video_list))]
    # last_vid_idx = 0
    for test_batchind, data in enumerate(test_dataloader):
        i3d.train(False)
        # get the inputs
        inputs, labels, vid_idx, frame_pad = data

        # wrap them in Variable
        inputs = Variable(inputs.to(compute), requires_grad=True)
        labels = Variable(labels.to(compute))

        t = inputs.size(2)
        logits = i3d(inputs)
        logits = F.interpolate(logits, t, mode='linear', align_corners=True)  # b x classes x frames

        acc = i3d_utils.accuracy_v2(torch.argmax(logits, dim=1), torch.argmax(labels, dim=1))
        avg_acc.append(acc.item())
        n_examples += batch_size
        print('batch Acc: {}, [{} / {}]'.format(acc.item(), test_batchind, len(test_dataloader)))
        logits = logits.permute(0, 2, 1)
        logits = logits.reshape(inputs.shape[0] * frames_per_clip, -1)
        pred_labels = torch.argmax(logits, 1).detach().cpu().numpy()
        gt_labels = torch.argmax(labels, dim=1).detach().cpu().numpy().reshape(-1)
        logits = torch.nn.functional.softmax(logits, dim=1).detach().cpu().numpy().tolist()

        pred_labels_per_video, logits_per_video, gt_labels_per_video = utils.accume_per_video_predictions_with_gt(
            vid_idx,
            frame_pad,
            pred_labels_per_video,
            logits_per_video,
            gt_labels_per_video,
            pred_labels,
            logits,
            gt_labels,
            frames_per_clip,
        )

    pred_labels_per_video = [np.array(pred_video_labels) for pred_video_labels in pred_labels_per_video]
    logits_per_video = [np.array(pred_video_logits) for pred_video_logits in logits_per_video]
    gt_labels_per_video = [np.array(gt_video_labels) for gt_video_labels in gt_labels_per_video]

    # Save predictions + aligned GT labels used during inference.
    np.save(pred_output_filename, {'pred_labels': pred_labels_per_video, 'logits': logits_per_video, 'gt_labels': gt_labels_per_video})
    utils.convert_frame_logits_to_segment_json(logits_per_video, json_output_filename, test_dataset.video_list,
                                               test_dataset.action_list)


if __name__ == '__main__':
    # need to add argparse
    output_path = os.path.join(args.model_path, 'results')
    os.makedirs(output_path, exist_ok=True)
    model_path = os.path.join(args.model_path, args.model)
    run(dataset_path=args.dataset_path, db_filename=args.db_filename, model_path=model_path,
        output_path=output_path, frame_skip=args.frame_skip,  mode=args.mode, batch_size=args.batch_size,
        device=args.device, force_cpu=args.force_cpu, compute_device=args.compute_device, num_workers=args.num_workers,
        dataset_mode=args.dataset_mode)
    # Evaluation expects GT and may not be meaningful/available for predicted-only runs; skip by default.
