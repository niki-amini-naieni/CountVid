import torch
from torch.utils.data import Dataset
import matplotlib.patches as patches
from torch.utils.data import DataLoader
import time
import multiprocessing
from itertools import repeat
from functools import reduce

# https://github.com/facebookresearch/segment-anything-2/blob/main/notebooks/video_predictor_example.ipynb
print("CUDA is available:", torch.cuda.is_available())
import copy
import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import shutil

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import re
import cv2
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

device = torch.device("cuda")

print(f"using device: {device}")

import argparse
from util.slconfig import SLConfig, DictAction
import datasets.transforms as T
import random
import json
from sam2.build_sam import build_sam2_video_predictor
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from util.misc import nested_tensor_from_tensor_list
import scipy.ndimage as ndimage

# Confidence threshold for CountGD
CONF_THRESH = 0.23


def get_args_parser():
    parser = argparse.ArgumentParser("Counting in Videos", add_help=False)

    parser.add_argument("--device", default="cuda", help="device to use for inference")
    parser.add_argument(
        "--options",
        nargs="+",
        action=DictAction,
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file.",
    )

    # Visualization parameters
    parser.add_argument(
        "--font_size", type=int, default=12, help="font size for object id labels"
    )

    # Model parameters
    parser.add_argument(
        "--min_obj_area",
        type=int,
        default=0,
        help="min pixel area for mask to be considered belonging to a distinct object",
    )
    parser.add_argument(
        "--sam_checkpoint",
        type=str,
        default="/work/nikian/sam2_checkpoints/sam2.1_hiera_small.pt",
        help="path to SAM 2 checkpoint",
    )
    parser.add_argument(
        "--sam_model_cfg",
        type=str,
        help="config file for SAM 2 checkpoint"
    )

    parser.add_argument("--temporal_filter", action="store_true", help="apply temporal filter")
    parser.add_argument("--w", type=int, default=3, help="window size for temporal filter (in frames)")
    # dataset parameters
    parser.add_argument("--convert_to_rgb", action="store_true", help="convert video frames and exemplar frames to RGB")
    parser.add_argument("--use_exemplars", action="store_true", help="Use visual exemplars to specify the object to count")
    parser.add_argument("--exemplar_file", type=str, help="name of file containing the visual exemplar bounding boxes")
    parser.add_argument("--exemplar_image_file", type=str, help="name of image file for the visual exemplars")
    parser.add_argument("--obj_batch_size", type=int, default=30, help="max number of objects to propagate through SAM 2 at once in Stage 3")
    parser.add_argument("--obj_batch_size_filter", type=int, default=100, help="max number of objects to propagate through SAM 2 at once for the temporal filter in Stage 2")
    parser.add_argument("--img_batch_size", type=int, default=1, help="batch size for independent CountGD + SAM 2 inference (before temporal propagation)")# Force to 1 for pseudo-exemplars
    parser.add_argument("--remove_difficult", action="store_true")
    parser.add_argument("--fix_size", action="store_true")
    parser.add_argument(
        "--video_dir", type=str, help="directory containing video frames"
    )
    parser.add_argument(
        "--input_text", type=str, help="text specifying object to count"
    )
    parser.add_argument(
        "--sample_frames",
        type=int,
        default=0,
        help="number of video frames to sample, defaults to 0 which indicates using all provided frames",
    )
    parser.add_argument(
        "--downsample_factor",
        type=float,
        default=1,
        help="downsample total number of frames by this factor, i.e., total_frames/downsample_factor=num_frames, sample_frames takes higher priority"
    )
    parser.add_argument(
        "--save_T",
        action="store_true",
        help="save dictionary with predicted object masklets",
    )
    parser.add_argument(
        "--save_countgd_video",
        action="store_true",
        help="save visualizations of CountGD's automatically generated box prompts per frame as a video",
    )
    parser.add_argument(
        "--save_sam_indep_video",
        action="store_true",
        help="save visualizations of sam applied to each image frame independently using CountGD box prompts as a video",
    )
    parser.add_argument(
        "--save_final_video",
        action="store_true",
        help="save final tracking results using our method for the whole video",
    )
    parser.add_argument(
        "--output_fps",
        type=float,
        default=3,
        help="frames per second of output videos"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="directory where to save video outputs"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="",
        help="save the counting results to an output json file"
    )
    parser.add_argument(
        "--temp_dir",
        type=str,
        default="./inference-frames",
        help="temporary directory used to store and process video frames, removed after counting finishes"
    )

    # Parameters required by CountGD app (have not tested removing, leaving in for now)
    parser.add_argument("--note", default="", help="add some notes to the experiment")
    parser.add_argument("--resume", default="", help="resume from checkpoint")
    parser.add_argument(
        "--pretrain_model_path",
        help="load from other checkpoint",
        default="checkpoint_best_regular.pth",
    )
    parser.add_argument("--finetune_ignore", type=str, nargs="+")
    parser.add_argument(
        "--start_epoch", default=0, type=int, metavar="N", help="start epoch"
    )
    parser.add_argument("--eval", action="store_false")# Sets model to eval mode by default since model only used for inference here
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--find_unused_params", action="store_true")
    parser.add_argument("--save_results", action="store_true")
    parser.add_argument("--save_log", action="store_true")

    # distributed training parameters
    parser.add_argument(
        "--world_size", default=1, type=int, help="number of distributed processes"
    )
    parser.add_argument(
        "--dist_url", default="env://", help="url used to set up distributed training"
    )
    parser.add_argument(
        "--rank", default=0, type=int, help="number of distributed processes"
    )
    parser.add_argument(
        "--local_rank", type=int, help="local rank for DistributedDataParallel"
    )
    parser.add_argument(
        "--local-rank", type=int, help="local rank for DistributedDataParallel"
    )
    parser.add_argument("--amp", action="store_true", help="Train with mixed precision")
    return parser


class VideoFrames(Dataset):
    def __init__(self, root, sorted_frame_names, transform, use_exemplars):
        self.video_dir = root
        self.frame_names = sorted_frame_names
        self.transform = transform
        # Specify exemplars here
        self.use_exemplars = use_exemplars
        if use_exemplars:
            with open(args.exemplar_file) as exemplar_file:
                exemplars = json.load(exemplar_file)["exemplars"]

            exemplars = get_box_inputs(exemplars)
            input_image_exemplars, exemplars = transform(
                Image.open(args.exemplar_image_file), {"exemplars": torch.tensor(exemplars)}
            )
            self.input_image_exemplars = input_image_exemplars
            self.exemplars = exemplars["exemplars"]

    def __len__(self):
        return len(self.frame_names)

    def __getitem__(self, idx):
        # Open image.
        input_frame = Image.open(os.path.join(self.video_dir, self.frame_names[idx]))

        # Transform input image.
        input_frame, _ = transform(input_frame, {"exemplars": torch.tensor([])})

        # Return input image and exemplars.
        label = torch.tensor([0])

        if self.use_exemplars:
            return (input_frame, self.input_image_exemplars, self.exemplars, label)
        else:
            return (input_frame, input_frame, torch.tensor([]), label)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


# **Begin code from CountGD app**
# Get counting model.
def build_model_and_transforms(args):
    normalize = T.Compose(
        [T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    )
    data_transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            normalize,
        ]
    )
    cfg = SLConfig.fromfile("cfg_app.py")
    cfg.merge_from_dict({"text_encoder_type": "checkpoints/bert-base-uncased"})
    cfg_dict = cfg._cfg_dict.to_dict()
    args_vars = vars(args)
    for k, v in cfg_dict.items():
        if k not in args_vars:
            setattr(args, k, v)
        else:
            raise ValueError("Key {} can used by args only".format(k))

    # fix the seed for reproducibility
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # we use register to maintain models from catdet6 on.
    from models.registry import MODULE_BUILD_FUNCS

    assert args.modelname in MODULE_BUILD_FUNCS._module_dict

    build_func = MODULE_BUILD_FUNCS.get(args.modelname)
    model, _, _ = build_func(args)

    checkpoint = torch.load(args.pretrain_model_path, map_location="cpu")["model"]
    model.load_state_dict(checkpoint, strict=False)

    model.eval()

    return model, data_transform

def get_ind_to_filter(text, word_ids, keywords):
    if len(keywords) <= 0:
        return list(range(len(word_ids)))
    input_words = text.split()
    keywords = keywords.split(",")
    keywords = [keyword.strip() for keyword in keywords]

    word_inds = []
    for keyword in keywords:
        if keyword in input_words:
            if len(word_inds) <= 0:
                ind = input_words.index(keyword)
                word_inds.append(ind)
            else:
                ind = input_words.index(keyword, word_inds[-1])
                word_inds.append(ind)
        else:
            raise Exception("Only specify keywords in the input text!")

    inds_to_filter = []
    for ind in range(len(word_ids)):
        word_id = word_ids[ind]
        if word_id in word_inds:
            inds_to_filter.append(ind)

    return inds_to_filter


def get_box_inputs(prompts):
    box_inputs = []
    for prompt in prompts:
        if prompt[2] == 2.0 and prompt[5] == 3.0:
            box_inputs.append([prompt[0], prompt[1], prompt[3], prompt[4]])

    return box_inputs

def get_pseudo_exemplars(model_output, image_size, box_threshold, num_exemplars=10):
    logits = model_output["pred_logits"].sigmoid()[0][:, :]
    boxes = model_output["pred_boxes"][0]

    # Get pseudo-exemplars
    box_mask = logits.max(dim=-1).values > box_threshold
    boxes = boxes[box_mask, :]
    logits = logits[box_mask, :]
    scores = logits.max(dim=-1).values
    num_exemplars = min(num_exemplars, boxes.shape[0])
    # Out of all the boxes, select at most [num_exemplars] of the highest scoring boxes.
    scores, indices = torch.sort(scores, dim=0, descending=True)
    boxes = boxes[indices, :]
    pseudo_exemplars = boxes[:num_exemplars, :]
    # Convert the normalized boxes to the exemplars format.
    (img_h, img_w) = (image_size[0], image_size[1])
    cx = img_w * pseudo_exemplars[:, 0]
    cy = img_h * pseudo_exemplars[:, 1]
    w = img_w * pseudo_exemplars[:, 2]
    h = img_h * pseudo_exemplars[:, 3]
    x0 = torch.clamp(cx - w/2, min=0, max=img_w)
    x1 = torch.clamp(cx + w/2, min=0, max=img_w)
    y0 = torch.clamp(cy - h/2, min=0, max=img_h)
    y1 = torch.clamp(cy + h/2, min=0, max=img_h)
    pseudo_exemplars = torch.stack([x0, y0, x1, y1], dim=-1)

    return pseudo_exemplars

# Define count function.
def count(input_image, input_image_exemplars, exemplars, label, text, device="cuda"):
    with torch.no_grad():
        return model(
            nested_tensor_from_tensor_list(list(input_image.to(device))),
            nested_tensor_from_tensor_list(list(input_image_exemplars.to(device))),
            list(exemplars.to(device)),
            list(label.to(device)),
            captions=[text + " ."] * len(input_image),
        )

# **End code from CountGD app**

def get_curr_count(frame_idx, T):
    num_objs = 0
    for obj_id in T:
        # Check if the object appears in the frame.
        if frame_idx in T[obj_id].keys():
            num_objs+=1
    return num_objs


def add_masks(out_frame_idx, T, cmap, img):
    """
    Visualize object masks in [T] at frame [out_frame_idx] stored as a numpy array in [img] using the color map [cmap].

    Effect: Shows the frame overlaid with the object masks and their corresponding object IDs.

    Used https://stackoverflow.com/questions/20081442/set-masked-pixels-in-a-3d-rgb-numpy-array
    """
    img_orig = copy.deepcopy(img)
    obj_centers = []
    num_objects = len(T.keys())
    for obj_id in T:
        # Check if the object appears in the frame.
        if out_frame_idx in T[obj_id].keys():
            color = np.array(cmap(obj_id / num_objects))
            mask = T[obj_id][out_frame_idx]
            img[mask[0], mask[1], :] = (255 * color[:3]).astype(int)

            # Label the mask:
            if len(mask[0]) > 0:
                y_center = int(np.mean(mask[0]))
                x_center = int(np.mean(mask[1]))
                obj_centers.append({"id": obj_id, "center": (x_center, y_center)})
            else:
                print("Object " + str(obj_id) + " is occluded in frame " + str(out_frame_idx))

    # Show the final image with masks.
    # Create a figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Display the first image in the first subplot
    axes[0].imshow(img_orig)
    axes[0].axis('off')  # Hide the axis

    # Display the second image in the second subplot
    axes[1].imshow(img)
    axes[1].axis('off')  # Hide the axis
    # Add text labels
    for label in obj_centers:
        obj_id = label["id"]
        (x_center, y_center) = label["center"]
        axes[1].text(
            x_center,
            y_center,
            f"{obj_id}",
            color="black",
            fontsize=args.font_size,
            ha="center",
            va="center",
        )


def add_masks_single_image(img, masks, cmap):
    """
    Visualize object masks by processing the logits in [out_mask_logits] for the image frame stored as a numpy array in [img] using the color map [cmap].

    Effect: Shows the frame overlaid with the object masks and their corresponding object IDs.

    Used https://stackoverflow.com/questions/20081442/set-masked-pixels-in-a-3d-rgb-numpy-array
    """
    num_objects = len(masks.keys())
    (h, w, c) = img.shape

    obj_centers = []
    for obj_id in masks:
        color = np.array(cmap(obj_id / num_objects))
        obj_mask = np.zeros((h, w), dtype=bool)
        obj_mask[masks[obj_id][0], masks[obj_id][1]] = True
        color = np.array(cmap(obj_id / num_objects))
        img[obj_mask, :] = (255 * color[:3]).astype(int)

        # Label the mask:
        if len(masks[obj_id][0]) > 0:
            y_center = int(np.mean(masks[obj_id][0]))
            x_center = int(np.mean(masks[obj_id][1]))
            obj_centers.append({"id": obj_id, "center": (x_center, y_center)})

    # Show the final image with masks.
    plt.imshow(img)
    # Add text labels
    for label in obj_centers:
        obj_id = label["id"]
        (x_center, y_center) = label["center"]
        plt.text(
            x_center,
            y_center,
            f"{obj_id}",
            color="black",
            fontsize=args.font_size,
            ha="center",
            va="center",
        )

def show_box(box, ax):
    """
    Adds the bounding box [box] to the matplotlib axis [ax].
    [box]: (x_leftmost, y_highest, x_rightmost, y_lowest)
    """
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="green", facecolor=(0, 0, 0, 0), lw=2)
    )

def get_indep_masks(predictor, images, box_prompts):
    # Note: sam masks from empty box prompts ignored as check in propagation loop that len(box_prompts) > 0 before checking for new objects.
    predictor.set_image_batch(images)
    masks_batch, scores_batch, _ = predictor.predict_batch(
            None, None, box_batch=box_prompts, multimask_output=False
        )
    return masks_batch

def get_short_term_masks(video_predictor, sam_pred_masks, w, args):
    # Store short-term masks in dictionary.
    sam_short_term_masks = {}
    for frame_ind_i in sam_pred_masks:
        sam_short_term_masks[frame_ind_i] = {}
        # Batch objects to speed up filtering while taking advantage of available memory.
        num_objs = len(sam_pred_masks[frame_ind_i]["box_prompts"])
        num_batches = int(np.ceil(num_objs / args.obj_batch_size_filter))
        print("Number of object batches for filtering in frame " + str(frame_ind_i) + ": " + str(num_batches))

        for batch_ind in range(num_batches):
            # Reset inference state.
            video_predictor.reset_state(inference_state)
            for obj_ind in range(batch_ind * args.obj_batch_size_filter, min((batch_ind + 1) * args.obj_batch_size_filter, num_objs)):
                obj_j = obj_ind + 1
                obj_j_box = sam_pred_masks[frame_ind_i]["box_prompts"][obj_ind]
                # Prompt SAM 2 with the box from [obj_j] in [frame_ind_i] and track the object max(0, frame_ind_i - (w - 1)) frames backward and min(N - 1, i + (w - 1)) frames forward, producing masks:
                # M^{track, j}_{max(0, frame_ind_i - (w - 1))}, ..., M^{track, j}_{i - 1},
                # M^{track, j}_{i + 1}, ..., M^{track, j}_{min(N - 1, i + (w - 1))}
                predictor.add_new_points_or_box(
                        inference_state=inference_state,
                        frame_idx=frame_ind_i,
                        obj_id=obj_j,
                        box=obj_j_box,
                    )
                sam_short_term_masks[frame_ind_i][obj_j] = {}
            # Track in reverse to get masks M^{track, j}_{max(0, frame_ind_i - (w - 1))}, ..., M^{track, j}_{i - 1}.
            for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
                inference_state, start_frame_idx=frame_ind_i, max_frame_num_to_track=w - 1, reverse=True
            ):
                for ind, obj_id in enumerate(out_obj_ids):
                    sam_short_term_masks[frame_ind_i][obj_id][out_frame_idx] = np.nonzero((out_mask_logits[ind] > 0.0).cpu().numpy().squeeze())
    
            # Track forward in time to get masks M^{track, j}_{i + 1}, ..., M^{track, j}_{min(N - 1, i + (w - 1))}.
            for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
                inference_state, start_frame_idx=frame_ind_i, max_frame_num_to_track=w - 1, reverse=False
            ):
                for ind, obj_id in enumerate(out_obj_ids):
                    sam_short_term_masks[frame_ind_i][obj_id][out_frame_idx] = np.nonzero((out_mask_logits[ind] > 0.0).cpu().numpy().squeeze())
    
    return sam_short_term_masks

def compute_iou(mask1_indices, mask2_indices):
    # Make indices hashable
    mask1_indices = list(zip(mask1_indices[0], mask1_indices[1]))
    mask2_indices = list(zip(mask2_indices[0], mask2_indices[1]))
    set1 = set(mask1_indices)
    set2 = set(mask2_indices)

    intersection = len(set1 & set2)
    if intersection == 0:
        return 0
    union = len(set1 | set2)

    if union == 0:
        return 0.0  # or define as 1.0 if both are empty masks

    return intersection / union

def compute_intersection(mask1_indices, mask2_indices):
    # Make indices hashable
    mask1_indices = list(zip(mask1_indices[0], mask1_indices[1]))
    mask2_indices = list(zip(mask2_indices[0], mask2_indices[1]))
    set1 = set(mask1_indices)
    set2 = set(mask2_indices)

    return len(set1 & set2)

def compute_union(mask1_indices, mask2_indices):
    # Make indices hashable
    mask1_indices = list(zip(mask1_indices[0], mask1_indices[1]))
    mask2_indices = list(zip(mask2_indices[0], mask2_indices[1]))
    set1 = set(mask1_indices)
    set2 = set(mask2_indices)

    return len(set1 | set2)
    
def check_obj_persistence(obj_j, frame_ind_i, N):
    global sam_short_term_masks
    global sam_pred_masks
    M_track_js = sam_short_term_masks[frame_ind_i][obj_j]
    # Find consecutive prior frames where [obj_j] is detected by CountGD.
    frame_ind_m = frame_ind_i
    for frame_ind_m in range(frame_ind_i - 1, max(0, frame_ind_i - (w - 1)) - 1, -1):
        M_track_j = M_track_js[frame_ind_m]
        IoUs = np.array([compute_iou(M_track_j, sam_pred_masks[frame_ind_m]["mask_ids"][obj_k]) for obj_k in sam_pred_masks[frame_ind_m]["mask_ids"]])
        if (IoUs > 0.5).sum() == 0:
            frame_ind_m = frame_ind_m + 1
            break

    start = frame_ind_m

    # Find consecutive future frames where [obj_j] is detected by CountGD.
    frame_ind_n = frame_ind_i
    for frame_ind_n in range(frame_ind_i + 1, min(N - 1, frame_ind_i + (w - 1)) + 1, 1):
        M_track_j = M_track_js[frame_ind_n]
        IoUs = np.array([compute_iou(M_track_j, sam_pred_masks[frame_ind_n]["mask_ids"][obj_k]) for obj_k in sam_pred_masks[frame_ind_n]["mask_ids"]])
        if (IoUs > 0.5).sum() == 0:
            frame_ind_n = frame_ind_n - 1
            break
            
    stop = frame_ind_n

    return stop + 1 - start >= w
            
def temporal_filter_fast(video_predictor, sam_pred_masks, countgd_pred_boxes, args, w=3):
    global sam_short_term_masks
    start_time_stage_2 = time.time()
    print("Starting to filter independent predictions with window size w=" + str(w))
    # Get short-term masks.
    sam_short_term_masks = get_short_term_masks(video_predictor, sam_pred_masks, w, args)
    time_stage_2_a = time.time() - start_time_stage_2
    sam_pred_masks_filtered = {}
    countgd_pred_boxes_filtered = {}
    N = len(sam_pred_masks.keys())
    args = []
    for frame_ind_i in range(N):
        countgd_pred_boxes_filtered[frame_ind_i] = []
        sam_pred_masks_filtered[frame_ind_i] = {"mask_ids": {}, "box_prompts": None}
        obj_ids = list(sam_pred_masks[frame_ind_i]["mask_ids"].keys())
        args = args + [(obj_id, frame_ind_i, N) for obj_id in obj_ids]
    with multiprocessing.Pool() as pool_obj:
        persistence = pool_obj.starmap(check_obj_persistence, args)
    frame_obj_pair_id = 0
    for frame_ind_i in range(N):
        persistent_obj_idx = 1
        num_objs = len(sam_pred_masks[frame_ind_i]["mask_ids"])
        for _ in range(num_objs):
            obj_j = args[frame_obj_pair_id][0]
            if persistence[frame_obj_pair_id]:
                # Keep [obj_j] if it persists in at least [w] consecutive frames (i.e., it is not transient).
                print(str(obj_j) + " is persistent, so it will be kept")
                obj_j_box = sam_pred_masks[frame_ind_i]["box_prompts"][obj_j - 1]
                countgd_pred_boxes_filtered[frame_ind_i].append(obj_j_box)
                sam_pred_masks_filtered[frame_ind_i]["mask_ids"][persistent_obj_idx] = sam_pred_masks[frame_ind_i]["mask_ids"][obj_j]
                persistent_obj_idx+=1
            else:
                print(str(obj_j) + " is transient, so it will be removed")
            frame_obj_pair_id+=1

        if len(countgd_pred_boxes_filtered[frame_ind_i]) > 0:
            countgd_pred_boxes_filtered[frame_ind_i] = torch.stack(countgd_pred_boxes_filtered[frame_ind_i], dim=0)
        else:
            countgd_pred_boxes_filtered[frame_ind_i] = torch.tensor([])
        sam_pred_masks_filtered[frame_ind_i]["box_prompts"] = countgd_pred_boxes_filtered[frame_ind_i]
    time_stage_2 = time.time() - start_time_stage_2
    time_stage_2_b = time_stage_2 - time_stage_2_a
    print("time stage 2: " + str(time_stage_2))
    print("time stage 2 a: " + str(time_stage_2_a))
    print("time stage 2 b: " + str(time_stage_2_b))
    return sam_pred_masks_filtered, countgd_pred_boxes_filtered

def temporal_filter(video_predictor, sam_pred_masks, countgd_pred_boxes, args, w=3):
    start_time_temporal_filter = time.time()
    print("Starting to filter independent predictions with window size w=" + str(w))
    # Get short-term masks.
    sam_short_term_masks = get_short_term_masks(video_predictor, sam_pred_masks, w, args)
    print("Time to get short-term masks: " + str(time.time() - start_time_temporal_filter) + "s")
    sam_pred_masks_filtered = {}
    countgd_pred_boxes_filtered = {}
    N = len(sam_pred_masks.keys())
    for frame_ind_i in range(N):
        print("filtering objects in frame " + str(frame_ind_i))
        countgd_pred_boxes_filtered[frame_ind_i] = []
        sam_pred_masks_filtered[frame_ind_i] = {"mask_ids": {}, "box_prompts": None}
        persistent_obj_idx = 1 
        for obj_j in sam_pred_masks[frame_ind_i]["mask_ids"]:
            print("checking object " + str(obj_j))
            M_track_js = sam_short_term_masks[frame_ind_i][obj_j]
            obj_j_box = sam_pred_masks[frame_ind_i]["box_prompts"][obj_j - 1]
            # Find consecutive prior frames where [obj_j] is detected by CountGD.
            frame_ind_m = frame_ind_i
            for frame_ind_m in range(frame_ind_i - 1, max(0, frame_ind_i - (w - 1)) - 1, -1):
                in_frame = False
                for obj_k in sam_pred_masks[frame_ind_m]["mask_ids"]:
                    # Get and reconstruct mask M^{indep, k}_{frame_ind_m} for [obj_k].
                    M_indep_k = np.zeros((H, W))
                    M_indep_k[sam_pred_masks[frame_ind_m]["mask_ids"][obj_k][0], sam_pred_masks[frame_ind_m]["mask_ids"][obj_k][1]] = 1
                    # Get and reconstruct the mask M^{track, j}_{frame_ind_m} for [obj_j].
                    M_track_j = np.zeros((H, W))
                    M_track_j[M_track_js[frame_ind_m][0], M_track_js[frame_ind_m][1]] = 1
                    # Calculate the mask IoU between M^{indep, k}_{frame_ind_m} and M^{track, j}_{frame_ind_m} to determine if they match.
                    intersection = np.logical_and(M_indep_k, M_track_j).sum()
                    union = np.logical_or(M_indep_k, M_track_j).sum()
                    if union > 0:
                        IoU = intersection / union
                    else:
                        IoU = 0
                    if IoU > 0.5:
                        in_frame = True
                        break
                if not in_frame:
                    frame_ind_m = frame_ind_m + 1
                    break
            start = frame_ind_m

            # Find consecutive future frames where [obj_j] is detected by CountGD.
            frame_ind_n = frame_ind_i
            for frame_ind_n in range(frame_ind_i + 1, min(N - 1, frame_ind_i + (w - 1)) + 1, 1):
                in_frame = False
                for obj_k in sam_pred_masks[frame_ind_n]["mask_ids"]:
                    # Get and reconstruct mask M^{indep, k}_{frame_ind_n} for [obj_k].
                    M_indep_k = np.zeros((H, W))
                    M_indep_k[sam_pred_masks[frame_ind_n]["mask_ids"][obj_k][0], sam_pred_masks[frame_ind_n]["mask_ids"][obj_k][1]] = 1
                    # Get and reconstruct the mask M^{track, j}_{frame_ind_n} for [obj_j].
                    M_track_j = np.zeros((H, W))
                    M_track_j[M_track_js[frame_ind_n][0], M_track_js[frame_ind_n][1]] = 1
                    # Calculate the mask IoU between M^{indep, k}_{frame_ind_n} and M^{track, j}_{frame_ind_n} to determine if they match.
                    intersection = np.logical_and(M_indep_k, M_track_j).sum()
                    union = np.logical_or(M_indep_k, M_track_j).sum()
                    if union > 0:
                        IoU = intersection / union
                    else:
                        IoU = 0
                    if IoU > 0.5:
                        in_frame = True
                        break
                if not in_frame:
                    frame_ind_n = frame_ind_n - 1
                    break
            stop = frame_ind_n

            # Keep [obj_j] if it persists in at least [w] consecutive frames (i.e., it is not transient).
            if stop + 1 - start >= w:
                print(str(obj_j) + " is persistent, so it will be kept")
                countgd_pred_boxes_filtered[frame_ind_i].append(obj_j_box)
                sam_pred_masks_filtered[frame_ind_i]["mask_ids"][persistent_obj_idx] = sam_pred_masks[frame_ind_i]["mask_ids"][obj_j]
                persistent_obj_idx = persistent_obj_idx + 1
            else:
                print(str(obj_j) + " is transient, so it will be removed")

        if len(countgd_pred_boxes_filtered[frame_ind_i]) > 0:
            countgd_pred_boxes_filtered[frame_ind_i] = torch.stack(countgd_pred_boxes_filtered[frame_ind_i], dim=0)
        else:
            countgd_pred_boxes_filtered[frame_ind_i] = torch.tensor([])
        sam_pred_masks_filtered[frame_ind_i]["box_prompts"] = countgd_pred_boxes_filtered[frame_ind_i]
    print("full temporal filter runtime: " + str(time.time() - start_time_temporal_filter))
    return sam_pred_masks_filtered, countgd_pred_boxes_filtered

def sam_image(predictor, inference_state, frame_id, boxes, frame_names, args):
    """
    Applies SAM 2.1 independently to the frame with ID [frame_id] using the boxes in [boxes] as box prompts. Each box corresponds to one distinct object.

    Returns: a dictionary with two keys, "mask_ids" (a dictionary with keys the object IDs and values the [image_height] x [image_width] binary masks for every object) and "box_prompts": the list of box prompts output by CountGD for the frame with ID [frame_id]
    """

    out_mask_logits = None  # in case no objects detected
    out_obj_ids = None
    num_objects = len(boxes)
    frame_preds = {"mask_ids": {}, "box_prompts": boxes}
    # Reset inference state for each frame independently.
    predictor.reset_state(inference_state)
    img = Image.open(os.path.join(args.video_dir, frame_names[frame_id]))
    # start with object id 1
    # Add a box prompt from CountGD for each object detected.
    for idx in range(len(boxes)):
        # for labels, `1` means positive click and `0` means negative click
        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=frame_id,
            obj_id=idx + 1,
            box=boxes[idx],
        )

    for idx in range(len(boxes)):
        obj_mask = (out_mask_logits[idx] > 0.0).cpu().numpy().squeeze()
        frame_preds["mask_ids"][predictor._obj_idx_to_id(inference_state, idx)] = np.nonzero(obj_mask)

    if args.save_sam_image:
        cmap = plt.get_cmap("rainbow")
        add_masks_single_image(
            out_obj_ids, out_mask_logits, cmap, num_objects, np.array(img)
        )
        plt.savefig("%05d-combined-image-output-frame.png" % frame_id)
        plt.close()

    return frame_preds

def set_or(a, b):
    return a | b
    
def get_new_objs(frame_idx_to_check, sam_img_pred, T, args):
    """
    Identifies new objects by checking two conditions that must *both* be true:
    (1) The mask of the new object has an area greater than args.min_obj_area
    (2) The mask of the new object is mutually exclusive (has no overlap) with any of the masks already predicted by SAM 2.1 for this frame

    Returns: the box prompts for any new objects identified
    """
    add_boxes = []
    if len(sam_img_pred["box_prompts"]) > 0:
        objs_in_frame = list(filter(lambda obj_id: frame_idx_to_check in T[obj_id], list(T.keys())))
        obj_coords = [set(list(zip(T[obj_id][frame_idx_to_check][0], T[obj_id][frame_idx_to_check][1]))) for obj_id in objs_in_frame]
        combined_obj_masks = reduce(set_or, obj_coords)

        # Check for new objects.
        for idx in range(len(sam_img_pred["box_prompts"])):
            obj_id = idx + 1
            obj_mask = set(list(zip(sam_img_pred["mask_ids"][obj_id][0], sam_img_pred["mask_ids"][obj_id][1])))
            if len(obj_mask) >= args.min_obj_area and (len(obj_mask & combined_obj_masks) == 0):
                add_boxes.append(
                    np.array([sam_img_pred["box_prompts"][idx]], dtype=np.float32)
                )
                print("new object detected at frame " + str(frame_idx_to_check))

    return add_boxes


# Convert video to sequence of frames with ffmeg, frame rate is specified with arg -r. In example below it is set to 1 sec:
# ffmpeg -i ../marbles-falling.mp4 -r 1 %05d.jpg
# `video_dir` a directory of JPEG frames with filenames like `<frame_index>.jpg`
# To convert the whole video to frames do:
# ffmpeg -i ../penguins.mp4 %05d.jpg
# To capture 3 frames per second do:
# ffmpeg -i ../penguins.mp4 -vf fps=3 %05d.jpg
# Got above from https://shotstack.io/learn/ffmpeg-extract-frames/#:~:text=Run%20the%20command%20below%20in%20your%20terminal%20to,is%20the%20name%20of%20the%20file%20we%27re%20using.
# Animated GIF maker: https://ezgif.com/maker
# Animated GIF combiner: https://ezgif.com/combine/ezgif-6-36dd3ea9-combine
# Animated GIF resizer: https://ezgif.com/resize/ezgif-3-49926e0ec9.gif

# Parse commandline arguments.
parser = get_args_parser()
args = parser.parse_args()
orig_vid_dir = args.video_dir

# Clean up after past runs.
try:
    shutil.rmtree(args.temp_dir)
except FileNotFoundError:
    print("no old img dir found")

# Load CountGD model.
device = get_device()
model, transform = build_model_and_transforms(args)
model = model.to(device)

# Get info for SAM 2.1 checkpoint.
sam2_checkpoint = args.sam_checkpoint
model_cfg = args.sam_model_cfg

video_dir = args.video_dir
print("Using directory " + video_dir)
input_text = args.input_text
print("Using text '" + input_text + "'")

# scan all the JPEG frame names in this directory
frame_names = [
    p
    for p in os.listdir(video_dir)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
]

# Sorting frame names based on their numeric parts.
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0][re.search(r"\d", p).start():]))
# Get frames.
max_frames = args.sample_frames
if max_frames > 0 and len(frame_names) > max_frames:
    idx = np.round(np.linspace(0, len(frame_names) - 1, max_frames)).astype(int)
    frame_names_new = []
    for id in idx:
        frame_names_new.append(frame_names[id])
    frame_names = frame_names_new

# If did not specify args.sample_frames, then apply args.downsample_factor.
print("original number of frames: " + str(len(frame_names)))
if args.sample_frames == 0:
    idx = np.round(np.linspace(0, len(frame_names) - 1, int(np.ceil(len(frame_names) / args.downsample_factor)))).astype(int)
    frame_names_new = []
    for id in idx:
        frame_names_new.append(frame_names[id])
    frame_names = frame_names_new
print("new number of frames: " + str(len(frame_names)))

# Get frame height and width.
init_frame = Image.open(os.path.join(video_dir, frame_names[0]))
W, H = init_frame.size

print("Using frames:")
for frame_name in frame_names:
    print(frame_name)

# Make video directory for SAM 2 inference.
os.mkdir(args.temp_dir)
for frame_ind, frame_name in enumerate(frame_names):
    shutil.copyfile(video_dir + "/" + frame_name, args.temp_dir + "/" + frame_name[re.search(r"\d", frame_name).start():])
    if args.convert_to_rgb:
        image_color = Image.open(args.temp_dir + "/" + frame_name[re.search(r"\d", frame_name).start():]).convert("RGB")
        image_color.save(args.temp_dir + "/" + frame_name[re.search(r"\d", frame_name).start():])
        
# Change video directory.
video_dir = args.temp_dir
args.video_dir = args.temp_dir
# Update frame names.
frame_names = [frame_name[re.search(r"\d", frame_name).start():] for frame_name in frame_names]

# Load SAM 2 image predictor.
sam2_image_predictor = SAM2ImagePredictor(build_sam2(model_cfg, sam2_checkpoint))

# Apply SAM 2 to all frames independently, using the boxes from CountGD as prompts.
test_video_data = VideoFrames(
    root=args.video_dir,
    sorted_frame_names=frame_names,
    transform=transform,
    use_exemplars=args.use_exemplars
)

video_frame_loader = DataLoader(test_video_data, batch_size=args.img_batch_size, shuffle=False, drop_last=False)

countgd_pred_boxes = {}
sam_pred_masks = {}
start_time_stage_1 = time.time()
deforming_exemplars = None
image_for_deforming_exemplars = None
start_updating_exemplars = False
for idx, (input_frame, input_image_exemplars, exemplars, label) in enumerate(video_frame_loader):
    print("batch idx: " + str(idx))

    if idx == 0:
        deforming_exemplars = exemplars
        image_for_deforming_exemplars = input_image_exemplars

    # Get box prompts for each frame.
    countgd_output = count(input_frame, image_for_deforming_exemplars, deforming_exemplars, label, input_text, device=args.device)

    pseudo_exemplars = get_pseudo_exemplars(countgd_output, (input_frame.shape[-2], input_frame.shape[-1]), CONF_THRESH).unsqueeze(0) # Add batch dimension
    if not start_updating_exemplars and deforming_exemplars[0].shape[0] < pseudo_exemplars[0].shape[0]:
        print("Starting to update exemplars dynamically")
        start_updating_exemplars = True
        print("original exemplars shape: " + str(deforming_exemplars.shape) + ", pseudo-exemplars.shape: " + str(pseudo_exemplars.shape))

    if start_updating_exemplars:
        deforming_exemplars = pseudo_exemplars
        image_for_deforming_exemplars = input_frame

        countgd_output = count(input_frame, image_for_deforming_exemplars, deforming_exemplars, label, input_text, device=args.device)
        
        pseudo_exemplars = get_pseudo_exemplars(countgd_output, (input_frame.shape[-2], input_frame.shape[-1]), CONF_THRESH).unsqueeze(0) # Add batch dimension
        
        deforming_exemplars = pseudo_exemplars

    logits = countgd_output["pred_logits"].sigmoid()
    boxes = countgd_output["pred_boxes"]
    (bs, _, _, _) = input_frame.shape
    box_mask = logits.max(dim=-1).values > CONF_THRESH
    batch_box_prompts = []
    sam_input_frames = []
    for intra_batch_ind in range(bs):
        box_mask_frame = box_mask[intra_batch_ind]
        boxes_frame = boxes[intra_batch_ind, box_mask_frame, :]
        boxes_w = W * boxes_frame[:, 2]
        boxes_h = H * boxes_frame[:, 3]
        boxes_x0 = W * boxes_frame[:, 0] - boxes_w / 2
        boxes_y0 = H * boxes_frame[:, 1] - boxes_h / 2
        boxes_x1 = W * boxes_frame[:, 0] + boxes_w / 2
        boxes_y1 = H * boxes_frame[:, 1] + boxes_h / 2
        box_prompts = torch.stack((boxes_x0, boxes_y0, boxes_x1, boxes_y1), dim=1)
        frame_id = idx * (args.img_batch_size) + intra_batch_ind
        box_prompts = box_prompts.cpu()
        countgd_pred_boxes[frame_id] = box_prompts
        if box_prompts.shape[0] > 0:
            batch_box_prompts.append(box_prompts)
        else:
            batch_box_prompts.append(None)
        sam_input_frames.append(np.array(Image.open(video_dir + "/" + frame_names[frame_id]).convert("RGB")))

    sam_masks = get_indep_masks(sam2_image_predictor, sam_input_frames, batch_box_prompts)
    del sam_input_frames
    for intra_batch_ind in range(bs):
        sam_mask = sam_masks[intra_batch_ind]
        frame_id = idx * (args.img_batch_size) + intra_batch_ind
        mask_ids = {}
        for mask_ind in range(len(countgd_pred_boxes[frame_id])):
            obj_mask = sam_mask[mask_ind].squeeze()
            mask_ids[mask_ind + 1] = np.nonzero(obj_mask)
        sam_pred_masks[frame_id] = {"mask_ids": mask_ids, "box_prompts": countgd_pred_boxes[frame_id]}

# Delete CountGD after use.
del model
# Delete SAM Image Predictor after use
del sam2_image_predictor
time_stage_1 = time.time() - start_time_stage_1
print("time stage 1: " + str(time_stage_1))

# Load SAM 2 video predictor.
predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
# Initialize the inference state.
inference_state = predictor.init_state(video_path=video_dir)
print("preventing masks from overlapping: " + str(predictor.non_overlap_masks))

if args.temporal_filter:
    # Apply a temporal filter to remove transient false positives from the independent frame predictions.
    w = args.w # temporal window of 1 sec (assum. 3 fps)
    sam_pred_masks, countgd_pred_boxes = temporal_filter_fast(predictor, sam_pred_masks, countgd_pred_boxes, args, w=w)

"""
-------------------------------------------------------
Boxes detected by CountGD, masks obtained by SAM 2

0. i <- first frame with objects detected by CountGD, N <- # of frames
1. Reset inference state.
2. Add boxes detected in frame i as box prompts.
3. Propagate the box prompts from step 2 through the video and add the resulting objects and tracked masks to the set T.
4. Check if there are object masks from single-image inference on frame i + 1 not covered by the predicted masks in T.
5. If there are object masks from single-image inference on frame i + 1 not covered by T, take branch (A).

Branch (A)
1. Reset inference state.
2. Add new boxes detected in frame i + 1 as box prompts.
3. Propagate the box prompts from Branch (A), step 2 through the video and add the resulting objects and tracked masks to the set T.

6. i <- i + 1. 
7. If i >= N - 1, stop and skip to END.
8. Go back to step 4.

END
Output the number of objects in T as the final count.
-------------------------------------------------------
"""
start_time_stage_3 = time.time()

# 0. i <- first frame with objects detected by CountGD, N <- # of frames
i = 0
N = len(frame_names)
for j in range(N):
    if len(countgd_pred_boxes[j]) > 0:
        i = j
        break

T = (
    {}
)  # Dictionary of tracked objects. Keys are the object IDs. The value at a an object ID is a dictionary with keys equal to the frame indices and values corresponding to the object's segmentation at each frame.

num_batches = int(np.ceil(len(countgd_pred_boxes[j]) / args.obj_batch_size))
print("Number of object batches: " + str(num_batches))

for batch_ind in range(num_batches):
  # 1. Reset inference state.
  predictor.reset_state(inference_state)

  # 2. Add boxes detected in frame i as box prompts.
  for box_ind in range(batch_ind * args.obj_batch_size, min((batch_ind + 1) * args.obj_batch_size, len(countgd_pred_boxes[i]))):
      _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
          inference_state=inference_state,
          frame_idx=i,
          obj_id=box_ind + 1,
          box=countgd_pred_boxes[i][box_ind],
      )

  # 3. Propagate the box prompts from step 2 through the video and add the resulting objects and tracked masks to the set T.
  for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
      inference_state
  ):
      for ind, obj_id in enumerate(out_obj_ids):
          if obj_id not in T:
              T[obj_id] = {}
          T[obj_id][out_frame_idx] = np.nonzero((out_mask_logits[ind] > 0.0).cpu().numpy().squeeze())

# 4. Check if there are object masks from single-image inference on frame i + 1 not covered by the predicted masks in T.
while i < N - 1:
    new_box_prompts = get_new_objs(i + 1, sam_pred_masks[i + 1], T, args)
    # 5. If there are object masks from single-image inference on frame i + 1 not covered by T, take branch (A).
    if len(new_box_prompts) > 0:
        # Branch (A):
        num_batches = int(np.ceil(len(new_box_prompts) / args.obj_batch_size))
        print("Number of object batches: " + str(num_batches))
        for batch_ind in range(num_batches):
            # 1. Reset inference state.
            predictor.reset_state(inference_state)
            # 2. Add new boxes detected in frame i + 1 as box prompts.
            last_obj_id = max(T.keys())
            
            for ind in range(min(args.obj_batch_size, len(new_box_prompts) - batch_ind * args.obj_batch_size)):
                box = new_box_prompts[ind]
                _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=i + 1,
                    obj_id=last_obj_id + ind + 1,
                    box=box,
                )

            # 3. Propagate the box prompts from Branch (A), step 2 through the video and add the resulting objects and tracked masks to the set T.
            for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
                inference_state
            ):
                # Note: [out_frame_idx] will start at the frame in which the new point prompts were provided, not 0.
                for ind, obj_id in enumerate(out_obj_ids):
                    if obj_id not in T:
                        T[obj_id] = {}
                    T[obj_id][out_frame_idx] = np.nonzero((out_mask_logits[ind] > 0.0).cpu().numpy().squeeze())
                    print(
                        "Added object with id "
                        + str(obj_id)
                        + " mask to T at frame "
                        + str(out_frame_idx)
                    )

    # 6. i <- i + 1.
    i = i + 1

# END
time_stage_3 = time.time() - start_time_stage_3
print("time stage 3: " + str(time_stage_3))
# Output the number of objects in T as the final count.
num_objects = len(T.keys())
print("Total Number of Objects: " + str(num_objects))

print(T.keys())
if args.save_T:
    # Save [T] for debugging.
    for obj_id in T:
        for frame_id in T[obj_id]:
            T[obj_id][frame_id] = [T[obj_id][frame_id][0].tolist(), T[obj_id][frame_id][1].tolist()]

    with open("T.json", "w") as fp:
        json.dump(T, fp)

# Save output videos if requested.
if len(args.output_dir) > 0:
    # Make directory for output.
    os.mkdir(args.output_dir)
if args.save_countgd_video:
    # Create images with CountGD boxes.
    output_frame_names = []
    for frame_id in countgd_pred_boxes:
        boxes = countgd_pred_boxes[frame_id].numpy()
        image = Image.open(video_dir + "/" + frame_names[frame_id])
        f_name = "countgd-output-frame-" + str(frame_id) + ".png"
        plt.imshow(image)
        for box in boxes:
            x0 = box[0]
            y0 = box[1]
            x1 = box[2]
            y1 = box[3]
            box_w = x1 - x0
            box_h = y1 - y0
            rect = patches.Rectangle(
            (x0, y0), box_w, box_h, linewidth=2, edgecolor="cyan", facecolor="none")
            plt.gca().add_patch(rect)
        plt.axis("off")
        plt.show()
        plt.savefig(args.output_dir + "/" + f_name)
        plt.close()
        image.close()
        output_frame_names.append(f_name)

    # Note this width and height may be different from original frame width and height due to matplotlib processing.
    frame = cv2.imread(os.path.join(args.output_dir, output_frame_names[0]))
    height, width, layers = frame.shape
    video = cv2.VideoWriter(args.output_dir + "/countgd-video.avi", 0, args.output_fps, (width, height))

    for output_frame_name in output_frame_names:
        video.write(cv2.imread(os.path.join(args.output_dir, output_frame_name)))

    cv2.destroyAllWindows()
    video.release()

if args.save_sam_indep_video:
    # Create images with independent SAM 2 masks.
    output_frame_names = []
    for frame_id in sam_pred_masks:
        masks = sam_pred_masks[frame_id]["mask_ids"]
        boxes = sam_pred_masks[frame_id]["box_prompts"]
        image = Image.open(video_dir + "/" + frame_names[frame_id])
        f_name = "sam-indep-output-frame-" + str(frame_id) + ".png"
        cmap = plt.get_cmap("rainbow")
        add_masks_single_image(np.array(image), masks, cmap)
        plt.axis("off")
        plt.show()
        plt.savefig(args.output_dir + "/" + f_name)
        plt.close()
        image.close()
        output_frame_names.append(f_name)

    # Note this width and height may be different from original frame width and height due to matplotlib processing.
    frame = cv2.imread(os.path.join(args.output_dir, output_frame_names[0]))
    height, width, layers = frame.shape
    video = cv2.VideoWriter(args.output_dir + "/sam-indep-video.avi", 0, args.output_fps, (width, height))

    for output_frame_name in output_frame_names:
        video.write(cv2.imread(os.path.join(args.output_dir, output_frame_name)))

    cv2.destroyAllWindows()
    video.release()
num_obj_by_frame = []
if args.save_final_video:
    # Create images with final masks.
    output_frame_names = []
    for frame_id in sam_pred_masks:
        image = Image.open(video_dir + "/" + frame_names[frame_id])
        f_name = "final-output-frame-" + str(frame_id) + ".png"
        cmap = plt.get_cmap("rainbow")
        add_masks(frame_id, T, cmap, np.array(image))
        plt.show()
        plt.savefig(args.output_dir + "/" + f_name)
        plt.close()
        image.close()
        output_frame_names.append(f_name)
        num_obj_by_frame.append(get_curr_count(frame_id, T))

    # Note this width and height may be different from original frame width and height due to matplotlib processing.
    frame = cv2.imread(os.path.join(args.output_dir, output_frame_names[0]))
    frame_height, frame_width, layers = frame.shape
    output_width = frame_width
    output_height = frame_height + 300
    video = cv2.VideoWriter(args.output_dir + "/final-video.mp4", cv2.VideoWriter_fourcc(*'MP4V'), args.output_fps, (output_width, output_height))

    x_data = []
    y_data = []
    for frame_ind in range(len(output_frame_names)):
        main_frame = cv2.resize(cv2.imread(os.path.join(args.output_dir, output_frame_names[frame_ind])), (frame_width, frame_height))
        combined_frame = np.zeros((output_height, output_width, 3), dtype=np.uint8)
        combined_frame[:frame_height, :, :] = main_frame

        # Add data to plot.
        x_data.append(frame_ind)
        y_data.append(num_obj_by_frame[frame_ind])
        fig, ax = plt.subplots(figsize=(8, 3), dpi=100, tight_layout=True)
        ax.plot(x_data, y_data, color='b', linewidth=2)
        ax.set_xlim(0, len(output_frame_names))
        ax.set_ylim(max(0, min(num_obj_by_frame) - 2), max(num_obj_by_frame) + 2)
        ax.set_xlabel("t")
        ax.set_ylabel("Count")

        # Render the plot to a numpy array
        canvas = FigureCanvas(fig)
        canvas.draw()
        graph_image = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
        graph_image = graph_image.reshape(canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)

        # Resize the graph image to fit the bottom of the video
        graph_image = cv2.resize(graph_image, (output_width, 300))

        # Place the graph in the bottom of the combined frame
        combined_frame[frame_height:, :, :] = graph_image
        plt.imshow(combined_frame)
        plt.savefig("combined.png")

        # Write the frame to the output video
        video.write(combined_frame)

    cv2.destroyAllWindows()
    video.release()

'''
if len(args.output_dir) > 0:
    # Remove all images used to compose output videos.
    out_dir_files = os.listdir(args.output_dir)
    
    for item in out_dir_files:
        if item.endswith(".png"):
            os.remove(os.path.join(args.output_dir, item))
'''

# Save counting results if requested.
if len(args.output_file) > 0:
    # Read and update current results.
    with open(args.output_file) as output_json:
        out_dict = json.load(output_json)
    if orig_vid_dir in out_dict:
        out_dict[orig_vid_dir][args.input_text] = num_objects
    else:
        out_dict[orig_vid_dir] = {args.input_text: num_objects}
    # Write new results.
    with open(args.output_file, 'w') as output_json:
        json.dump(out_dict, output_json)

# Remove the created directory with inference frames.
shutil.rmtree(args.temp_dir)
    
