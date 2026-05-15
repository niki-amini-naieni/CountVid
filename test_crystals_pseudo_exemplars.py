import json
import argparse
import subprocess
import shlex
import os

parser = argparse.ArgumentParser("Testing on Crystals", add_help=False)
parser.add_argument(
    "--output_file", type=str, default="crystals-count-predicted.json", help="file where to save predicted counts"
)
parser.add_argument(
  "--data_dir", type=str, default="VideoCount/Crystals", help="path to Penguins dataset"
)
parser.add_argument(
    "--checkpoint_dir",
    type=str,
    default="checkpoints",
    help="directory with the checkpoints for SAM 2 and CountGD-Box",
)
parser.add_argument("--use_exemplars", action="store_true", help="whether or not to use the visual exemplars")
parser.add_argument("--no_text", action="store_true", help="whether or not to drop the text")

args = parser.parse_args()

# Use the specified args to get inputs for main [count_in_videos.py] script.
gt_file = os.path.join(args.data_dir, "anno", "crystals-count-gt.json") 
video_dir = os.path.join(args.data_dir, "frames")
sam_checkpoint = os.path.join(args.checkpoint_dir, "sam2.1_hiera_large.pt")
countgd_box_path = os.path.join(args.checkpoint_dir, "countgd_box.pth")
exemplar_dir = os.path.join(args.data_dir, "exemplars")



with open(gt_file) as crystals_json:
  crystals_gt = json.load(crystals_json)

# Create empty output file for results.
with open(args.output_file, 'w') as out_file:
  json.dump({}, out_file)

for video in crystals_gt:
  for input_text in crystals_gt[video]:
    if args.no_text:
        input_text = ""
    command = 'python count_in_videos_pseudo_exemplars.py --video_dir "' + os.path.join(video_dir, video) + '" --input_text "' + input_text + '" --sam_checkpoint "' + sam_checkpoint + '" --sam_model_cfg "configs/sam2.1/sam2.1_hiera_l.yaml" --obj_batch_size 30 --img_batch_size 10 --downsample_factor ' + str(3/3) + ' --pretrain_model_path "' + countgd_box_path + '" --output_file "' + args.output_file + '" --temp_dir "' + os.path.join(os.getcwd(), "crystals_temp_dir") + '" --temporal_filter --w 3 --convert_to_rgb'
    if args.use_exemplars:
        command = command + ' --use_exemplars --exemplar_file "' + os.path.join(exemplar_dir, video, "exemplars.json") + '" --exemplar_image_file "' + os.path.join(exemplar_dir, video, "exemplar_image.jpg") + '"'

    print("running: " + command)

    subprocess.run(
      shlex.split(command)
    )
