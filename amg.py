# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.



import sys
import os
import traceback

# print("os path = "+ os.getcwd() + "\n")
# print("sys path = "+ str(sys.path) + "\n")

sys.path.append(os.getcwd())


os.chdir(os.getcwd())

bundle_dir = os.path.dirname(os.path.abspath(__file__))

print ('Location : ' + bundle_dir) # Where the base file exists
# print("os path = "+ os.getcwd() + "\n")
# print("sys path = "+ str(sys.path) + "\n")

# os.chdir(wd)
# print(os.getcwd())
# print(sys.path)

import cv2  # type: ignore
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import argparse
import json
import os
from typing import Any, Dict, List
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm


parser = argparse.ArgumentParser(
    description=(
        "Runs automatic mask generation on an input image or directory of images, "
        "and outputs masks as either PNGs or COCO-style RLEs. Requires open-cv, "
        "as well as pycocotools if saving in RLE format."
    )
)
FLAG_VIDEO=1
FLAG_IMAGE=2
FLAG_NONE=0
parser.add_argument(
    "--input",
    type=str,
    default='./',
    # required=True,
    help="Path to either a single input image or folder of images.",
)

parser.add_argument(
    "--output",
    type=str,
    default='output',
    # required=True,
    help=(
        "Path to the directory where masks will be output. Output will be either a folder "
        "of PNGs per image or a single json with COCO-style masks."
    ),
)

parser.add_argument(
    "--model-type",
    type=str,
    default='vit_h',
    # required=True,
    help="The type of model to load, in ['default', 'vit_h', 'vit_l', 'vit_b']",
)

parser.add_argument(
    "--checkpoint",
    type=str,
    default='dist/models/sam_vit_h_4b8939.pth',
    # required=True,
    help="The path to the SAM checkpoint to use for mask generation.",
)

parser.add_argument(
    "--second",
    type=float,
    default=60,
    help="Get frame within every pointed second",
)

parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="The device to run generation on.")

parser.add_argument(
    "--convert-to-rle",
    action="store_true",
    help=(
        "Save masks as COCO RLEs in a single json instead of as a folder of PNGs. "
        "Requires pycocotools."
    ),
)

amg_settings = parser.add_argument_group("AMG Settings")

amg_settings.add_argument(
    "--points-per-side",
    type=int,
    default=None,
    help="Generate masks by sampling a grid over the image with this many points to a side.",
)

amg_settings.add_argument(
    "--points-per-batch",
    type=int,
    default=None,
    help="How many input points to process simultaneously in one batch.",
)

amg_settings.add_argument(
    "--pred-iou-thresh",
    type=float,
    default=None,
    help="Exclude masks with a predicted score from the model that is lower than this threshold.",
)

amg_settings.add_argument(
    "--stability-score-thresh",
    type=float,
    default=None,
    help="Exclude masks with a stability score lower than this threshold.",
)
amg_settings.set_defaults()
amg_settings.add_argument(
    "--stability-score-offset",
    type=float,
    default=None,
    help="Larger values perturb the mask more when measuring stability score.",
)

amg_settings.add_argument(
    "--box-nms-thresh",
    type=float,
    default=None,
    help="The overlap threshold for excluding a duplicate mask.",
)

amg_settings.add_argument(
    "--crop-n-layers",
    type=int,
    default=None,
    help=(
        "If >0, mask generation is run on smaller crops of the image to generate more masks. "
        "The value sets how many different scales to crop at."
    ),
)

amg_settings.add_argument(
    "--crop-nms-thresh",
    type=float,
    default=None,
    help="The overlap threshold for excluding duplicate masks across different crops.",
)

amg_settings.add_argument(
    "--crop-overlap-ratio",
    type=int,
    default=None,
    help="Larger numbers mean image crops will overlap more.",
)

amg_settings.add_argument(
    "--crop-n-points-downscale-factor",
    type=int,
    default=None,
    help="The number of points-per-side in each layer of crop is reduced by this factor.",
)

amg_settings.add_argument(
    "--min-mask-region-area",
    type=int,
    default=None,
    help=(
        "Disconnected mask regions or holes with area smaller than this value "
        "in pixels are removed by postprocessing."
    ),
)


def write_masks_to_folder(masks: List[Dict[str, Any]], path: str) -> None:
    header = "id,area,bbox_x0,bbox_y0,bbox_w,bbox_h,point_input_x,point_input_y,predicted_iou,stability_score,crop_box_x0,crop_box_y0,crop_box_w,crop_box_h"  # noqa
    metadata = [header]
    for i, mask_data in enumerate(masks):
        mask = mask_data["segmentation"]
        filename = f"{i}.png"
        cv2.imwrite(os.path.join(path, filename), mask * 255)
        mask_metadata = [
            str(i),
            str(mask_data["area"]),
            *[str(x) for x in mask_data["bbox"]],
            *[str(x) for x in mask_data["point_coords"][0]],
            str(mask_data["predicted_iou"]),
            str(mask_data["stability_score"]),
            *[str(x) for x in mask_data["crop_box"]],
        ]
        row = ",".join(mask_metadata)
        metadata.append(row)
    metadata_path = os.path.join(path, "metadata.csv")
    with open(metadata_path, "w") as f:
        f.write("\n".join(metadata))

    return

def write_masks_to_folder(image, masks: List[Dict[str, Any]], path: str) -> None:
    header = "id,area,bbox_x0,bbox_y0,bbox_w,bbox_h,point_input_x,point_input_y,predicted_iou,stability_score,crop_box_x0,crop_box_y0,crop_box_w,crop_box_h"  # noqa
    metadata = [header]
    metadata1=[]

    for i, mask_data in enumerate(masks):
        mask = mask_data["segmentation"]
        
        mask2=mask*255
        print(mask2)
        mask2 = mask2.astype('uint8')

        result = cv2.bitwise_and(image, image, mask=mask2)
        filename = f"contour_{i+1}.png"
        
        # Create an alpha channel
        b, g, r = cv2.split(result)
        alpha = np.ones(b.shape, dtype=b.dtype) * 255  # Initialize alpha channel to 255 (opaque)

        # Set alpha to 0 where the mask is not applied (black background)
        alpha[mask2 == 0] = 0  # Set alpha to 0 for the transparent background

        # Merge the original image with the alpha channel
        result_with_alpha = cv2.merge((b, g, r, alpha))

        # cv2.imwrite(os.path.join(path, filename), mask * 255)
        cv2.imwrite(os.path.join(path, filename), result_with_alpha)
        mask_metadata = [
            str(i),
            str(mask_data["area"]),
            *[str(x) for x in mask_data["bbox"]],
            *[str(x) for x in mask_data["point_coords"][0]],
            str(mask_data["predicted_iou"]),
            str(mask_data["stability_score"]),
            *[str(x) for x in mask_data["crop_box"]],
        ]
        row = ",".join(mask_metadata)
        metadata.append(row)
        mask_metadata2 = [
            *[str(x) for x in mask_data["bbox"]],
        ]
        row = ",".join(mask_metadata2)
        metadata1.append(row)
    metadata_path = os.path.join(path, "metadata.csv")
    with open(metadata_path, "w") as f:
        f.write("\n".join(metadata))
    with open(os.path.join(path, "a_position.txt"), "w") as f:
        f.write("\n".join(metadata1))
    return


def get_amg_kwargs(args):
    amg_kwargs = {
        "points_per_side": args.points_per_side,
        "points_per_batch": args.points_per_batch,
        "pred_iou_thresh": args.pred_iou_thresh,
        "stability_score_thresh": args.stability_score_thresh,
        "stability_score_offset": args.stability_score_offset,
        "box_nms_thresh": args.box_nms_thresh,
        "crop_n_layers": args.crop_n_layers,
        "crop_nms_thresh": args.crop_nms_thresh,
        "crop_overlap_ratio": args.crop_overlap_ratio,
        "crop_n_points_downscale_factor": args.crop_n_points_downscale_factor,
        "min_mask_region_area": args.min_mask_region_area,
    }
    amg_kwargs = {k: v for k, v in amg_kwargs.items() if v is not None}
    return amg_kwargs

def is_video_file(filename):
    filename=str(filename)
    video_file_extensions = (
        '.264', '.3g2', '.3gp', '.3gp2', '.3gpp', '.3gpp2', '.3mm', '.3p2', '.60d', '.787', '.89', '.aaf', '.aec', '.aep', '.aepx',
        '.aet', '.aetx', '.ajp', '.ale', '.am', '.amc', '.amv', '.amx', '.anim', '.aqt', '.arcut', '.arf', '.asf', '.asx', '.avb',
        '.avc', '.avd', '.avi', '.avp', '.avs', '.avs', '.avv', '.axm', '.bdm', '.bdmv', '.bdt2', '.bdt3', '.bik', '.bin', '.bix',
        '.bmk', '.bnp', '.box', '.bs4', '.bsf', '.bvr', '.byu', '.camproj', '.camrec', '.camv', '.ced', '.cel', '.cine', '.cip',
        '.clpi', '.cmmp', '.cmmtpl', '.cmproj', '.cmrec', '.cpi', '.cst', '.cvc', '.cx3', '.d2v', '.d3v', '.dat', '.dav', '.dce',
        '.dck', '.dcr', '.dcr', '.ddat', '.dif', '.dir', '.divx', '.dlx', '.dmb', '.dmsd', '.dmsd3d', '.dmsm', '.dmsm3d', '.dmss',
        '.dmx', '.dnc', '.dpa', '.dpg', '.dream', '.dsy', '.dv', '.dv-avi', '.dv4', '.dvdmedia', '.dvr', '.dvr-ms', '.dvx', '.dxr',
        '.dzm', '.dzp', '.dzt', '.edl', '.evo', '.eye', '.ezt', '.f4p', '.f4v', '.fbr', '.fbr', '.fbz', '.fcp', '.fcproject',
        '.ffd', '.flc', '.flh', '.fli', '.flv', '.flx', '.gfp', '.gl', '.gom', '.grasp', '.gts', '.gvi', '.gvp', '.h264', '.hdmov',
        '.hkm', '.ifo', '.imovieproj', '.imovieproject', '.ircp', '.irf', '.ism', '.ismc', '.ismv', '.iva', '.ivf', '.ivr', '.ivs',
        '.izz', '.izzy', '.jss', '.jts', '.jtv', '.k3g', '.kmv', '.ktn', '.lrec', '.lsf', '.lsx', '.m15', '.m1pg', '.m1v', '.m21',
        '.m21', '.m2a', '.m2p', '.m2t', '.m2ts', '.m2v', '.m4e', '.m4u', '.m4v', '.m75', '.mani', '.meta', '.mgv', '.mj2', '.mjp',
        '.mjpg', '.mk3d', '.mkv', '.mmv', '.mnv', '.mob', '.mod', '.modd', '.moff', '.moi', '.moov', '.mov', '.movie', '.mp21',
        '.mp21', '.mp2v', '.mp4', '.mp4v', '.mpe', '.mpeg', '.mpeg1', '.mpeg4', '.mpf', '.mpg', '.mpg2', '.mpgindex', '.mpl',
        '.mpl', '.mpls', '.mpsub', '.mpv', '.mpv2', '.mqv', '.msdvd', '.mse', '.msh', '.mswmm', '.mts', '.mtv', '.mvb', '.mvc',
        '.mvd', '.mve', '.mvex', '.mvp', '.mvp', '.mvy', '.mxf', '.mxv', '.mys', '.ncor', '.nsv', '.nut', '.nuv', '.nvc', '.ogm',
        '.ogv', '.ogx', '.osp', '.otrkey', '.pac', '.par', '.pds', '.pgi', '.photoshow', '.piv', '.pjs', '.playlist', '.plproj',
        '.pmf', '.pmv', '.pns', '.ppj', '.prel', '.pro', '.prproj', '.prtl', '.psb', '.psh', '.pssd', '.pva', '.pvr', '.pxv',
        '.qt', '.qtch', '.qtindex', '.qtl', '.qtm', '.qtz', '.r3d', '.rcd', '.rcproject', '.rdb', '.rec', '.rm', '.rmd', '.rmd',
        '.rmp', '.rms', '.rmv', '.rmvb', '.roq', '.rp', '.rsx', '.rts', '.rts', '.rum', '.rv', '.rvid', '.rvl', '.sbk', '.sbt',
        '.scc', '.scm', '.scm', '.scn', '.screenflow', '.sec', '.sedprj', '.seq', '.sfd', '.sfvidcap', '.siv', '.smi', '.smi',
        '.smil', '.smk', '.sml', '.smv', '.spl', '.sqz', '.srt', '.ssf', '.ssm', '.stl', '.str', '.stx', '.svi', '.swf', '.swi',
        '.swt', '.tda3mt', '.tdx', '.thp', '.tivo', '.tix', '.tod', '.tp', '.tp0', '.tpd', '.tpr', '.trp', '.ts', '.tsp', '.ttxt',
        '.tvs', '.usf', '.usm', '.vc1', '.vcpf', '.vcr', '.vcv', '.vdo', '.vdr', '.vdx', '.veg','.vem', '.vep', '.vf', '.vft',
        '.vfw', '.vfz', '.vgz', '.vid', '.video', '.viewlet', '.viv', '.vivo', '.vlab', '.vob', '.vp3', '.vp6', '.vp7', '.vpj',
        '.vro', '.vs4', '.vse', '.vsp', '.w32', '.wcp', '.webm', '.wlmp', '.wm', '.wmd', '.wmmp', '.wmv', '.wmx', '.wot', '.wp3',
        '.wpl', '.wtv', '.wve', '.wvx', '.xej', '.xel', '.xesc', '.xfl', '.xlmv', '.xmv', '.xvid', '.y4m', '.yog', '.yuv', '.zeg',
        '.zm1', '.zm2', '.zm3', '.zmv'  )
    image_file_extensions = (
        '.png', '.jpg', '.bmp', '.jpeg', '.svg', '.JPG', '.JPEG', '.PNG', '.SVG', '.BMP')
    if filename.endswith(video_file_extensions):
        return FLAG_VIDEO
    elif filename.endswith(image_file_extensions):
        return FLAG_IMAGE
    return FLAG_NONE


def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask

    ax.imshow(img)


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


def main(args: argparse.Namespace) -> None:
    print("Loading model...")
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    _ = sam.to(device=args.device)
    output_mode = "binary_mask"
    print(args)
    amg_kwargs = get_amg_kwargs(args)
    generator = SamAutomaticMaskGenerator(sam, output_mode=output_mode, **amg_kwargs)
    if not os.path.isdir(args.input):
        if(is_video_file([args.input][0]) == FLAG_VIDEO):
            targets = [args.input]
        elif (is_video_file([args.input][0]) == FLAG_IMAGE):
            targets = [args.input]
        else:
            print("The file or directory does not exist in '" + args.input + "'.\n" + "Please check this file or directory.")
            return
    else:
        targets = [
            f for f in os.listdir(args.input) if not os.path.isdir(os.path.join(args.input, f))
        ]
        targets = [os.path.join(args.input, f) for f in targets]

    os.makedirs(args.output, exist_ok=True)

    for t in targets:
        t=t.replace('\\', '/')
        print(f"Processing '{t}'...\n\r\t")
        if (is_video_file(t) == FLAG_VIDEO):
            video_name1 = t
            video_name = video_name1
            property_id = int(cv2.CAP_PROP_FRAME_COUNT)
            cap = cv2.VideoCapture(video_name)  # video_name is the video being called
            frames = []
            time_milli = cap.get(cv2.CAP_PROP_POS_MSEC)

            fps = cap.get(cv2.CAP_PROP_FPS)  # OpenCV v2.x used "CV_CAP_PROP_FPS"
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps
            base = os.path.basename(video_name1)
            base = os.path.splitext(base)[0]
            save_base = os.path.join(args.output, base)
            os.makedirs(save_base, exist_ok=True)
            rate = (args.second * 1000 / (1000 / fps))
            cnt=frame_count//int(rate)
            for i in tqdm(range(0, cnt*int(rate), int(rate))):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)  # Where frame_no is the frame you want
                ret, image = cap.read()  # Read the frame
                image2 = image
                if image is None:
                    print(f"Could not load '{i}' as an image, skipping...")
                    continue
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                masks = generator.generate(image)

                if output_mode == "binary_mask":
                    save_base1 = os.path.join(save_base, f"frame_{i+1}")
                    os.makedirs(save_base1, exist_ok=True)
                    plt.figure(figsize=(100, 100))
                    plt.imshow(image)
                    show_anns(masks)
                    plt.axis('off')
                    plt.savefig(save_base1 + f"/result.jpg")
                    plt.imsave(save_base1 + f"/source.jpg", image)
                    plt.clf()
                    plt.cla()
                    plt.close('all')
                    write_masks_to_folder(image2, masks, save_base1)
                else:
                    save_file = save_base + ".json"
                    with open(save_file, "w") as f:
                        json.dump(masks, f)
                # print(f"finished frame {i+1}/{frame_count//int(rate)}")
            cv2.destroyAllWindows()
            cap.release()

        elif (is_video_file(t) == FLAG_IMAGE):
            try:
                print('**************************************************************** here is image file **************************************************************')
                # for j in tqdm(range(0,1)):
                image = cv2.imread(t)
                plt.figure(figsize=(20, 20))
                plt.imshow(image)
                plt.axis('off')
                image2 = image
                if image is None:
                    print(f"Could not load '{t}' as an image, skipping...")
                    continue
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                masks = generator.generate(image)

                base = os.path.basename(t)
                base = os.path.splitext(base)[0]
                save_base = os.path.join(args.output, base)

                if output_mode == "binary_mask":
                    os.makedirs(save_base, exist_ok=True)

                    plt.figure(figsize=(20, 20))
                    plt.imshow(image)
                    show_anns(masks)
                    plt.axis('off')
                    plt.savefig(save_base + f"/result.jpg")
                    plt.imsave(save_base + f"/source.jpg", image)
                    plt.clf()
                    plt.cla()
                    # plt.close('all')
                    write_masks_to_folder(image2, masks, save_base)
                else:
                    save_file = save_base + ".json"
                    with open(save_file, "w") as f:
                        json.dump(masks, f)
            except Exception as e:
                traceback.extract_stack()
                print(e)
        else:
            print(f"Could not load '{t}' as an image, skipping...")
            continue
    print("Done!")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
