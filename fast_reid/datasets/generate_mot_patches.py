import os
import argparse
import math
import cv2
import matplotlib.pyplot as plt
import numpy as np


def generate_trajectories(file_path, groundTrues):
    f = open(file_path, 'r')

    lines = f.read().split('\n')
    values = []
    for l in lines:
        split = l.split(',')
        if len(split) < 2:
            break
        numbers = [float(i) for i in split]
        values.append(numbers)

    values = np.array(values, np.float_)

    if groundTrues:
        # values = values[values[:, 6] == 1, :]  # Remove ignore objects
        # values = values[values[:, 7] == 1, :]  # Pedestrian only
        values = values[values[:, 8] > 0.4, :]  # visibility only

    values = np.array(values)
    values[:, 4] += values[:, 2]
    values[:, 5] += values[:, 3]

    return values


def make_parser():
    parser = argparse.ArgumentParser("MOTChallenge ReID dataset")

    parser.add_argument("--data_path", default="", help="path to MOT data")
    parser.add_argument("--save_path", default="fast_reid/datasets", help="Path to save the MOT-ReID dataset")
    parser.add_argument("--mot", default=17, help="MOTChallenge dataset number e.g. 17, 20")

    return parser


def main(args):

    # Create folder for outputs
    save_path = os.path.join(args.save_path, 'MOT' + str(args.mot) + '-ReID')
    os.makedirs(save_path, exist_ok=True)

    save_path = os.path.join(args.save_path, 'MOT' + str(args.mot) + '-ReID')
    train_save_path = os.path.join(save_path, 'bounding_box_train')
    os.makedirs(train_save_path, exist_ok=True)
    test_save_path = os.path.join(save_path, 'bounding_box_test')
    os.makedirs(test_save_path, exist_ok=True)

    # Get gt data
    data_path = os.path.join(args.data_path, 'MOT' + str(args.mot), 'train')

    if args.mot == '17':
        seqs = [f for f in os.listdir(data_path) if 'FRCNN' in f]
    else:
        seqs = os.listdir(data_path)

    seqs.sort()

    id_offset = 0

    for seq in seqs:
        print(seq)
        print(id_offset)

        ground_truth_path = os.path.join(data_path, seq, 'gt/gt.txt')
        gt = generate_trajectories(ground_truth_path, groundTrues=True)  # f, id, x_tl, y_tl, x_br, y_br, ...

        images_path = os.path.join(data_path, seq, 'img1')
        img_files = os.listdir(images_path)
        img_files.sort()

        num_frames = len(img_files)
        max_id_per_seq = 0
        for f in range(num_frames):

            img = cv2.imread(os.path.join(images_path, img_files[f]))

            if img is None:
                print("ERROR: Receive empty frame")
                continue

            H, W, _ = np.shape(img)

            det = gt[f + 1 == gt[:, 0], 1:].astype(np.int_)

            for d in range(np.size(det, 0)):
                id_ = det[d, 0]
                x1 = det[d, 1]
                y1 = det[d, 2]
                x2 = det[d, 3]
                y2 = det[d, 4]

                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(x2, W)
                y2 = min(y2, H)

                # patch = cv2.cvtColor(img[y1:y2, x1:x2, :], cv2.COLOR_BGR2RGB)
                patch = img[y1:y2, x1:x2, :]

                max_id_per_seq = max(max_id_per_seq, id_)

                # plt.figure()
                # plt.imshow(patch)
                # plt.show()

                fileName = (str(id_ + id_offset)).zfill(7) + '_' + seq + '_' + (str(f)).zfill(7) + '_acc_data.bmp'

                if f < num_frames // 2:
                    cv2.imwrite(os.path.join(train_save_path, fileName), patch)
                else:
                    cv2.imwrite(os.path.join(test_save_path, fileName), patch)

        id_offset += max_id_per_seq


if __name__ == "__main__":
    args = make_parser().parse_args()
    main(args)
