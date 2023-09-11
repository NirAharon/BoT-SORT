import sys
import argparse
import os
import os.path as osp
import time
import cv2
import torch
import asyncio
import numpy as np

from loguru import logger

sys.path.append('.')

from yolox.data.data_augment import preproc
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess
from yolox.utils.visualize import plot_tracking
from tracker.bot_sort import BoTSORT
from tracker.tracking_utils.timer import Timer

from multicam import MultiCameraTracking
from video_stream import RoundRobinVideoStream
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QWidget, QMainWindow, QApplication

from transform import four_point_transform
import torch
#torch.cuda.set_device(1)

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]


def make_parser():
    parser = argparse.ArgumentParser("BoT-SORT Demo!")
    parser.add_argument("demo", default="image", help="demo type, eg. image, video and webcam")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")
    parser.add_argument("--path", required=True, nargs='+', default="", help="path to images or video")
    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
    parser.add_argument("--save_result", action="store_true",help="whether to save the inference result of image/video")
    parser.add_argument("-f", "--exp_file", default=None, type=str, help="pls input your expriment description file")
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument("--device", default="gpu", type=str, help="device to run our model, can either be cpu or gpu")
    parser.add_argument("--conf", default=None, type=float, help="test conf")
    parser.add_argument("--nms", default=None, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument("--fps", default=30, type=int, help="frame rate (fps)")
    parser.add_argument("--fp16", dest="fp16", default=False, action="store_true",help="Adopting mix precision evaluating.")
    parser.add_argument("--fuse", dest="fuse", default=False, action="store_true", help="Fuse conv and bn for testing.")
    parser.add_argument("--trt", dest="trt", default=False, action="store_true", help="Using TensorRT model for testing.")

    # tracking args
    parser.add_argument("--track_high_thresh", type=float, default=0.6, help="tracking confidence threshold")
    parser.add_argument("--track_low_thresh", default=0.1, type=float, help="lowest detection threshold")
    parser.add_argument("--new_track_thresh", default=0.7, type=float, help="new track thresh")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
    parser.add_argument("--aspect_ratio_thresh", type=float, default=1.6, help="threshold for filtering out boxes of which aspect ratio are above the given value.")
    parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
    parser.add_argument("--fuse-score", dest="fuse_score", default=False, action="store_true", help="fuse score and iou for association")

    # CMC
    parser.add_argument("--cmc-method", default="orb", type=str, help="cmc method: files (Vidstab GMC) | orb | ecc")

    # ReID
    parser.add_argument("--with-reid", dest="with_reid", default=False, action="store_true", help="test mot20.")
    parser.add_argument("--fast-reid-config", dest="fast_reid_config", default=r"fast_reid/configs/MOT20/sbs_S50.yml", type=str, help="reid config file path")
    parser.add_argument("--fast-reid-weights", dest="fast_reid_weights", default=r"pretrained/mot20_sbs_S50.pth", type=str,help="reid config file path")
    parser.add_argument('--proximity_thresh', type=float, default=0.5, help='threshold for rejecting low overlap reid matches')
    parser.add_argument('--appearance_thresh', type=float, default=0.25, help='threshold for rejecting low appearance similarity reid matches')
    return parser


def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = osp.join(maindir, filename)
            ext = osp.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names


def write_results(filename, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},{s},-1,-1,-1\n'
    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids, scores in results:
            for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                line = save_format.format(frame=frame_id, id=track_id, x1=round(x1, 1), y1=round(y1, 1), w=round(w, 1), h=round(h, 1), s=round(score, 2))
                f.write(line)
    logger.info('save results to {}'.format(filename))


class Predictor(object):
    def __init__(
        self,
        model,
        exp,
        trt_file=None,
        decoder=None,
        device=torch.device("cpu"),
        fp16=False
    ):
        self.model = model
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones((1, 3, exp.test_size[0], exp.test_size[1]), device=device)
            self.model(x)
            self.model = model_trt
        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

    def inference(self, img, timer):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = osp.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        img, ratio = preproc(img, self.test_size, self.rgb_means, self.std)
        img_info["ratio"] = ratio
        img = torch.from_numpy(img).unsqueeze(0).float().to(self.device)
        if self.fp16:
            img = img.half()  # to FP16

        with torch.no_grad():
            timer.tic()
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(outputs, self.num_classes, self.confthre, self.nmsthre)
        return outputs, img_info

class MultiCamTracker(QWidget):
    frame_update = Signal(int)
    def __init__(self, exp, args):
        super().__init__()
        self.exp = exp
        self.args = args
        self.output_list = []
        # self.floor_plan = cv2.imread('random_floor_plan.jpg')  self.argts.path[]
        # self.tran_matrix = np.load('transformation_matrix.npy')
        self.floor_plan = np.zeros(((600, 600, 3)), np.uint8)
        # self.args.plot_rect_tlwh = [[50, 50, 100, 100], [350, 50, 100, 100]]
        self.tran_matrix = np.array([np.load(f'/home/abhilash/BoT-SORT/Transformation_matrices/{fname}.npy') for fname in self.args.cam_name])
        self.warped_images = [cv2.imread(f'/home/abhilash/BoT-SORT/Warped_images/{fname}.jpg') for fname in self.args.cam_name]
        # matrices_names = os.listdir('/home/abhilash/BoT-SORT/Transformation_matrices')
        # print("Matrices: ",matrices_names)
        # if len(matrices_names) > 0: # Transformation_matrices
        #     self.tran_matrix = np.array([np.load(f'/home/abhilash/BoT-SORT/Transformation_matrices/{fname}') for fname in matrices_names])
        # else:
        #     print("No transformation matrices available")

        
        # # Also, store the ROI warped images with the same names
        # warped_images = os.listdir('/home/abhilash/BoT-SORT/Warped_images')
        # print("Warped images: ",warped_images)
        # if len(warped_images) > 0:
        #     self.warped_images = [cv2.imread(f'/home/abhilash/BoT-SORT/Warped_images/{fname}') for fname in warped_images]
        # else:
        #     print("No warped images found!")

    def main(self):
        if not self.args.experiment_name:
            self.args.experiment_name = self.exp.exp_name

        output_dir = osp.join(self.exp.output_dir, self.args.experiment_name)
        os.makedirs(output_dir, exist_ok=True)

        if self.args.save_result:
            vis_folder = osp.join(output_dir, "track_vis")
            os.makedirs(vis_folder, exist_ok=True)

        if self.args.trt:
            self.args.device = "gpu"
        self.args.device = torch.device("cuda" if self.args.device == "gpu" else "cpu")

        logger.info("Args: {}".format(self.args))

        if self.args.conf is not None:
            self.exp.test_conf = self.args.conf
        if self.args.nms is not None:
            self.exp.nmsthre = self.args.nms
        if self.args.tsize is not None:
            self.exp.test_size = (self.args.tsize, self.args.tsize)

        model = self.exp.get_model().to(self.args.device)
        logger.info("Model Summary: {}".format(get_model_info(model, self.exp.test_size)))
        model.eval()

        if not self.args.trt:
            if self.args.ckpt is None:
                ckpt_file = osp.join(output_dir, "best_ckpt.pth.tar")
            else:
                ckpt_file = self.args.ckpt
            logger.info("loading checkpoint")
            ckpt = torch.load(ckpt_file, map_location="cpu")
            # load the model state dict
            model.load_state_dict(ckpt["model"])
            logger.info("loaded checkpoint done.")

        if self.args.fuse:
            logger.info("\tFusing model...")
            model = fuse_model(model)

        if self.args.fp16:
            model = model.half()  # to FP16

        if self.args.trt:
            assert not self.args.fuse, "TensorRT model is not support model fusing!"
            trt_file = osp.join(output_dir, "model_trt.pth")
            assert osp.exists(trt_file), "TensorRT model is not found!\n Run python3 tools/trt.py first!"
            model.head.decode_in_inference = False
            decoder = model.head.decode_outputs
            logger.info("Using TensorRT to inference")
        else:
            trt_file = None
            decoder = None

        predictor = Predictor(model, self.exp, trt_file, decoder, self.args.device, self.args.fp16)
        current_time = time.localtime()
        if self.args.demo == "image" or self.args.demo == "images":
            self.image_demo(predictor, vis_folder, current_time)
        elif self.args.demo == "video" or self.args.demo == "webcam":
            self.multicam(predictor, vis_folder, current_time, self.args)
        else:
            raise ValueError("Error: Unknown source: " + self.args.demo)

        # Initializing the top view variables
        # self.floor_plan = cv2.imread(self.args.path['floor_plan'])
        # self.floor_plan = cv2.imread('/home/abhilash/BoT-SORT/random_floor_plan.jpg')
    
        # There will be one matrix for one camera. Store every matrix according to the camera name.
        # Store every matrix in one folder called Transformation_matrices
        # self.tran_matrix = np.load('/home/abhilash/BoT-SORT/transformation_matrix.npy')
        # matrices_names = os.listdir('Transformation_matrices')
        # if len(matrices_names) > 0:
        #     self.tran_matrix = np.array([np.load(f'Transformation_matrices/{fname}') for fname in matrices_names])
        # else:
        #     print("No transformation matrices available")

        
        # Also, store the ROI warped images with the same names
        # warped_images = os.listdir('Warped_images')
        # if len(warped_images) > 0:
        #     self.warped_images = [cv2.imread(f'Warped_images{fname}') for fname in warped_images]
        # else:
        #     print("No warped images found!")

        # Not using a floor plan currently. Will be plotting on a black image.

        # This is the variable which has the co-ordinates for calibration
        self.selected_ROI = []

        # This has the co-ordinates for where to plot on the floor plan.
        # self.args.plot_rect_tlwh = [[50, 50, 100, 100], [350, 50, 100, 100]]

    def image_demo(predictor, vis_folder, current_time, args):
        if osp.isdir(args.path):
            files = get_image_list(args.path)
        else:
            files = [args.path]
        files.sort()

        tracker = BoTSORT(args, frame_rate=args.fps)

        timer = Timer()
        results = []

        for frame_id, img_path in enumerate(files, 1):

            # Detect objects
            outputs, img_info = predictor.inference(img_path, timer)
            scale = min(exp.test_size[0] / float(img_info['height'], ), exp.test_size[1] / float(img_info['width']))

            detections = []
            if outputs[0] is not None:
                outputs = outputs[0].cpu().numpy()
                detections = outputs[:, :7]
                detections[:, :4] /= scale

                # Run tracker
                online_targets = tracker.update(detections, img_info['raw_img'])

                online_tlwhs = []
                online_ids = []
                online_scores = []
                for t in online_targets:
                    tlwh = t.tlwh
                    tid = t.track_id
                    vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                    if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                        online_scores.append(t.score)
                        # save results
                        results.append(
                            f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                        )
                timer.toc()
                online_im = plot_tracking(
                    img_info['raw_img'], online_tlwhs, online_ids, frame_id=frame_id, fps=1. / timer.average_time
                )

            else:
                timer.toc()
                online_im = img_info['raw_img']

            # result_image = predictor.visual(outputs[0], img_info, predictor.confthre)
            if args.save_result:
                timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
                save_folder = osp.join(vis_folder, timestamp)
                os.makedirs(save_folder, exist_ok=True)
                cv2.imwrite(osp.join(save_folder, osp.basename(img_path)), online_im)

            if frame_id % 20 == 0:
                logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))

            ch = cv2.waitKey(0)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break

        if args.save_result:
            res_file = osp.join(vis_folder, f"{timestamp}.txt")
            with open(res_file, 'w') as f:
                f.writelines(results)
            logger.info(f"save results to {res_file}")


    def imageflow_demo(tracker, predictor, vis_folder, current_time, args, path_video, cam_id):
        cap = cv2.VideoCapture(path_video if args.demo == "video" else args.camid)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
        fps = cap.get(cv2.CAP_PROP_FPS)
        timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
        save_folder = osp.join(vis_folder, timestamp, str(cam_id))
        os.makedirs(save_folder, exist_ok=True)
        if args.demo == "video":
            save_path = osp.join(save_folder, path_video.split("/")[-1])
        else:
            save_path = osp.join(save_folder, "camera.mp4")
        logger.info(f"video save_path is {save_path}")
        vid_writer = cv2.VideoWriter(
            save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
        )
        # tracker = MultiCameraTracking(args, frame_rate=args.fps)
        timer = Timer()
        frame_id = 0
        results = []
        while True:
            if frame_id % 20 == 0:
                logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))
            ret_val, frame = cap.read()
            if ret_val:
                # Detect objects
                outputs, img_info = predictor.inference(frame, timer)
                scale = min(exp.test_size[0] / float(img_info['height'], ), exp.test_size[1] / float(img_info['width']))

                if outputs[0] is not None:
                    outputs = outputs[0].cpu().numpy()
                    detections = outputs[:, :7]
                    detections[:, :4] /= scale

                    # Run tracker
                    online_targets = tracker.process(detections, img_info["raw_img"], cam_id)

                    online_tlwhs = []
                    online_ids = []
                    online_scores = []
                    for t in online_targets:
                        tlwh = t.tlwh
                        tid = t.track_id
                        vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                        if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                            online_tlwhs.append(tlwh)
                            online_ids.append(tid)
                            online_scores.append(t.score)
                            results.append(
                                f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                            )
                    timer.toc()
                    online_im = plot_tracking(
                        img_info['raw_img'], online_tlwhs, online_ids, frame_id=frame_id + 1, fps=1. / timer.average_time
                    )
                else:
                    timer.toc()
                    online_im = img_info['raw_img']
                if args.save_result:
                    vid_writer.write(online_im)
                ch = cv2.waitKey(1)
                if ch == 27 or ch == ord("q") or ch == ord("Q"):
                    break
            else:
                break
            frame_id += 1

        if args.save_result:
            res_file = osp.join(vis_folder, f"{timestamp}.txt")
            with open(res_file, 'w') as f:
                f.writelines(results)
            logger.info(f"save results to {res_file}")

    
    def convert_list_to_bottom_center(self, coords_list):
        bottom_center_coords_list = []
        for tl_x, tl_y, width, height in coords_list:
            bc_x = tl_x + width / 2
            bc_y = tl_y + height
            bottom_center_coords_list.append((bc_x, bc_y))
        return bottom_center_coords_list            
    
    def map_to_target_region(self, point, cam_id):
        # This x,y,width,height is supposed to be the area on the floor plan where the points are to be plotted.
        # This will also come from the user.
        # self.rectangle_tlwhs = 
        print("Rectangle dimensions: ",self.args.plot_rect_tlwh[cam_id] )   #self.args.path['plot_rect_tlwh[cam_id] ']
        x, y, width, height = self.args.plot_rect_tlwh[cam_id]  #self.args.path['plot_rect_tlwh[cam_id] ']
        # x, y, width, height = 50, 50, 75, 75
        # if cam_id == 0:
            # x, y, width, height = self.args.plot_rect_tlwh[0]
        # else:
            # x, y, width, height = self.args.plot_rect_tlwh[1]
        
        # Map the point based on the dimensions of the target region
        # Instead of using the floor_plan shape. 
        # I think we should be using the shape of the image stored in the calibrate function.
        # Change floor_plan.shape to the image selected.
        
        # mapped_x = x + int(point[0] * (width / self.floor_plan.shape[0]))   # self.args.path
        # mapped_y = y + int(point[1] * (height / self.floor_plan.shape[1]))
        print("Warped image shape: ",self.warped_images[cam_id].shape)
        mapped_x = x + int(point[0] * (width / self.warped_images[cam_id].shape[0]))
        mapped_y = y + int(point[1] * (height / self.warped_images[cam_id].shape[1]))
        return int(mapped_x), int(mapped_y)            

    def plot_on_top_view(self, points, ids, cam_ids):
        print("Plot view cam ids: ",cam_ids)
        points = self.convert_list_to_bottom_center(points)
        pts = cv2.perspectiveTransform(
        np.array(points, dtype=np.float32,).reshape(1, -1, 2), self.tran_matrix[cam_ids[0]])
        # print("Inside plot top view",pts[0], ids)
        # self.floor_plan = np.zeros(((800, 1200, 3)), np.uint8)
        cv2.rectangle(self.floor_plan,(50,50), (125,125), (0, 255, 0), 2)
        cv2.rectangle(self.floor_plan,(350,50), (450,150), (0, 255, 0), 2)
        for (point,id,cam) in zip(pts[0],ids, cam_ids):
            print(point)
            if point[0] > 0 and point[1] > 0:
                x, y = self.map_to_target_region(point, cam)
                # cv2.circle(self.floor_plan, (p1, p2), 5, (0, 0, 255), -1)
                cv2.putText(self.floor_plan, str(id), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)

    def multicam(self,predictor, vis_folder, current_time, args):

        #tasks = []
        cam_id = 0
        tracker = MultiCameraTracking(self.args, frame_rate=self.args.fps)
        # vs = iter(RoundRobinVideoStream(["rtsp://admin:dataeazeDZ!@192.168.0.103/cam/realmonitor?channel=1&subtype=0&unicast=true&proto=Onvif","rtsp://admin:dataeazeDZ!@192.168.0.102/cam/realmonitor?channel=1&subtype=0&unicast=true&proto=Onvif"]))
        vs = iter(RoundRobinVideoStream(args.path))
        num_cameras = vs.get_num_cameras()
        # width, height, fps = next(vs)[1:4]
        # timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
        # save_folder = osp.join(vis_folder, timestamp)
        # os.makedirs(save_folder, exist_ok=True)
        # save_path = osp.join(save_folder, "output.mp4")
        # vid_writer = cv2.VideoWriter(
        #         save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps[0], (int(width[0]), int(height[0]))
        #     )
        vid_writers = []
        for c in range(num_cameras):
            width, height, fps = next(vs)[1:4]  # Get width, height, and fps for camera c
            timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
            save_folder = osp.join(vis_folder, timestamp, str(c))
            os.makedirs(save_folder, exist_ok=True)
            save_path = osp.join(save_folder, "campus{}.mp4".format(c))
            vid_writer = cv2.VideoWriter(
                save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps[c], (int(width[c]), int(height[c]))
            )
            vid_writers.append(vid_writer)
        grid_rows = 1  # Number of rows in the grid
        grid_cols = num_cameras # Number of columns in the grid
        grid_size = grid_rows * grid_cols 
        grid_images = [] # Total number of frames in the grid
        frame_index = 0
        for i, (frames, widths, heights, fps_vals) in enumerate(vs):
            for c in range(num_cameras):
                timer = Timer()
                frame_id = i
                results = []
                if frame_id % 20 == 0:
                    logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))
                
                outputs, img_info = predictor.inference(frames[c], timer)
                scale = min(exp.test_size[0] / float(img_info['height'], ), exp.test_size[1] / float(img_info['width']))

                if outputs[0] is not None:
                    outputs = outputs[0].cpu().numpy()
                    detections = outputs[:, :7]
                    detections[:, :4] /= scale

                    # Run tracker
                    online_targets = tracker.process(detections, img_info["raw_img"], c)

                    online_tlwhs = []
                    online_ids = []
                    online_scores = []
                    online_names = []
                    online_cam_ids = []
                    for t in online_targets:
                        print("Cam Id:", t.cam_id)
                        tlwh = t.tlwh
                        tid = t.track_id
                        vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                        if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                            online_tlwhs.append(tlwh)
                            online_ids.append(tid)
                            online_scores.append(t.score)
                            online_names.append(t.name)
                            online_cam_ids.append(t.cam_id)
                            results.append(
                                f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                            )
                    timer.toc()
                    online_im = plot_tracking(
                        img_info['raw_img'], online_tlwhs, online_ids, online_names, frame_id=frame_id + 1, fps=1. / timer.average_time
                    )

                else:
                    timer.toc()
                    online_im = img_info['raw_img']
                    
                # print("Online im: ",len(online_im))
                self.output_list.append(online_im)
                # print("Calling plot top view: ", online_tlwhs, online_ids)
                # If no ROI is selected then no top view. Also, if no rectangle is available for plotting then no top view.
                # If self.co-ordinate is empty don't plot on top view
                # Assuming self.roi_area is the variable for the selected area on the camera feed
                # self.args.plot_rect_tlwh is the variable for the co-ordinates of the rectangle on the floor plan in the form of top left and width and height.
                # self.tran_matrix for storing all the transformation matrices
                # self.warped_images for storing all the warped images.
                # if len(self.args.plot_rect_tlwh) > 0 and len(self.tran_matrix) > 0 and len(self.warped_images) > 0 and len(self.roi_area) > 0:
                # if len(self.args.plot_rect_tlwh) > 0 and len(self.args.cam_name) > 0:
                if len(self.tran_matrix) > 0 and len(self.warped_images) > 0:
                    if len(online_tlwhs) > 0 and len(online_ids)> 0:
                        print("Calling plot top view: ", online_cam_ids)
                        # self.floor_plan = np.zeros(((800, 1200, 3)), np.uint8)
                        self.plot_on_top_view(online_tlwhs, online_ids, online_cam_ids)
                    else:
                        print("Empty")

                grid_images.append(online_im)

                if len(grid_images) > grid_size:
                    # Remove the first row of frames
                    grid_images = grid_images[grid_cols:]

                if len(grid_images) >= grid_size:
                    grid_height = grid_rows * int(heights[0])
                    grid_width = grid_cols * int(widths[0])
                    grid = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)

                    for i in range(grid_size):
                        row = i // grid_cols
                        col = i % grid_cols
                        frame = grid_images[i]
                        x = col * int(widths[0])
                        y = row * int(heights[0])
                        grid[y:y+int(heights[0]), x:x+int(widths[0])] = frame

                    if grid_width > 0 and grid_height > 0:
                        print("Emitting")
                        print("Output list: ",len(self.output_list))
                        self.copy_output_list = self.output_list
                        self.frame_update.emit(1)
                        print("Emited")
                        self.output_list = []
                        cv2.imshow("Grid", self.floor_plan)
                        # cv2.waitKey(1)

                if args.save_result:
                    vid_writer = vid_writers[c]
                    vid_writer.write(online_im)
                ch = cv2.waitKey(1)
                if ch == 27 or ch == ord("q") or ch == ord("Q"):
                    break
            self.floor_plan = np.zeros(((600, 600, 3)), np.uint8)
        tracker.conn.close()
        # for video_path in args.path:
        #     task = imageflow_demo(tracker, predictor, vis_folder, current_time, args, video_path, cam_id)
        #     tasks.append(task)
        #     cam_id += 1
        # await asyncio.gather(*tasks)

def print_this():
    print("hello")

class MainApp(QMainWindow):
    def __init__(self, args, exp):
        print("Inside init")
        super().__init__()
        self.args = args
        self.args.plot_rect_tlwh = [[50, 50, 100, 100], [350, 50, 100, 100]]
        self.args.cam_name = ['campus_short0']
        self.exp = exp
        print("Creating multicam object")
        self.tracker = MultiCamTracker(self.exp, self.args)
        self.tracker.frame_update.connect(self.print_this)
        self.tracker.main()
        print("multi cam main called")
    def print_this(self, i):
        print("Inside print this")
        #print("Copy of output list: ",self.tracker.copy_output_list)
        print("hello", i)
        for image in self.tracker.copy_output_list:
            cv2.imshow("image", image)
def q_app_main(args, exp):
    print("inside q app main")
    app = QApplication(sys.argv)
    window = MainApp(args, exp)
    window.setMinimumSize(500, 500)
    window.setStyleSheet("background-color: #DBDFEA;")
    window.show()   
    print("done with q app")
    sys.exit(app.exec_())       

def calibrate(cam_id, screenshot_img, calibrate_points):
        #screenshot_img = [] # screenshot_img (The snapshot taken to mark the ROI)
        #calibrate_points = [] # The points selected to mark the ROI
        warped, temp, transformation_matrix = four_point_transform(
            screenshot_img, np.array(calibrate_points[:4], dtype="float32"), (0,0)
        )
        np.save(f"Transformation_matrices/{cam_id}.npy",transformation_matrix)
        cv2.imwrite(f"Warped_images/{cam_id}.jpg", warped)                                                                                                                                   

if __name__ == "__main__":
    print("In main")
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)
    print("args ", args)
    print("exp ", exp)

    args.ablation = False
    args.mot20 = not args.fuse_score
    print("calling q app main")
    q_app_main(args, exp)
    # yolox_tracker = MultiCamTracker(exp, args)
    # yolox_tracker.main()
    yolox_tracker.frame_update.connect(print_this)
