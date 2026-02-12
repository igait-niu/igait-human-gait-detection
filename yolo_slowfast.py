import torch
import numpy as np
import os,cv2,time,torch,random,pytorchvideo,warnings,argparse,math,json
warnings.filterwarnings("ignore",category=UserWarning)

from pytorchvideo.transforms.functional import (
    uniform_temporal_subsample,
    short_side_scale_with_boxes,
    clip_boxes_to_image,)
from torchvision.transforms._functional_video import normalize
from pytorchvideo.data.ava import AvaLabeledVideoFramePaths
from pytorchvideo.models.hub import slowfast_r50_detection
from deep_sort.deep_sort import DeepSort
from pathlib import Path
import sys

class MyVideoCapture:
    """
    Wrapper around cv2.VideoCapture, used for 
    - Reads frames (read()).
    - Stores frames (stack).
    - Converts frames to PyTorch tensors (to_tensor).
    - Bundles frames into a clip for action recognition (get_video_clip).
    - Releases the video stream (release).
    """
    def __init__(self, source):
        self.cap = cv2.VideoCapture(source)
        self.idx = -1
        self.end = False
        self.stack = []
        
    def read(self):
        self.idx += 1
        ret, img = self.cap.read()
        if ret:
            self.stack.append(img)
        else:
            self.end = True
        return ret, img
    
    def to_tensor(self, img):
        img = torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        return img.unsqueeze(0)
        
    def get_video_clip(self):
        assert len(self.stack) > 0, "clip length must large than 0 !"
        self.stack = [self.to_tensor(img) for img in self.stack]
        clip = torch.cat(self.stack).permute(-1, 0, 1, 2)
        del self.stack
        self.stack = []
        return clip
    
    def release(self):
        self.cap.release()
        
def tensor_to_numpy(tensor):
    img = tensor.cpu().numpy().transpose((1, 2, 0))
    return img

def ava_inference_transform(
    clip, 
    boxes,
    num_frames = 32, #if using slowfast_r50_detection, change this to 32, 4 for slow 
    crop_size = 640, 
    data_mean = [0.45, 0.45, 0.45], 
    data_std = [0.225, 0.225, 0.225],
    slow_fast_alpha = 4, #if using slowfast_r50_detection, change this to 4, None for slow
):
    """
    Prepares a video clip for the SlowFast action recognition model
    """
    
    boxes = np.array(boxes)
    roi_boxes = boxes.copy()
    clip = uniform_temporal_subsample(clip, num_frames)
    clip = clip.float()
    clip = clip / 255.0
    height, width = clip.shape[2], clip.shape[3]
    boxes = clip_boxes_to_image(boxes, height, width)
    clip, boxes = short_side_scale_with_boxes(clip,size=crop_size,boxes=boxes,)
    clip = normalize(clip,
        np.array(data_mean, dtype=np.float32),
        np.array(data_std, dtype=np.float32),) 
    boxes = clip_boxes_to_image(boxes, clip.shape[2],  clip.shape[3])
    if slow_fast_alpha is not None:
        fast_pathway = clip
        slow_pathway = torch.index_select(clip,1,
            torch.linspace(0, clip.shape[1] - 1, clip.shape[1] // slow_fast_alpha).long())
        clip = [slow_pathway, fast_pathway]
    
    return clip, torch.from_numpy(boxes), roi_boxes

def plot_one_box(x, img, color=[100,100,100], text_info="None",
                 velocity=None, thickness=1, fontsize=0.5, fontthickness=1):
    """
    Plots one bounding box on image img, and overlays a readable text label on top
    """
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness, lineType=cv2.LINE_AA)
    t_size = cv2.getTextSize(text_info, cv2.FONT_HERSHEY_TRIPLEX, fontsize , fontthickness+2)[0]
    cv2.rectangle(img, c1, (c1[0] + int(t_size[0]), c1[1] + int(t_size[1]*1.45)), color, -1)
    cv2.putText(img, text_info, (c1[0], c1[1]+t_size[1]+2), 
                cv2.FONT_HERSHEY_TRIPLEX, fontsize, [255,255,255], fontthickness)
    return img

def deepsort_update(Tracker, pred, xywh, np_img):
    """
    Wwrapper around the DeepSORT tracker that updates tracked objects in a video frame
    """
    outputs = Tracker.update(xywh, pred[:,4:5],pred[:,5].tolist(),cv2.cvtColor(np_img,cv2.COLOR_BGR2RGB))
    return outputs

def save_yolopreds_tovideo(yolo_preds, id_to_ava_labels, color_map, output_video, vis=False):
    """
    Draws YOLO detections and action labels on each frame and saves them to a video
    """
    for i, (im, pred) in enumerate(zip(yolo_preds.ims, yolo_preds.pred)):
        im=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
        if pred.shape[0]:
            for j, (*box, cls, trackid, vx, vy) in enumerate(pred):
                if int(cls) != 0:
                    ava_label = ''
                elif trackid in id_to_ava_labels.keys():
                    ava_label = id_to_ava_labels[trackid].split(' ')[0]
                else:
                    ava_label = 'Unknow'
                text = '{} {} {}'.format(int(trackid),yolo_preds.names[int(cls)],ava_label)
                color = color_map[int(cls)]
                im = plot_one_box(box,im,color,text)
        im = im.astype(np.uint8)
        output_video.write(im)
        if vis:
            im=cv2.cvtColor(im,cv2.COLOR_RGB2BGR)
            cv2.imshow("demo", im)

def check_one_person_walking(pred, id_to_ava_labels):
    """
    Identifys if a single person is walking in a video
    """
    person_count = sum(1 for (*_, cls, _, _, _) in pred if int(cls) == 0)
    if person_count != 1:
        return False
        
    # For the one person, check if they are walking
    for (*_, cls, trackid, _, _) in pred:
        if int(cls) == 0:  # This is the one person
            if trackid in id_to_ava_labels:
                action = id_to_ava_labels[trackid].split(' ')[0]
                return action == 'walk'
    return False

def main(config):
    device = config.device
    imsize = config.imsize

    detect_walk_only = (getattr(config, "mode", "full") == "walk")
    max_seconds = getattr(config, "max_seconds", None)
    output_json_path = getattr(config, "output_json", None)

    model = torch.hub.load('ultralytics/yolov5', 'yolov5l6').to(device) 
    model.conf = config.conf
    model.iou = config.iou
    model.max_det = 100
    if config.classes:
        model.classes = config.classes
    
    video_model = slowfast_r50_detection(True).eval().to(device)
    
    deepsort_tracker = DeepSort("deep_sort/deep_sort/deep/checkpoint/ckpt.t7")
    ava_labelnames,_ = AvaLabeledVideoFramePaths.read_label_map("selfutils/temp.pbtxt")
    coco_color_map = [[random.randint(0, 255) for _ in range(3)] for _ in range(80)]

    vide_save_path = config.output
    video=cv2.VideoCapture(config.input)
    width,height = int(video.get(3)),int(video.get(4))
    video.release()
    outputvideo = cv2.VideoWriter(vide_save_path,cv2.VideoWriter_fourcc(*'mp4v'), 25, (width,height))
    print("processing...")
    
    cap = MyVideoCapture(config.input)
    id_to_ava_labels = {}
    a=time.time()
    one_person_walking = False
    human_detected = False
    clips_processed = 0
    clips_with_person = 0
    clips_with_walking = 0
    
    while not cap.end:
        ret, img = cap.read()
        if not ret:
            continue

        # Limit processing time
        if max_seconds is not None and cap.idx / 25 >= max_seconds:
            print(f"Reached max processing time of {max_seconds} seconds.")
            break

        yolo_preds=model([img], size=imsize)
        yolo_preds.files=["img.jpg"]
        
        deepsort_outputs=[]
        for j in range(len(yolo_preds.pred)):
            temp=deepsort_update(deepsort_tracker,yolo_preds.pred[j].cpu(),yolo_preds.xywh[j][:,0:4].cpu(),yolo_preds.ims[j])
            if len(temp)==0:
                temp=np.ones((0,8))
            deepsort_outputs.append(temp.astype(np.float32))
            
        yolo_preds.pred=deepsort_outputs
        
        if len(cap.stack) == 25:
            print(f"processing {cap.idx // 25}th second clips")
            clip = cap.get_video_clip()
            clips_processed += 1
            if yolo_preds.pred[0].shape[0]:
                inputs, inp_boxes, _=ava_inference_transform(clip, yolo_preds.pred[0][:,0:4], crop_size=imsize)
                inp_boxes = torch.cat([torch.zeros(inp_boxes.shape[0],1), inp_boxes], dim=1)
                if isinstance(inputs, list):
                    inputs = [inp.unsqueeze(0).to(device) for inp in inputs]
                else:
                    inputs = inputs.unsqueeze(0).to(device)
                with torch.no_grad():
                    slowfaster_preds = video_model(inputs, inp_boxes.to(device))
                    slowfaster_preds = slowfaster_preds.cpu()
                for tid,avalabel in zip(yolo_preds.pred[0][:,5].tolist(), np.argmax(slowfaster_preds, axis=1).tolist()):
                    id_to_ava_labels[tid] = ava_labelnames[avalabel+1]

        # Track person detection and walking status
        if len(yolo_preds.pred) > 0 and len(yolo_preds.pred[0]) > 0:
            person_count = sum(1 for (*_, cls, _, _, _) in yolo_preds.pred[0] if int(cls) == 0)
            if person_count > 0:
                human_detected = True
                if cap.idx % 25 == 0:
                    clips_with_person += 1
            frame_walking = check_one_person_walking(yolo_preds.pred[0], id_to_ava_labels)
            if frame_walking:
                one_person_walking = True
                if cap.idx % 25 == 0:
                    clips_with_walking += 1
                print(f"Walking detected at frame {cap.idx}")
            # Return early if found and in walk mode
            if one_person_walking and detect_walk_only:
                _write_validity_json(output_json_path, True, human_detected, True,
                                     cap.idx + 1, clips_processed, clips_with_person,
                                     clips_with_walking, time.time() - a)
                sys.exit(0)
        
        save_yolopreds_tovideo(yolo_preds, id_to_ava_labels, coco_color_map, outputvideo, config.show)
    
    processing_time = time.time() - a
    total_frames = cap.idx + 1
    print("Total cost: {:.3f} s, video length: {} s".format(processing_time, cap.idx / 25))
    
    cap.release()
    outputvideo.release()
    print('Saved video to:', vide_save_path)
    
    # Write validity JSON results
    _write_validity_json(output_json_path, one_person_walking, human_detected,
                         one_person_walking, total_frames, clips_processed,
                         clips_with_person, clips_with_walking, processing_time)
    
    # Return whether we detected one person walking in the video
    if one_person_walking:
        print("Person walking detected")
        sys.exit(0)
    else:
        print("Person walking not detected")
        sys.exit(1)

def _write_validity_json(path, valid, human_detected, walking_detected,
                         total_frames, clips_processed, clips_with_person,
                         clips_with_walking, processing_time):
    """Write structured validity results to a JSON file."""
    if path is None:
        return
    result = {
        "valid": valid,
        "human_detected": human_detected,
        "walking_detected": walking_detected,
        "total_frames": total_frames,
        "clips_processed": clips_processed,
        "clips_with_person": clips_with_person,
        "clips_with_walking": clips_with_walking,
        "processing_time_seconds": round(processing_time, 3),
    }
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Validity results written to: {path}")

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default="/home/wufan/images/video/vad.mp4", help='test imgs folder or video or camera')
    parser.add_argument('--output', type=str, default="/output/output.mp4", help='folder to save result imgs, can not use input folder')
    # object detect config
    parser.add_argument('--imsize', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou', type=float, default=0.4, help='IOU threshold for NMS')
    parser.add_argument('--device', default='cuda', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--show', action='store_true', help='show img')
    # different mode config
    parser.add_argument('--max-seconds', type=int, default=None, help='Maximum number of seconds to process from the video')
    parser.add_argument('--mode', type=str, default='full', choices=['full', 'walk'], help='Run mode: "full" for normal, "walk" for human walking detection only')
    parser.add_argument('--output-json', type=str, default=None, dest='output_json', help='Path to write validity results JSON')
    config = parser.parse_args()

    if config.input.isdigit():
        print("using local camera.")
        config.input = int(config.input)

    print(config)
    main(config)