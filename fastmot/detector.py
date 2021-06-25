from collections import defaultdict
from pathlib import Path
import configparser
import numpy as np
import numba as nb
import cv2

from . import models
from .utils import InferenceBackend
from .utils.rect import as_rect, to_tlbr, get_size, area
from .utils.rect import union, crop, multi_crop, iom, diou_nms

# YOLOV5 related
import pycuda.driver as cuda
import tensorrt as trt
import ctypes
import torch
import torchvision
import random


DET_DTYPE = np.dtype(
    [('tlbr', float, 4),
     ('label', int),
     ('conf', float)],
    align=True
)


class Detector:
    def __init__(self, size):
        self.size = size

    def __call__(self, frame):
        self.detect_async(frame)
        return self.postprocess()

    def detect_async(self, frame):
        """
        Asynchronous detection.
        """
        raise NotImplementedError

    def postprocess(self):
        """
        Synchronizes, applies postprocessing, and returns a record array
        of detections (DET_DTYPE).
        This function should be called after `detect_async`.
        """
        raise NotImplementedError


class SSDDetector(Detector):
    def __init__(self, size, config):
        super().__init__(size)
        self.label_mask = np.zeros(len(models.LABEL_MAP), dtype=bool)
        self.label_mask[list(config['class_ids'])] = True

        self.model = getattr(models, config['model'])
        self.tile_overlap = config['tile_overlap']
        self.tiling_grid = config['tiling_grid']
        self.conf_thresh = config['conf_thresh']
        self.max_area = config['max_area']
        self.merge_thresh = config['merge_thresh']

        self.batch_size = int(np.prod(self.tiling_grid))
        self.inp_stride = np.prod(self.model.INPUT_SHAPE)
        self.tiles, self.tiling_region_size = self._generate_tiles()
        self.scale_factor = np.asarray(self.size) / self.tiling_region_size
        self.backend = InferenceBackend(self.model, self.batch_size)

    def detect_async(self, frame):
        self._preprocess(frame)
        self.backend.infer_async()

    def postprocess(self):
        det_out = self.backend.synchronize()[0]
        detections, tile_ids = self._filter_dets(det_out, self.tiles, self.model.TOPK,
                                                 self.label_mask, self.max_area,
                                                 self.conf_thresh, self.scale_factor)
        detections = self._merge_dets(detections, tile_ids)
        return detections

    def _preprocess(self, frame):
        frame = cv2.resize(frame, self.tiling_region_size)
        self._normalize(frame, self.tiles, self.backend.input_handle, self.inp_stride)

    def _generate_tiles(self):
        tile_size = np.asarray(self.model.INPUT_SHAPE[:0:-1])
        tiling_grid = np.asarray(self.tiling_grid)
        step_size = (1 - self.tile_overlap) * tile_size
        total_size = (tiling_grid - 1) * step_size + tile_size
        total_size = np.rint(total_size).astype(int)
        tiles = np.array([to_tlbr((c * step_size[0], r * step_size[1], *tile_size))
                          for r in range(tiling_grid[1]) for c in range(tiling_grid[0])])
        return tiles, tuple(total_size)

    def _merge_dets(self, detections, tile_ids):
        detections = np.asarray(detections, dtype=DET_DTYPE).view(np.recarray)
        tile_ids = np.asarray(tile_ids)
        if len(detections) == 0:
            return detections
        detections = self._merge(detections, tile_ids, self.batch_size, self.merge_thresh)
        return detections.view(np.recarray)

    @staticmethod
    @nb.njit(parallel=True, fastmath=True, cache=True)
    def _normalize(frame, tiles, out, stride):
        imgs = multi_crop(frame, tiles)
        for i in nb.prange(len(imgs)):
            offset = i * stride
            bgr = imgs[i]
            # BGR to RGB
            rgb = bgr[..., ::-1]
            # HWC -> CHW
            chw = rgb.transpose(2, 0, 1)
            # Normalize to [-1.0, 1.0] interval
            normalized = chw * (2 / 255) - 1
            out[offset:offset + stride] = normalized.ravel()

    @staticmethod
    @nb.njit(fastmath=True, cache=True)
    def _filter_dets(det_out, tiles, topk, label_mask, max_area, thresh, scale_factor):
        detections = []
        tile_ids = []
        for tile_idx in range(len(tiles)):
            tile = tiles[tile_idx]
            size = get_size(tile)
            tile_offset = tile_idx * topk
            for det_idx in range(topk):
                offset = (tile_offset + det_idx) * 7
                label = int(det_out[offset + 1])
                conf = det_out[offset + 2]
                if conf < thresh:
                    break
                if label_mask[label]:
                    tl = (det_out[offset + 3:offset + 5] * size + tile[:2]) * scale_factor
                    br = (det_out[offset + 5:offset + 7] * size + tile[:2]) * scale_factor
                    tlbr = as_rect(np.append(tl, br))
                    if 0 < area(tlbr) <= max_area:
                        detections.append((tlbr, label, conf))
                        tile_ids.append(tile_idx)
        return detections, tile_ids

    @staticmethod
    @nb.njit(fastmath=True, cache=True)
    def _merge(dets, tile_ids, num_tile, thresh):
        # find duplicate neighbors across tiles
        neighbors = [[0 for _ in range(0)] for _ in range(len(dets))]
        for i, det in enumerate(dets):
            max_ioms = np.zeros(num_tile)
            for j, other in enumerate(dets):
                if tile_ids[i] != tile_ids[j] and det.label == other.label:
                    overlap = iom(det.tlbr, other.tlbr)
                    # use the detection with the greatest IoM from each tile
                    if overlap >= thresh and overlap > max_ioms[tile_ids[j]]:
                        max_ioms[tile_ids[j]] = overlap
                        neighbors[i].append(j)

        # merge neighbors using depth-first search
        keep = set(range(len(dets)))
        stack = []
        for i in range(len(dets)):
            if len(neighbors[i]) > 0 and tile_ids[i] != -1:
                tile_ids[i] = -1
                stack.append(i)
                candidates = []
                while len(stack) > 0:
                    for j in neighbors[stack.pop()]:
                        if tile_ids[j] != -1:
                            candidates.append(j)
                            tile_ids[j] = -1
                            stack.append(j)
                for k in candidates:
                    dets[i].tlbr[:] = union(dets[i].tlbr, dets[k].tlbr)
                    dets[i].conf = max(dets[i].conf, dets[k].conf)
                    keep.discard(k)
        keep = np.asarray(list(keep))
        return dets[keep]


class YOLODetector(Detector):
    def __init__(self, size, config):
        super().__init__(size)
        self.model = getattr(models, config['model'])
        self.class_ids = config['class_ids']
        self.conf_thresh = config['conf_thresh']
        self.max_area = config['max_area']
        self.nms_thresh = config['nms_thresh']

        self.backend = InferenceBackend(self.model, 1)
        self.input_handle, self.upscaled_sz, self.bbox_offset = self._create_letterbox()

    def detect_async(self, frame):
        self._preprocess(frame)
        self.backend.infer_async()

    def postprocess(self):
        det_out = self.backend.synchronize()
        det_out = np.concatenate(det_out).reshape(-1, 7)
        detections = self._filter_dets(det_out, self.upscaled_sz, self.class_ids, self.conf_thresh,
                                       self.nms_thresh, self.max_area, self.bbox_offset)
        detections = np.asarray(detections, dtype=DET_DTYPE).view(np.recarray)
        return detections

    def _preprocess(self, frame):
        frame = cv2.resize(frame, self.input_handle.shape[:0:-1])
        self._normalize(frame, self.input_handle)

    def _create_letterbox(self):
        src_size = np.asarray(self.size)
        dst_size = np.asarray(self.model.INPUT_SHAPE[:0:-1])
        if self.model.LETTERBOX:
            scale_factor = min(dst_size / src_size)
            scaled_size = np.rint(src_size * scale_factor).astype(int)
            img_offset = (dst_size - scaled_size) / 2
            insert_roi = to_tlbr(np.r_[img_offset, scaled_size])
            upscaled_sz = np.rint(dst_size / scale_factor).astype(int)
            bbox_offset = (upscaled_sz - src_size) / 2
            self.backend.input_handle = 0.5
        else:
            upscaled_sz = src_size
            insert_roi = to_tlbr(np.r_[0, 0, dst_size])
            bbox_offset = np.zeros(2)

        input_handle = self.backend.input_handle.reshape(self.model.INPUT_SHAPE)
        input_handle = crop(input_handle, insert_roi, chw=True)
        return input_handle, upscaled_sz, bbox_offset

    @staticmethod
    @nb.njit(fastmath=True, cache=True)
    def _normalize(frame, out):
        # BGR to RGB
        rgb = frame[..., ::-1]
        # HWC -> CHW
        chw = rgb.transpose(2, 0, 1)
        # Normalize to [0, 1] interval
        normalized = chw / 255.
        out[:] = normalized

    @staticmethod
    @nb.njit(fastmath=True, cache=True)
    def _filter_dets(det_out, size, class_ids, conf_thresh, nms_thresh, max_area, offset):
        """
        det_out: a list of 3 tensors, where each tensor
                 contains a multiple of 7 float32 numbers in
                 the order of [x, y, w, h, box_confidence, class_id, class_prob]
        """
        # drop detections with low score
        scores = det_out[:, 4] * det_out[:, 6]
        keep = np.where(scores >= conf_thresh)
        det_out = det_out[keep]

        # scale to pixel values
        det_out[:, :4] *= np.append(size, size)
        det_out[:, :2] -= offset

        keep = []
        for class_id in class_ids:
            class_idx = np.where(det_out[:, 5] == class_id)[0]
            class_dets = det_out[class_idx]
            class_keep = diou_nms(class_dets[:, :4], class_dets[:, 4], nms_thresh)
            keep.extend(class_idx[class_keep])
        keep = np.asarray(keep)
        nms_dets = det_out[keep]

        detections = []
        for i in range(len(nms_dets)):
            tlbr = to_tlbr(nms_dets[i, :4])
            # clip inside frame
            tlbr = np.maximum(tlbr, 0)
            tlbr = np.minimum(tlbr, np.append(size, size))
            label = int(nms_dets[i, 5])
            conf = nms_dets[i, 4] * nms_dets[i, 6]
            if 0 < area(tlbr) <= max_area:
                detections.append((tlbr, label, conf))
        return detections


class YOLOV5Detector(Detector):

    def __init__(self, size, frame_skip, config):
        super().__init__(size)
        print("creating yolov5 detector")

        self.engine_file_path = config["engine_file"]
        PLUGIN_LIBRARY = config["plugins_library"]
        ctypes.CDLL(PLUGIN_LIBRARY)
        self.CONF_THRESHOLD = config["conf_threshold"]
        self.IOU_THRESHOLD = config["iou_threshold"]

        self.categories = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
            "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
            "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
            "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
            "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
            "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
            "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
            "hair drier", "toothbrush"]

        # Create a Context on this device,
        self.ctx = cuda.Device(0).make_context()
        stream = cuda.Stream()
        TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        runtime = trt.Runtime(TRT_LOGGER)

        # Deserialize the engine from file
        with open(self.engine_file_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        context = engine.create_execution_context()

        host_inputs = []
        cuda_inputs = []
        host_outputs = []
        cuda_outputs = []
        bindings = []

        for binding in engine:
            print('bingding:', binding, engine.get_binding_shape(binding))
            size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings.
            bindings.append(int(cuda_mem))
            # Append to the appropriate list.
            if engine.binding_is_input(binding):
                self.input_w = engine.get_binding_shape(binding)[-1]
                self.input_h = engine.get_binding_shape(binding)[-2]
                host_inputs.append(host_mem)
                cuda_inputs.append(cuda_mem)
            else:
                host_outputs.append(host_mem)
                cuda_outputs.append(cuda_mem)

        # Store
        self.stream = stream
        self.context = context
        self.engine = engine
        self.host_inputs = host_inputs
        self.cuda_inputs = cuda_inputs
        self.host_outputs = host_outputs
        self.cuda_outputs = cuda_outputs
        self.bindings = bindings
        self.batch_size = engine.max_batch_size

    def detect_async(self, frame):
        # self._preprocess(frame)
        # self.backend.infer_async()
        self.infer(frame)
    
    def infer(self, frame):

        # Make self the active context, pushing it on top of the context stack.
        self.ctx.push()
        # Restore
        stream = self.stream
        context = self.context
        engine = self.engine
        host_inputs = self.host_inputs
        cuda_inputs = self.cuda_inputs
        host_outputs = self.host_outputs
        cuda_outputs = self.cuda_outputs
        bindings = self.bindings

        input_image, image_raw, origin_h, origin_w = self.preprocess_image(frame)
        # batch_image_raw.append(image_raw)
        # batch_origin_h.append(origin_h)
        # batch_origin_w.append(origin_w)
        # np.copyto(batch_input_image[i], input_image)
        
        # Copy input image to host buffer
        np.copyto(host_inputs[0], input_image.ravel())

        # Transfer input data  to the GPU.
        cuda.memcpy_htod_async(cuda_inputs[0], host_inputs[0], stream)
        # Run inference.
        context.execute_async(batch_size=self.batch_size, bindings=bindings, stream_handle=stream.handle)
        # Transfer predictions back from the GPU.
        cuda.memcpy_dtoh_async(host_outputs[0], cuda_outputs[0], stream)

        # Remove any context from the top of the context stack, deactivating it.
        self.ctx.pop()
        # Here we use the first row of output in that batch_size = 1
        output = host_outputs[0]

        self.result_boxes, self.result_scores, self.result_classid = self.post_process(
                        output[0 * 6001: (0 + 1) * 6001], origin_h, origin_w
                        )

        # # Draw rectangles and labels on the original image
        # for j in range(len(self.result_boxes)):
        #     box = self.result_boxes[j]
        #     self.plot_one_box(
        #         box,
        #         image_raw,
        #         label="{}:{:.2f}".format(
        #             self.categories[int(self.result_classid[j])], self.result_scores[j]
        #         ),
        #     )


    def post_process(self, output, origin_h, origin_w):
        """
        description: postprocess the prediction
        param:
            output:     A tensor likes [num_boxes,cx,cy,w,h,conf,cls_id, cx,cy,w,h,conf,cls_id, ...] 
            origin_h:   height of original image
            origin_w:   width of original image
        return:
            result_boxes: finally boxes, a boxes tensor, each row is a box [x1, y1, x2, y2]
            result_scores: finally scores, a tensor, each element is the score correspoing to box
            result_classid: finally classid, a tensor, each element is the classid correspoing to box
        """

        # Get the num of boxes detected
        num = int(output[0])
        # Reshape to a two dimentional ndarray
        pred = np.reshape(output[1:], (-1, 6))[:num, :]
        # to a torch Tensor
        pred = torch.Tensor(pred).cuda()
        # Get the boxes
        boxes = pred[:, :4]
        # Get the scores
        scores = pred[:, 4]
        # Get the classid
        classid = pred[:, 5]
        # Choose those boxes that score > CONF_THRESH
        si = scores > self.CONF_THRESHOLD
       
        boxes = boxes[si, :]
        scores = scores[si]
        classid = classid[si]
        
        # Transform bbox from [center_x, center_y, w, h] to [x1, y1, x2, y2]
        boxes = self.xywh2xyxy(origin_h, origin_w, boxes)
        # Do nms
        indices = torchvision.ops.nms(boxes, scores, iou_threshold=self.IOU_THRESHOLD).cpu()
        result_boxes = boxes[indices, :].cpu()
        result_scores = scores[indices].cpu()
        result_classid = classid[indices].cpu()
        return result_boxes, result_scores, result_classid


    def xywh2xyxy(self, origin_h, origin_w, x):
        """
        description:    Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        param:
            origin_h:   height of original image
            origin_w:   width of original image
            x:          A boxes tensor, each row is a box [center_x, center_y, w, h]
        return:
            y:          A boxes tensor, each row is a box [x1, y1, x2, y2]
        """
        y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
        r_w = self.input_w / origin_w
        r_h = self.input_h / origin_h
        if r_h > r_w:
            y[:, 0] = x[:, 0] - x[:, 2] / 2
            y[:, 2] = x[:, 0] + x[:, 2] / 2
            y[:, 1] = x[:, 1] - x[:, 3] / 2 - (self.input_h - r_w * origin_h) / 2
            y[:, 3] = x[:, 1] + x[:, 3] / 2 - (self.input_h - r_w * origin_h) / 2
            y /= r_w
        else:
            y[:, 0] = x[:, 0] - x[:, 2] / 2 - (self.input_w - r_h * origin_w) / 2
            y[:, 2] = x[:, 0] + x[:, 2] / 2 - (self.input_w - r_h * origin_w) / 2
            y[:, 1] = x[:, 1] - x[:, 3] / 2
            y[:, 3] = x[:, 1] + x[:, 3] / 2
            y /= r_h

        return y


    def preprocess_image(self, raw_bgr_image):
        """
        description: Convert BGR image to RGB,
                     resize and pad it to target size, normalize to [0,1],
                     transform to NCHW format.
        param:
            input_image_path: str, image path
        return:
            image:  the processed image
            image_raw: the original image
            h: original height
            w: original width
        """
        image_raw = raw_bgr_image
        h, w, c = image_raw.shape
        image = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)

        # Calculate widht and height and paddings
        r_w = self.input_w / w
        r_h = self.input_h / h
        if r_h > r_w:
            tw = self.input_w
            th = int(r_w * h)
            tx1 = tx2 = 0
            ty1 = int((self.input_h - th) / 2)
            ty2 = self.input_h - th - ty1
        else:
            tw = int(r_h * w)
            th = self.input_h
            tx1 = int((self.input_w - tw) / 2)
            tx2 = self.input_w - tw - tx1
            ty1 = ty2 = 0
        # Resize the image with long side while maintaining ratio
        image = cv2.resize(image, (tw, th))
        
        # Pad the short side with (128,128,128)
        image = cv2.copyMakeBorder(
            image, ty1, ty2, tx1, tx2, cv2.BORDER_CONSTANT, (128, 128, 128)
        )
        image = image.astype(np.float32)
        # Normalize to [0,1]
        image /= 255.0
        # HWC to CHW format:
        image = np.transpose(image, [2, 0, 1])
        # CHW to NCHW format
        image = np.expand_dims(image, axis=0)
        # Convert the image to row-major order, also known as "C order":
        image = np.ascontiguousarray(image)
        return image, image_raw, h, w

    
    # def plot_one_box(self, x, img, color=None, label=None, line_thickness=None):
    #     """
    #     description: Plots one bounding box on image img,
    #                 this function comes from YoLov5 project.
    #     param: 
    #         x:      a box likes [x1,y1,x2,y2]
    #         img:    a opencv image object
    #         color:  color to draw rectangle, such as (0,255,0)
    #         label:  str
    #         line_thickness: int
    #     return:
    #         no return

    #     """
    #     tl = (
    #         line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    #     )  # line/font thickness
    #     color = color or [random.randint(0, 255) for _ in range(3)]
    #     c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    #     cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    #     if label:
    #         tf = max(tl - 1, 1)  # font thickness
            
    #         t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]

    #         c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3

    #         cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
    #         cv2.putText(
    #             img,
    #             label,
    #             (c1[0], c1[1] - 2),
    #             0,
    #             tl / 3,
    #             [225, 255, 255],
    #             thickness=tf,
    #             lineType=cv2.LINE_AA,
    #         )

    def postprocess(self):
        detections = list(zip(self.result_boxes, self.result_classid, self.result_scores))
        detections = np.asarray(detections, dtype=DET_DTYPE).view(np.recarray)
        return detections

    def destroy(self):
        # Remove any context from the top of the context stack, deactivating it.
        print("destroying yolov5..")
        self.ctx.pop()


class PublicDetector(Detector):
    def __init__(self, size, frame_skip, config):
        super().__init__(size)
        self.frame_skip = frame_skip
        self.seq_root = Path(__file__).parents[1] / config['sequence']
        self.conf_thresh = config['conf_thresh']
        self.max_area = config['max_area']

        seqinfo = configparser.ConfigParser()
        seqinfo.read(self.seq_root / 'seqinfo.ini')
        self.seq_size = (int(seqinfo['Sequence']['imWidth']), int(seqinfo['Sequence']['imHeight']))

        self.detections = defaultdict(list)
        self.frame_id = 0

        det_txt = self.seq_root / 'det' / 'det.txt'
        for mot_det in np.loadtxt(det_txt, delimiter=','):
            frame_id = int(mot_det[0]) - 1
            tlbr = to_tlbr(mot_det[2:6])
            conf = 1.0 # mot_det[6]
            label = 1 # mot_det[7] (person)
            # scale and clip inside frame
            tlbr[:2] = tlbr[:2] / self.seq_size * self.size
            tlbr[2:] = tlbr[2:] / self.seq_size * self.size
            tlbr = np.maximum(tlbr, 0)
            tlbr = np.minimum(tlbr, np.append(self.size, self.size))
            tlbr = as_rect(tlbr)
            if conf >= self.conf_thresh and area(tlbr) <= self.max_area:
                self.detections[frame_id].append((tlbr, label, conf))

    def detect_async(self, frame):
        pass

    def postprocess(self):
        detections = np.asarray(self.detections[self.frame_id], dtype=DET_DTYPE).view(np.recarray)
        self.frame_id += self.frame_skip
        return detections
