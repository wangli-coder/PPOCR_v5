import cv2
import numpy as np
from shapely.geometry import Polygon
import pyclipper

class DetInference(object):
    def __init__(self, args, session):
        super(DetInference).__init__()

        self.args = args
        self.unclip_ratio = 1.5
        self.max_candidates = 1000
        self.min_size = 3
        #det init
        self.det_onnx_session = session
        self.det_input_name = session.get_inputs()[0].name
        self.det_output_name = [output.name for output in session.get_outputs()]
        self.det_model_shape = list(map(int, args.det_input_shape.split(',')))
    
    def __call__(self, ori_img):

        img, ratio, (top_pad, left_pad) = self.letterbox(ori_img, new_shape=self.det_model_shape[1:])

        #det normalize & inference
        if self.args.engine == "onnx":
            img = self.normalize(img, scale=1/255., mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        img = np.expand_dims(img, axis=0).transpose((0, 3, 1, 2)).astype(np.float32)

        outputs = self.det_onnx_session.run(self.det_output_name, input_feed={self.det_input_name:img})[0]
        
        boxes, scores = self.get_det_box(outputs)
        boxes = self.filter_tag_det_res(boxes, image_shape=self.det_model_shape[1:]) #w,h
        if boxes.shape[0] == 0:
            return None
        boxes = self.reversed_box(boxes, ratio, *(top_pad, left_pad), ori_img.shape)
        boxes = self.sorted_boxes(boxes)

        return boxes

    def get_det_box(self, outputs):

        output = outputs[0, 0, :, :]
        mask = output > 0.3
        return self.boxes_from_bitmap(output, mask)

    def boxes_from_bitmap(self, output, mask, src_w=960, src_h=960):

        outs = cv2.findContours((mask * 255).astype(np.uint8), cv2.RETR_LIST,
                                cv2.CHAIN_APPROX_SIMPLE)
        if len(outs) == 3:
            img, contours, _ = outs[0], outs[1], outs[2]
        elif len(outs) == 2:
            contours, _ = outs[0], outs[1]

        num_contours = min(len(contours), self.max_candidates)

        boxes = []
        scores = []
        for index in range(num_contours):
            contour = contours[index]
            points, sside = self.get_mini_boxes(contour)
            if sside < self.min_size:
                continue
            points = np.array(points)
            score = self.box_score_fast(output, points.reshape(-1, 2))
            if score < 0.6:
                continue

            box = self.unclip(points, self.unclip_ratio).reshape(-1, 1, 2)
            box, sside = self.get_mini_boxes(box)
            if sside < self.min_size + 2:
                continue
            box = np.array(box)

            box[:, 0] = np.clip(np.round(box[:, 0]), 0, src_w)
            box[:, 1] = np.clip(np.round(box[:, 1]), 0, src_h)
            boxes.append(box.astype("int32"))
            scores.append(score)
        return np.array(boxes, dtype="int32"), scores
    
    def unclip(self, box, unclip_ratio):
        poly = Polygon(box)
        distance = poly.area * unclip_ratio / poly.length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded = np.array(offset.Execute(distance))
        return expanded

    def get_mini_boxes(self, contour):
        bounding_box = cv2.minAreaRect(contour)
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

        index_1, index_2, index_3, index_4 = 0, 1, 2, 3
        if points[1][1] > points[0][1]:
            index_1 = 0
            index_4 = 1
        else:
            index_1 = 1
            index_4 = 0
        if points[3][1] > points[2][1]:
            index_2 = 2
            index_3 = 3
        else:
            index_2 = 3
            index_3 = 2

        box = [
            points[index_1], points[index_2], points[index_3], points[index_4]
        ]
        return box, min(bounding_box[1])
    
    def box_score_fast(self, bitmap, _box):
        '''
        box_score_fast: use bbox mean score as the mean score
        '''
        h, w = bitmap.shape[:2]
        box = _box.copy()
        xmin = np.clip(np.floor(box[:, 0].min()).astype("int32"), 0, w - 1)
        xmax = np.clip(np.ceil(box[:, 0].max()).astype("int32"), 0, w - 1)
        ymin = np.clip(np.floor(box[:, 1].min()).astype("int32"), 0, h - 1)
        ymax = np.clip(np.ceil(box[:, 1].max()).astype("int32"), 0, h - 1)

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        box[:, 0] = box[:, 0] - xmin
        box[:, 1] = box[:, 1] - ymin
        cv2.fillPoly(mask, box.reshape(1, -1, 2).astype("int32"), 1)
        return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]
    
    def order_points_clockwise(self, pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        tmp = np.delete(pts, (np.argmin(s), np.argmax(s)), axis=0)
        diff = np.diff(np.array(tmp), axis=1)
        rect[1] = tmp[np.argmin(diff)]
        rect[3] = tmp[np.argmax(diff)]
        return rect

    def clip_det_res(self, points, img_height, img_width):
        for pno in range(points.shape[0]):
            points[pno, 0] = int(min(max(points[pno, 0], 0), img_width - 1))
            points[pno, 1] = int(min(max(points[pno, 1], 0), img_height - 1))
        return points

    def filter_tag_det_res(self, dt_boxes, image_shape):
        img_height, img_width = image_shape[0:2]
        dt_boxes_new = []
        for box in dt_boxes:
            if type(box) is list:
                box = np.array(box)
            box = self.order_points_clockwise(box)
            box = self.clip_det_res(box, img_height, img_width)
            rect_width = int(np.linalg.norm(box[0] - box[1]))
            rect_height = int(np.linalg.norm(box[0] - box[3]))
            if rect_width <= 3 or rect_height <= 3:
                continue
            dt_boxes_new.append(box)
        dt_boxes = np.array(dt_boxes_new)
        return dt_boxes
    
    def sorted_boxes(self, dt_boxes):
        """
        Sort text boxes in order from top to bottom, left to right
        args:
            dt_boxes(array):detected text boxes with shape [4, 2]
        return:
            sorted boxes(array) with shape [4, 2]
        """
        dt_boxes = dt_boxes.astype(np.int32)
        num_boxes = dt_boxes.shape[0]
        sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
        _boxes = list(sorted_boxes)

        for i in range(num_boxes - 1):
            for j in range(i, -1, -1):
                if abs(_boxes[j + 1][0][1] - _boxes[j][0][1]) < 10 and (
                    _boxes[j + 1][0][0] < _boxes[j][0][0]
                ):
                    tmp = _boxes[j]
                    _boxes[j] = _boxes[j + 1]
                    _boxes[j + 1] = tmp
                else:
                    break
        return _boxes

    def reversed_box(self, boxes, ratio, top_pad, left_pad, image_shape):
        
        boxes[:, :, 0] = boxes[:, :, 0] - left_pad
        boxes[:, :, 1] = boxes[:, :, 1] - top_pad
        boxes = boxes / ratio

        boxes[:, :, 0] = np.clip(boxes[:, :, 0], 0, image_shape[1])
        boxes[:, :, 1] = np.clip(boxes[:, :, 1], 0, image_shape[0])

        return boxes.astype(np.int32)

    @staticmethod
    def letterbox(im, new_shape=(960, 960), color=(0, 0, 0), auto=False, scaleFill=False, scaleup=True, stride=32):
        """Resizes and pads image to new_shape with stride-multiple constraints, returns resized image, ratio, padding."""
        shape = im.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
        elif scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)

        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border for 2 sides

        return im, ratio, (top, left)
    
    @staticmethod
    def normalize(img, scale, mean, std, order='hwc'):

        shape = (3, 1, 1) if order == 'chw' else (1, 1, 3)
        mean = np.array(mean).reshape(shape).astype('float32')
        std = np.array(std).reshape(shape).astype('float32')
        img = (img.astype('float32') * scale - mean) / std

        return img