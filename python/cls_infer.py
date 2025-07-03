import cv2
import numpy as np
import math
import copy

class ClsInference(object):
    def __init__(self, args, session):
        super(ClsInference).__init__()

        self.args = args
        #cls init
        self.cls_onnx_session = session
        self.cls_input_name = session.get_inputs()[0].name
        self.cls_output_name = [output.name for output in session.get_outputs()]

        self.label_list = ["0", "180"]
        self.cls_batch_num = 1
        self.cls_model_shape = list(map(int, args.cls_input_shape.split(',')))

    def __call__(self, det_boxes, img):

        img_crop_list = self.get_rotate_crop_image(det_boxes, img)

        img_list = copy.deepcopy(img_crop_list)
        img_num = len(img_list)
        # Calculate the aspect ratio of all text bars
        width_list = []
        for img in img_list:
            width_list.append(img.shape[1] / float(img.shape[0]))
        # Sorting can speed up the cls process
        indices = np.argsort(np.array(width_list))

        cls_res = [["", 0.0]] * img_num
        batch_num = self.cls_batch_num

        for beg_img_no in range(0, img_num, batch_num):

            end_img_no = min(img_num, beg_img_no + batch_num)
            norm_img_batch = []

            for ino in range(beg_img_no, end_img_no):
                norm_img = self.resize_norm_img(img_list[indices[ino]])
                norm_img = norm_img[np.newaxis, :]
                norm_img_batch.append(norm_img)
            norm_img_batch = np.concatenate(norm_img_batch)
            norm_img_batch = norm_img_batch.copy()

            outputs = self.cls_onnx_session.run(
                self.cls_output_name, input_feed={self.cls_input_name:norm_img_batch}
            )

            prob_out = outputs[0]

            cls_result = self.postprocess(prob_out)
            for rno in range(len(cls_result)):
                label, score = cls_result[rno]
                cls_res[indices[beg_img_no + rno]] = [label, score]
                if "180" in label and score > 0.9:
                    img_list[indices[beg_img_no + rno]] = cv2.rotate(
                        img_list[indices[beg_img_no + rno]], 1
                    )
        return img_list, cls_res

    def resize_norm_img(self, img):
        imgC, imgH, imgW = self.cls_model_shape
        h = img.shape[0]
        w = img.shape[1]
        ratio = w / float(h)
        if math.ceil(imgH * ratio) > imgW:
            resized_w = imgW
        else:
            resized_w = int(math.ceil(imgH * ratio))
        resized_image = cv2.resize(img, (resized_w, imgH))
        resized_image = resized_image.astype("float32")
        resized_image = resized_image.transpose((2, 0, 1))

        if self.args.engine == "onnx":
            resized_image /= 255.
            resized_image -= 0.5
            resized_image /= 0.5

        padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
        padding_im[:, :, 0:resized_w] = resized_image
        return padding_im
    
    def postprocess(self, preds):

        label_list = self.label_list
        pred_idxs = preds.argmax(axis=1)
        decode_out = [(label_list[idx], preds[i, idx])
                      for i, idx in enumerate(pred_idxs)]
        
        return decode_out

    @staticmethod
    def get_rotate_crop_image(det_boxes, img):
        """
        img_height, img_width = img.shape[0:2]
        left = int(np.min(points[:, 0]))
        right = int(np.max(points[:, 0]))
        top = int(np.min(points[:, 1]))
        bottom = int(np.max(points[:, 1]))
        img_crop = img[top:bottom, left:right, :].copy()
        points[:, 0] = points[:, 0] - left
        points[:, 1] = points[:, 1] - top
        """
        img_crop_list = []

        # 图片裁剪
        for bno in range(len(det_boxes)):
            points = det_boxes[bno].astype(np.float32)

            assert len(points) == 4, "shape of points must be 4*2"
            img_crop_width = int(
                max(
                    np.linalg.norm(points[0] - points[1]), np.linalg.norm(points[2] - points[3])
                )
            )
            img_crop_height = int(
                max(
                    np.linalg.norm(points[0] - points[3]), np.linalg.norm(points[1] - points[2])
                )
            )
            pts_std = np.float32(
                [
                    [0, 0],
                    [img_crop_width, 0],
                    [img_crop_width, img_crop_height],
                    [0, img_crop_height],
                ]
            )

            M = cv2.getPerspectiveTransform(points, pts_std)
            dst_img = cv2.warpPerspective(
                img,
                M,
                (img_crop_width, img_crop_height),
                borderMode=cv2.BORDER_REPLICATE,
                flags=cv2.INTER_CUBIC,
            )
            dst_img_height, dst_img_width = dst_img.shape[0:2]
            if dst_img_height * 1.0 / dst_img_width >= 1.5:
                dst_img = np.rot90(dst_img)

            img_crop_list.append(dst_img)

        return img_crop_list