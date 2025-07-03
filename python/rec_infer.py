import cv2
import numpy as np
import math

class RecInference(object):
    def __init__(self, args, session):
        super(RecInference).__init__()

        self.args = args
        #cls init
        self.rec_onnx_session = session
        self.rec_input_name = session.get_inputs()[0].name
        self.rec_output_name = [output.name for output in session.get_outputs()]

        self.rec_batch_num = 1
        self.rec_model_shape = list(map(int, args.rec_input_shape.split(',')))

    def __call__(self, img_list):
        img_num = len(img_list)
        # Calculate the aspect ratio of all text bars
        width_list = []
        for img in img_list:
            width_list.append(img.shape[1] / float(img.shape[0]))
        # Sorting can speed up the recognition process
        indices = np.argsort(np.array(width_list))
        rec_res = [["", 0.0]] * img_num
        batch_num = self.rec_batch_num

        for beg_img_no in range(0, img_num, batch_num):
            end_img_no = min(img_num, beg_img_no + batch_num)
            norm_img_batch = []
            imgC, imgH, imgW = self.rec_model_shape[:3]
            max_wh_ratio = imgW / imgH

            for ino in range(beg_img_no, end_img_no):
                norm_img = self.resize_norm_img(img_list[indices[ino]], max_wh_ratio)
                norm_img = norm_img[np.newaxis, :]
                norm_img_batch.append(norm_img)

            norm_img_batch = np.concatenate(norm_img_batch)
            norm_img_batch = norm_img_batch.copy()

            preds = self.rec_onnx_session.run(
                self.rec_output_name, input_feed={self.rec_input_name:norm_img_batch}
            )[0]

            rec_result = CTCLabelDecode(preds, self.args.rec_char_dict_path)
            for rno in range(len(rec_result)):
                rec_res[indices[beg_img_no + rno]] = rec_result[rno]

        return rec_res
    
    def resize_norm_img(self, img, max_wh_ratio):
        imgC, imgH, imgW = self.rec_model_shape
        assert imgC == img.shape[2]
        imgW = int((imgH * max_wh_ratio))

        h, w = img.shape[:2]
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

def CTCLabelDecode(preds, character_dict_path, is_remove_duplicate=True):
    """Convert between text-label and text-index"""
    if isinstance(preds, tuple) or isinstance(preds, list):
        preds = preds[-1]

    character_str = ["blank"]
    with open(character_dict_path, "rb") as fin:
        lines = fin.readlines()
        for line in lines:
            line = line.decode("utf-8").strip("\n").strip("\r\n")
            character_str.append(line)
    character_str.append(" ")
    dict_character = {}
    for i, char in enumerate(character_str):
        dict_character[i] = char

    preds_idx = preds.argmax(axis=2)
    preds_prob = preds.max(axis=2)

    result_list = []
    ignored_tokens = [0]
    batch_size = len(preds_idx)
    for batch_idx in range(batch_size):
        selection = np.ones(len(preds_idx[batch_idx]), dtype=bool)
        if is_remove_duplicate:
            selection[1:] = preds_idx[batch_idx][1:] != preds_idx[batch_idx][:-1]
        for ignored_token in ignored_tokens:
            selection &= preds_idx[batch_idx] != ignored_token
        
        char_list = [
            dict_character[text_id] for text_id in preds_idx[batch_idx][selection]
        ]
        if preds_prob is not None:
            conf_list = preds_prob[batch_idx][selection]
        else:
            conf_list = [1] * len(selection)
        if len(conf_list) == 0:
            conf_list = [0]

        text = "".join(char_list)

        result_list.append((text, np.mean(conf_list).tolist()))
    return result_list