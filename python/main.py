import os
import math
import copy
import argparse
from glob import glob
import cv2
import numpy as np
import pyclipper
from shapely.geometry import Polygon
from utils import sav2Img
from det_infer import DetInference
from cls_infer import ClsInference
from rec_infer import RecInference

def main(opt):

    imgs = list()
    if os.path.isfile(opt.source):
        imgs.append(opt.source)
    elif os.path.isdir(opt.source):
        imgs.extend(glob(opt.source+'/*'))
    else:
        print("GET ERROR SOURCE.")

    if opt.engine == "onnx":
        import onnxruntime as engine
        providers=['CPUExecutionProvider']
    elif opt.engine == "ax":
        import axengine as engine
        providers=['AxEngineExecutionProvider']
    else:
        print(f"The {opt.engine} engine not support.")

    det_session = engine.InferenceSession(opt.det_model, providers=providers)
    cls_session = engine.InferenceSession(opt.cls_model, providers=providers)
    rec_session = engine.InferenceSession(opt.rec_model, providers=providers)

    for img in imgs:
        ori_img = cv2.imread(img)
        text_det = DetInference(opt, session=det_session)
        det_boxes = text_det(ori_img)

        #get text img direction and refine text img
        text_cls = ClsInference(opt, session=cls_session)
        img_list, _ = text_cls(det_boxes, ori_img)

        #get rec results
        text_rec = RecInference(opt, session=rec_session)
        rec_results = text_rec(img_list)

        #filter low quality result
        filter_result = []
        for box, rec_result in zip(det_boxes, rec_results):
            _, score = rec_result
            if score >= 0.5:
                filter_result.append([box.tolist(), rec_result])
                print(f"text result: {rec_result}")

        save_name = os.path.basename(img).split(".")[0]
        os.makedirs(f"./results", exist_ok=True)
        sav2Img(ori_img, filter_result, name=f"./results/{save_name}_{opt.engine}_result.jpg")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--det_model", type=str, default="../models/det_mobile_slim_static.onnx", help="Det model path, onnx model or axmodel.")
    parser.add_argument("--cls_model", type=str, default="../models/cls_x0_25_slim_static.onnx", help="Cls model path, onnx model or axmodel.")
    parser.add_argument("--rec_model", type=str, default="../models/rec_mobile_slim_static.onnx", help="Rec model path, onnx model or axmodel.")
    parser.add_argument("--det_input_shape", type=list, default=[3, 960, 960], help="Det model input shape.")
    parser.add_argument("--cls_input_shape", type=list, default=[3, 80, 160], help="Cls model input shape.")
    parser.add_argument("--rec_input_shape", type=list, default=[3, 48, 320], help="Rec model input shape.")

    #config
    parser.add_argument("--source", type=str, default="./test_images", help="Input image path, filename or image name.")
    parser.add_argument("--engine", type=str, default="onnx", help="Inference engine, onnxruntime or axengine.")
    parser.add_argument("--rec_char_dict_path", type=str, default="./fonts/ppocrv5_dict.txt", help="Rec char dict file.")
    
    opt = parser.parse_args()
    main(opt)