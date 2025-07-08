# 模型转换

## 创建虚拟环境

```
conda create -n PPOCR python=3.12 -y
conda activate PPOCR
```
建议创建python虚拟环境， 需要使用到的库如下：
- [paddle2onnx](https://github.com/PaddlePaddle/Paddle2ONNX)
- onnx-simplifier
- onnx
- [pulsar2](https://github.com/AXERA-TECH/pulsar2-docs)(AXmodel转换工具链)

## 导出模型（Paddle -> ONNX）
下载相关PPOCRv5的相关推理模型：
- [文字检测模型](https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/version3.x/module_usage/text_detection.md)
- [文本方向分类模型](https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/version3.x/module_usage/textline_orientation_classification.md)
- [文字识别模型](https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/version3.x/module_usage/text_recognition.md)

用以下命令将相关模型导出为对应的onnx模型：
```
paddle2onnx --model_dir [your model dir] --model_filename inference.json --params_filename inference.pdiparams --save_file [your save onnx name] --opset_version 11 --enable_onnx_checker True

#example：det model
paddle2onnx --model_dir ./PP-OCRv5_mobile_det_infer --model_filename inference.json --params_filename inference.pdiparams --save_file ./det_mobile.onnx --opset_version 11 --enable_onnx_checker True
```
导出成功后会生成'det_mobile.onnx'、'rec_mobile.onnx'、'cls_mobile.onnx'三个模型。

## 动态onnx转静态
使用onnxsim对相关导出的onnx模型进行优化，并转为静态输入(输入维度仅参考)。
```
onnxsim det_mobile.onnx  det_mobile_sim_static.onnx --overwrite-input-shape=1,3,960,960
onnxsim rec_mobile.onnx  rec_mobile_sim_static.onnx --overwrite-input-shape=1,3,48,320
onnxsim cls_mobile.onnx  cls_x0_25_slim_static.onnx --overwrite-input-shape=1,3,80,160
```

## 转换模型（ONNX -> Axera）
使用模型转换工具 `Pulsar2` 将 ONNX 模型转换成适用于 Axera 的 NPU 运行的模型文件格式 `.axmodel`，通常情况下需要经过以下两个步骤：

- 生成适用于该模型的 PTQ 量化校准数据集
- 使用 `Pulsar2 build` 命令集进行模型转换（PTQ 量化、编译），更详细的使用说明请参考 [AXera Pulsar2 工具链指导手册](https://pulsar2-docs.readthedocs.io/zh-cn/latest/index.html) 

### 模型转换

#### 修改配置文件

- 下载相关量化数据集[dataset](https://github.com/wangli-coder/PPOCR_v5/releases/download/V1.0.0/dataset.zip)并解压

- 配置`config.json` 中 `calibration_dataset` 字段数为对用量化据集路径

#### Pulsar2 build

参考命令如下(以650n为例)：

```
pulsar2 build --input det_mobile_sim_static.onnx --config ./ppdet.json --output_dir ./det --output_name ppocrv5_det.axmodel  --target_hardware AX650 --compiler.check 0

也可将参数写进json中，直接执行：
pulsar2 build --config ./ppdet_650n.json
pulsar2 build --config ./pprec_650n.json
pulsar2 build --config ./ppcls_650n.json
```
