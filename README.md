# PPOCRv5
PPOCRv5 DEMO on Axera

- 目前支持  Python 语言 
- 预编译模型下载[models](https://github.com/wzf19947/PPOCR_v5/releases/download/v1.0.0/model.tar.gz)。如需自行转换请参考[模型转换](/model_convert/README.md)

## 支持平台

- [x] AX650N
- [ ] AX630C

## 模型转换

[模型转换](./model_convert/README.md)

## 上板部署

- AX650N 的设备已预装 Ubuntu22.04
- 以 root 权限登陆 AX650N 的板卡设备
- 链接互联网，确保 AX650N 的设备能正常执行 `apt install`, `pip install` 等指令
- 已验证设备：AX650N DEMO Board

### Python API 运行

#### Requirements

```
cd python
pip3 install -r requirements.txt
``` 

#### 运行

##### 基于 ONNX Runtime 运行  
可在开发板或PC运行 

在开发板或PC上，运行以下命令  
```  
cd python
python3 infer_onnx.py
```
输出结果
![output](asserts/res_onnx.jpg)

##### 基于AXEngine运行  
在开发板上运行命令

```
cd python  
python3 infer_axmodel.py
```  
输出结果
![output](asserts/res_ax.jpg)


运行参数说明:  
| 参数名称 | 说明  |
| --- | --- | 
| --img_path | 输入图片路径 | 
| --det_model_dir | 检测模型路径 | 
| --rec_model_dir | 识别模型路径 | 
| --cls_model_dir | 分类模型路径 | 
| --character_dict_path | 识别字典路径 | 
| --det_limit_side_len | 检测模型尺寸 | 
| --rec_image_shape | 识别模型尺寸 | 
| --cls_image_shape | 分类模型尺寸 | 

### Latency

#### AX650N

| model | latency(ms) |
|---|---|
|PP-OCRv5_mobile_det|28.6|
|PP-OCRv5_mobile_rec|3.6|
|PP-LCNet_x0_25_textline_ori|0.3|



## 技术讨论

- Github issues
- QQ 群: 139953715
