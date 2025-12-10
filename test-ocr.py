from paddleocr import PaddleOCR
# 初始化 PaddleOCR 实例
ocr = PaddleOCR(
    use_doc_orientation_classify=False, # 通过 use_doc_orientation_classify 参数指定不使用文档方向分类模型
    use_doc_unwarping=False, # 通过 use_doc_unwarping 参数指定不使用文本图像矫正模型
    use_textline_orientation=False, # 通过 use_textline_orientation 参数指定不使用文本行方向分类模型
    ocr_version="PP-OCRv5",
    device="gpu",
    lang="en",
    text_detection_model_name="PP-OCRv5_server_det",
    text_detection_model_dir=r"PaddleOCR\PP-OCRv5_server_det_infer",
    text_recognition_model_name="PP-OCRv5_server_rec",
    text_recognition_model_dir=r"PaddleOCR\PP-OCRv5_server_rec_infer"
)

# ocr = PaddleOCR(lang="en") # 通过 lang 参数来使用英文模型
# ocr = PaddleOCR(ocr_version="PP-OCRv4") # 通过 ocr_version 参数来使用 PP-OCR 其他版本
# ocr = PaddleOCR(device="gpu") # 通过 device 参数使得在模型推理时使用 GPU
# ocr = PaddleOCR(
#     text_detection_model_name="PP-OCRv5_server_det",
#     text_recognition_model_name="PP-OCRv5_server_rec",
#     use_doc_orientation_classify=False,
#     use_doc_unwarping=False,
#     use_textline_orientation=False,
# ) # 更换 PP-OCRv5_server 模型

# 对示例图像执行 OCR 推理 
result = ocr.predict(
    #input=r"img\img\Pkg_Template_CN1_Body_T1\Panel.Board1.Q1.Body_18__25-09-23-16-13-09.png"
    #input=r"img\img\Pkg_Template_CN31_Body_T1\Panel.Board1.Y1.Body_94__25-09-23-16-09-32.png"
    #input=r"img\img\Pkg_Template_CN47_Body_T1\Panel.Board1.X1.Body_87__25-09-23-16-13-09.png"
    #input=r"img\img\Pkg_Template_CN74_Body_T1\Panel.Board1.U2.Body_55__25-09-23-16-09-37.png"
    input=r"PaddleOCR\train_data\det\train\Panel.Board1.Y15.Body_100__25-09-23-16-13-05.png"
    


    )

# 可视化结果并保存 json 结果
for res in result:
    res.print()
    res.save_to_img("output")
    res.save_to_json("output")