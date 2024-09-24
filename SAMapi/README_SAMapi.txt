文件目录，-指文件夹，--指文件
SAMapi
    - SAM-Med2D
        #- assets
        - data_demo
        #- examples/SAM-Med2D-onnxruntime
        - onnx_model # 必须 model_path="./SAM-Med2D/onnx_model/sam-med2d_b.encoder.onnx"
            -- sam-med2d_b.decoder.onnx
            -- sam-med2d_b.encoder.onnx
        #- pretrained_model
        #- scripts
        #- segment_anything
        #- 其他文件，除了队友加入的run_py.txt与environment.md没有变化
    - upload
        空文件夹
    -- api.py # 必须，运行这个文件
    -- main.py # 复印自eg_api/SAM-Med2D/examples/SAM-Med2D-onnxruntime/main.py # 必须
    -- test_api.ipynb

python3.12.5
requiements2.txt

cd SAMapi

pip install -r requiements2.txt

python api.py # main加载模型，创建api

运行test_api.ipynb第一个块块 # 输入含有图片路径的request