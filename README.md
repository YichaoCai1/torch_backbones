# torch_backbones
- **Unofficial implementations of some classical backbone CNNs with pytorch** 
- This is pretty much just a practicing project. If you want to do some research, I strongly recommend using officially issued models in torchvision.

- **Requirements**：

  · torch，torch-vision

  · torchsummary
  
  · mlflow
  
  · onnx onnnruntime
  
  · openvino


- **A training script is supplied in “train.py”，the arguments are in “utils/args.yml"**.

- new features：
  1. mlflow traning track. You need a mlflow server, and to intall mlflow on you training server.
  2. onnx inference. 
  3. openvino inference.
 
  N.B. If you don't want these features, pelease check release branch v1.0.0.

  
