import onnxruntime
import numpy as np
from PIL import Image
import time
import argparse
import torchvision.transforms as transforms

parser = argparse.ArgumentParser()
parser.add_argument('--file', '-f', type=str)
parse_args = parser.parse_args()


classes = ['infrared', 'normal']

ort_session = onnxruntime.InferenceSession("onnx_ckpt/model.onnx")
transform = transforms.Compose([transforms.Resize((224, 224)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

start_time = time.time()
image = Image.open(parse_args.file)
image = transform(image)
image = image.unsqueeze(0).cpu().numpy()
ort_inputs = {ort_session.get_inputs()[0].name:image}

ort_outs = ort_session.run(None, ort_inputs)
predict = np.argmax(ort_outs[0], axis=1)
print(classes[predict[0]])
print("onnxrt cost time: %.4f"%(time.time()-start_time))