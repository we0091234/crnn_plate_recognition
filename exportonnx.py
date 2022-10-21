import numpy as np
from plateNet import myNet_ocr
import time
import cv2
import torch
from torch.autograd import Variable
import lib.utils.utils as utils
import lib.models.crnn as crnn
import lib.config.alphabets as alphabets
import yaml
from easydict import EasyDict as edict
import argparse
import onnx
from onnxsim import simplify
def cv_imread(path):
    img=cv2.imdecode(np.fromfile(path,dtype=np.uint8),-1)
    return img

plateName1=alphabets.plateName1
def decodePlate(preds):
    pre=0
    newPreds=[]
    for i in range(len(preds)):
        if preds[i]!=0 and preds[i]!=pre:
            newPreds.append(preds[i])
        pre=preds[i]
    return newPreds
def parse_arg():
    parser = argparse.ArgumentParser(description="demo")

    parser.add_argument('--cfg', help='experiment configuration filename', type=str, default='lib/config/360CC_config.yaml')
    parser.add_argument('--image_path', type=str, default='images/test.jpg', help='the path to your image')
    parser.add_argument('--checkpoint', type=str, default='/mnt/Gpan/Mydata/pytorchPorject/Chinese_license_plate_detection_recognition/plate_recognition/model/checkpoint_61_acc_0.9715.pth',
                        help='the path to your checkpoints')
    
    args = parser.parse_args()

    with open(args.cfg, 'r') as f:
        config = yaml.load(f)
        config = edict(config)

    config.DATASET.ALPHABETS = alphabets.plateName
    config.MODEL.NUM_CLASSES = len(config.DATASET.ALPHABETS)

    return config, args

def recognition(config, img, model, converter, device):

    # github issues: https://github.com/Sierkinhane/CRNN_Chinese_Characters_Rec/issues/211
    img = cv2.resize(img, (168,48))
    img = np.reshape(img, (48, 168, 3))

    # normalize
    img = img.astype(np.float32)
    img = (img / 255. - config.DATASET.MEAN) / config.DATASET.STD
    img = img.transpose([2, 0, 1])
    img = torch.from_numpy(img)

    img = img.to(device)
    img = img.view(1, *img.size())
    model.eval()
    preds = model(img)
    preds=preds.view(-1).detach().cpu().numpy()
    # _, preds = preds.max(2)
    # preds = preds.transpose(1, 0).contiguous().view(-1)

    # preds_size = Variable(torch.IntTensor([preds.size(0)]))
    # sim_pred = converter.decode(preds.data, preds_size.data, raw=False)

    # print('results: {0}'.format(sim_pred))
    newPreds=decodePlate(preds)
    plate=""
    for i in newPreds:
        plate+=plateName1[int(i)]
    print(plate)
    return img


if __name__ == '__main__':

    config, args = parse_arg()
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    # model = crnn.get_crnn(config,export=True).to(device)
    model = myNet_ocr(num_classes=78,export=True).to(device)
    print('loading pretrained model from {0}'.format(args.checkpoint))
    checkpoint = torch.load(args.checkpoint)
    if 'state_dict' in checkpoint.keys():
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    started = time.time()

    img_raw = cv_imread(args.image_path)
    if img_raw.shape[-1]!=3:
        img_raw=cv2.cvtColor(img_raw,cv2.COLOR_BGRA2BGR)
    # img = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY)
    converter = utils.strLabelConverter(config.DATASET.ALPHABETS)

    in_im = recognition(config, img_raw, model, converter, device)
    print('input image shape: ', in_im.shape)
    finished = time.time()
    print('elapsed time: {0}'.format(finished - started))
    
    onnx_f = args.checkpoint.replace('.pth', '.onnx')
    torch.onnx.export(model, in_im, onnx_f,input_names=["images"],output_names=["output"], verbose=False, opset_version=11)

    input_shapes = {"images": list(in_im.shape)}
    onnx_model = onnx.load(onnx_f)
    model_simp, check = simplify(onnx_model,test_input_shapes=input_shapes)
    onnx.save(model_simp, onnx_f)


    # cv2.imshow('raw', img_raw)
    # cv2.waitKey(0)

    

