import json
import torch
import numpy as np

from get_input_args import get_predict_args
from torchvision import models
from PIL import Image

alexnet = models.alexnet(pretrained=True)
vgg16 = models.vgg16(pretrained=True)
densenet121 = models.densenet121(pretrained=True)

models = {"alexnet": alexnet, "vgg": vgg16, "densenet": densenet121}

def load_checkpoint(filepath, model):
    checkpoint = torch.load(filepath)
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    
    return model

def process_image(image):
    im = Image.open(image)
    im.thumbnail((256, 256))
    
    # Center crop
    width, height = im.size
    desired_width, desired_height = 224, 224
    left = (width-desired_width) / 2
    upper = (height-desired_height) / 2
    right = (width+desired_width) / 2
    lower = (height+desired_height) / 2
    
    im = im.crop((left, upper, right, lower))
    
    # Normalize
    np_image = np.array(im)
    np_image = np_image / 255
    means = np.array([0.485, 0.456, 0.406])
    st_dev = np.array([0.229, 0.224, 0.225])
    np_image = np_image - means
    np_image = np_image / st_dev
    np_image = np_image.transpose(2, 0 , 1)
    return np_image

def predict(image_path, model, topk, gpu):
    device = torch.device("cuda" if torch.cuda.is_available() and gpu else "cpu")
    model = model.to(device)
    
    image = torch.tensor(process_image(image_path)).type(torch.FloatTensor)
    image.unsqueeze_(0)
    inputs = image.to(device)
    
    logps = model(inputs)
    ps = torch.exp(logps)
    top_ps, top_class = ps.topk(topk, dim=1)
    return image.squeeze(), top_ps, top_class

def get_class_name(top_class):
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
        
    return [cat_to_name[str(var)] for var in top_class.cpu().numpy()[0]]
    
def main():
    predict_args = get_predict_args()
    model = models[predict_args.arch]
    model = load_checkpoint(predict_args.checkpoint, model)
    
    image_tensor, top_ps, top_class = predict(predict_args.data_dir, model, predict_args.top_k, predict_args.gpu)
    
    class_name = get_class_name(top_class)
    
    print(f"Top classes: {class_name}")
    print(f"Top probabilities: {top_ps}")
    
    
if __name__ == "__main__":
    main()