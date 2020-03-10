import torch
import json
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models

from get_input_args import get_train_args
from workspace_utils import active_session
from collections import OrderedDict

alexnet = models.alexnet(pretrained=True)
vgg16 = models.vgg16(pretrained=True)
densenet121 = models.densenet121(pretrained=True)

models = {"alexnet": alexnet, "vgg": vgg16, "densenet": densenet121}

def transform_data(train_args):
    # Load the data
    train_dir = train_args.data_dir + '/train'
    valid_dir = train_args.data_dir + '/valid'
    
    # Define transforms
    data_transforms = {"training" : transforms.Compose([transforms.RandomRotation(30),
                                                        transforms.RandomResizedCrop(224),
                                                        transforms.RandomHorizontalFlip(),
                                                        transforms.ToTensor(),
                                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                                             [0.229, 0.224, 0.225])]),
                       "testing" : transforms.Compose([transforms.Resize(225),
                                                       transforms.CenterCrop(224),
                                                       transforms.ToTensor(),
                                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                                             [0.229, 0.224, 0.225])])}
    
    # Load datasets with ImageFolder
    image_datasets = {"train_data" : datasets.ImageFolder(train_dir, transform=data_transforms['training']),
                      "valid_data" : datasets.ImageFolder(valid_dir, transform=data_transforms['testing'])}
    
    return image_datasets

def get_dataloaders(image_datasets):
    
    # Define the dataloaders
    dataloaders = {"trainloader" : torch.utils.data.DataLoader(image_datasets['train_data'], batch_size=64, shuffle=True),
                   "validloader" : torch.utils.data.DataLoader(image_datasets['valid_data'], batch_size=64)}
    
    return dataloaders

def define_classifiers(hidden_units):
    # recommend alexnet = 4096, vgg = 4096, and densenet = 512
    classifiers = {"alexnet": nn.Sequential(OrderedDict([
                            ("fc1", nn.Linear(9216, hidden_units)),
                            ("relu", nn.ReLU()),
                            ("dropout", nn.Dropout(p=0.2)),
                            ("fc3", nn.Linear(hidden_units, 102)),
                            ("output", nn.LogSoftmax(dim=1))])),
               "vgg": nn.Sequential(OrderedDict([
                            ("fc1", nn.Linear(25088, hidden_units)),
                            ("relu", nn.ReLU()),
                            ("dropout", nn.Dropout(p=0.2)),
                            ("fc2", nn.Linear(hidden_units, 102)),
                            ("output", nn.LogSoftmax(dim=1))])),
               "densenet": nn.Sequential(OrderedDict([
                            ("fc1", nn.Linear(1024, hidden_units)),
                            ("relu", nn.ReLU()),
                            ("dropout", nn.Dropout(p=0.2)),
                            ("fc2", nn.Linear(hidden_units, 102)),
                            ("output", nn.LogSoftmax(dim=1))]))}
    return classifiers

def train(classifiers, dataloaders, train_args):
    
    model = models[train_args.arch]
    
    # Freeze parameters
    for param in model.parameters():
        param.requires_grad = False
        
    model.classifier = classifiers[train_args.arch]
    
    device = torch.device("cuda" if torch.cuda.is_available() and train_args.gpu else "cpu")
    model = model.to(device)
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=train_args.learning_rate)
    
    epochs = train_args.epochs
    steps = 0
    running_loss = 0
    print_every = 10

    with active_session():
        for epoch in range(epochs):
            for inputs, labels in dataloaders["trainloader"]:
                steps += 1
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                logps = model.forward(inputs)
                loss = criterion(logps, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                if steps % print_every == 0:
                    test_loss = 0
                    accuracy = 0
                    model.eval()

                    with torch.no_grad():
                        for inputs, labels in dataloaders["validloader"]:
                            inputs, labels = inputs.to(device), labels.to(device)

                            logps = model.forward(inputs)
                            batch_loss = criterion(logps, labels)

                            test_loss += batch_loss.item()

                            ps = torch.exp(logps)
                            top_p, top_class = ps.topk(1, dim=1)
                            equals = top_class == labels.view(*top_class.shape)
                            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                    print(f"Epoch {epoch+1}/{epochs}.. "
                         f"Train loss: {running_loss/print_every:.3f}.. "
                         f"Test loss: {test_loss/len(dataloaders['validloader']):.3f} "
                         f"Test accuracy: {accuracy/len(dataloaders['validloader']):.3f} ")
                    running_loss = 0
                    model.train()
    return model, optimizer

def save_checkpoint(image_datasets, model, optimizer, train_args):
    model.class_to_idx = image_datasets['train_data'].class_to_idx
    checkpoint = {'epochs': train_args.epochs,
                  'learning_rate': train_args.learning_rate,
                  'optimizer_state': optimizer.state_dict(),
                  'classifier': model.classifier,
                  'state_dict': model.state_dict()}
    torch.save(checkpoint, 'checkpoint.pth')

def main():

    train_args = get_train_args()
    
    image_datasets = transform_data(train_args)
    dataloaders = get_dataloaders(image_datasets)
    classifiers = define_classifiers(train_args.hidden_units)
    model, optimizer = train(classifiers, dataloaders, train_args)
    save_checkpoint(image_datasets, model, optimizer, train_args)

if __name__ == "__main__":
    main()