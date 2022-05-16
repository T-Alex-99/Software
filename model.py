#pip3 install torch torchvision torchaudio
import torch
import torchvision.models as models
import torch.nn.functional as F
import torchvision.transforms as T
import torch.nn as nn
from torchvision.datasets import ImageFolder

CLASSES = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

VALIDATION_TRANSFORM = T.Compose([T.Resize((150,150)), T.ToTensor(), T.Normalize((0.3,0.3,0.3),(1,1,1))])
def accuracy(outputs, labels):
    
    _, preds = torch.max(outputs, dim = 1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


class ImageClassificationBase(nn.Module):
    
    def training_step(self, batch):
        images, labels = batch 
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss
        return loss
    
    def validation_step(self, batch):
        
        images, labels = batch 
        out = self(images)                    # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}
        
    def validation_epoch_end(self, outputs):
        
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        
        print("Epoch [{}], last_lr: {:.5f}, train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, result['lrs'][-1], result['train_loss'], result['val_loss'], result['val_acc']))




class LandscapeCNN(ImageClassificationBase):
    
    def __init__(self):
        
        super().__init__()
        
        # Using a pretrained model
        self.network = models.resnet50(pretrained = True)
        
        # Replace last layer
        num_ftrs = self.network.fc.in_features
        self.network.fc = nn.Linear(num_ftrs, 6)
    
    
    def forward(self, inpt):
        return torch.sigmoid(self.network(inpt))
    
    
    # freeze function trains the fully connected layer to make predictions
    def freeze(self):
        
        # To freeze the residual layers
        for param in self.network.parameters():
            param.require_grad = False
            
        for param in self.network.fc.parameters():
            param.require_grad = True
            
            
    def unfreeze(self):
        # Unfreeze all layers
        for param in self.network.parameters():
            param.require_grad = True



def get_default_device():
    
    #Pick GPU if available, else CPU
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def to_device(data, device):
    
    #Move tensor(s) to chosen device
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


class DeviceDataLoader():
    
    #Wrap a dataloader to move data to a device
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        #Yield a batch of data after moving it to device
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        #Number of batches
        return len(self.dl)

# getting the gpu 
device = get_default_device()
device

# function to predict a single image
def predict(image, model):
    
    xb = image.unsqueeze(0)
    xb = to_device(xb, device)
    preds = model(xb)
    
    max_ = torch.max(preds[0]).item()
    prediction = CLASSES[(preds[0] == max_).nonzero()]
    
    print("Prediction: ", prediction)




def load_model():
    model = LandscapeCNN()
    model.load_state_dict(torch.load("./landscape_model.pth", map_location=torch.device('cpu')))
    model.eval()

    return model


def predict_images(pred_dir):
    model = load_model()
    pred_ds = ImageFolder(pred_dir, VALIDATION_TRANSFORM)
    for i in range(10):
        print(pred_ds.imgs[i])
        print("Type: ",type(pred_ds[i]))
        print("Object: ",pred_ds[i])
        print("Type 1: ",type(pred_ds[i][0]))
        predict(pred_ds[i][0], model)
    
    #method2
    # Read a PIL image
    from PIL import Image
    image = Image.open("./Images/Pre/130.jpg")

    # transform = transforms.PILToTensor()
    # Convert the PIL image to Torch tensor
    img_tensor = VALIDATION_TRANSFORM(image)
    predict(img_tensor, model)
    # print the converted Torch tensor
    print(img_tensor)


from PIL import Image
import cv2
import numpy as np
def predict_frames(frames):
    model = load_model()
    VALIDATION_TRANSFORM2 = T.Compose([T.ToTensor(), T.Normalize((0.3,0.3,0.3),(1,1,1))])
    for i in range(3):
        img = Image.fromarray(frames[i], 'RGB')
        img = cv2.resize(frames[i], (213, 140), interpolation = cv2.INTER_CUBIC)
        cv2.imshow("Simple_black", img)
        frame_tensor = VALIDATION_TRANSFORM2(img)
        predict(frame_tensor, model)

    cv2.imshow("Simple_black", img)
    cv2.waitKey(0)
    cv2.displayAllWindows()