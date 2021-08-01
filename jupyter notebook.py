import os
import torch
import torchvision
from torch.utils.data import random_split
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F

data_dir  = 'I:\Coding\Jupyter\\archive\Garbage classification\Garbage classification'

classes = os.listdir(data_dir)
print(classes)

from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms

transformations = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])

dataset = ImageFolder(data_dir, transform = transformations)

print(dataset)

%matplotlib notebook

import matplotlib.pyplot as plt

def show_sample(img, label):
    print("Label:", dataset.classes[label], "(Class No: "+ str(label) + ")")
    plt.imshow(img.permute(1, 2, 0))

random_seed = 42
torch.manual_seed(random_seed)

train_ds, val_ds, test_ds = random_split(dataset, [1593, 176, 758])
len(train_ds), len(val_ds), len(test_ds)

from torch.utils.data.dataloader import DataLoader
batch_size = 8

train_dl = DataLoader(train_ds, batch_size, shuffle = True, num_workers = 4, pin_memory = True)
val_dl = DataLoader(val_ds, batch_size*2, num_workers = 4, pin_memory = True)

from torchvision.utils import make_grid

def show_batch(dl):
    for images, labels in dl:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(make_grid(images, nrow = 16).permute(1, 2, 0))
        break

#show_batch(train_dl)

#show_batch(val_dl)

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
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
        print("Epoch {}: train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch+1, result['train_loss'], result['val_loss'], result['val_acc']))

class ResNet(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        
        # Use a pretrained model
        self.network = models.resnet50(pretrained=True)
    
        # Replace last layer
        num_ftrs = self.network.fc.in_features
        self.network.fc = nn.Linear(num_ftrs, len(dataset.classes))
    
    def forward(self, xb):
        return torch.sigmoid(self.network(xb))

model = ResNet()

def get_default_device():
    
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    
def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)

device = get_default_device()
device = 'cpu'

train_dl = DeviceDataLoader(train_dl, device)
val_dl = DeviceDataLoader(val_dl, device)
to_device(model, device)

@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    history = []
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):
        
        # Training Phase 
        model.train()
        train_losses = []
        for batch in train_loader:
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
        # Validation phase
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        model.epoch_end(epoch, result)
        history.append(result)
    return history

model = to_device(ResNet(), device)

evaluate(model, val_dl)

def predict_image(img, model):
    # Convert to a batch of 1
    xb = to_device(img.unsqueeze(0), device)
    # Get predictions from model
    yb = model(xb)
    # Pick index with highest probability
    prob, preds  = torch.max(yb, dim=1)
    # Retrieve the class label
    return dataset.classes[preds[0].item()]

pip install anvil-uplink

import anvil.server

anvil.server.connect("I4X5MZUZ546MD7WDWILMDGUN-BGPH3VO3SQJRL643")    

import anvil.media

@anvil.server.callable
def classify_image(file):
    with anvil.media.TempFile(file) as filename:
        img = Image.open(filename)
        img.save('I:\Coding\Jupyter\image.jpg', "JPEG")

loaded_model = model

from PIL import Image
from pathlib import Path

@anvil.server.callable
def predict_external_image(image_name):
    image = Image.open(Path('./' + image_name))

    example_image = transformations(image)
    plt.imshow(example_image.permute(1, 2, 0))
    material = predict_image(example_image, loaded_model)
    materialText = "Error"
    if material == 'plastic':
        materialText = '''What can be recycled:

Plastic bottles and jars (6 ounces or larger), jugs (milk, juice, etc.), round food containers (6 ounces or larger), and buckets (5 gallons or smaller), and plant pots (4-inch diameter or larger).

What can't be recycled:

Any plastic that is not shaped like a bottle, tub, bucket, or jug. This includes: plastic bags or plastic film of any type, such as pallet wrap, bubble wrap, and stretch wrap, plastic caps and lids, plastic 6 pack can holders (all types, including rigid plastic), plastic take-out food containers and disposable plates, cups and cutlery, prescription medicine bottles and other plastic containers under 6 oz, disposable plastic or latex gloves, and bottles that have come in contact with motor oil, pesticides or herbicides.

Additional information:

Don't forget to rinse your containers. There should not be food or residue.'''
    elif material == 'glass':
        materialText = '''What can be recycled:

All colors of glass (labels are OK too).

What can't be recycled:

Drinking glasses, flower vases, ceramics, dishware or drinkware of any kind, light bulbs.

Additional information:

Glass should always be collected separately from other recycling. Never put glass in your mixed recycling container. If glass ends up with other recyclables, it can cause harm to the staff and machinery at local recycling facilities.
    '''
    elif material == 'metal':
        materialText = '''What can be recycled:

Aluminum, tin and steel food cans, empty dry metal paint cans, empty aerosol cans, aluminum foil, scrap metal (smaller than 30 inches and less than 30 pounds).

What can't be recycled:

Aerosol cans that still contain liquids should be emptied (if non-toxic, like cooking spray) or disposed of at a hazardous waste facility (if toxic, like chemicals or paint).

Additional information:

To recycle small metal pieces (under 2 inches), like metal lids, screws and nails, collect inside a soup can, crimp tightly closed, then put into mixed recycling.'''
    elif material == 'paper':
        materialText = '''Paper & Cardboard:
        
What can be recycled: 

Cardboard boxes, newspapers, magazines, catalogs, phone books, scrap paper, junk mail, cartons (milk, juice, soup; empty and dry). Shredded paper is allowed if contained in a paper bag

What can't be recycled:

Coffee cups, take-out food containers, paper plates, paper towels, napkins, facial tissue, wax-coated cardboard, pizza boxes, frozen food boxes, label backing sheets; or paper coated with food, wax, foil or plastic.

Additional information:

With extra cardboard try to fit as much into your bin as possible by flattening and cutting into smaller pieces. When your bin is full, flatten extra cardboard, tape together in one bundle, and lean next to your blue recycling bin. Bundles should be no larger than 3 feet in any direction.'''
    elif material == 'cardboard':
        materialText = '''Paper & Cardboard:

What can be recycled: 

Cardboard boxes, newspapers, magazines, catalogs, phone books, scrap paper, junk mail, cartons (milk, juice, soup; empty and dry). Shredded paper is allowed if contained in a paper bag

What can't be recycled:

Coffee cups, take-out food containers, paper plates, paper towels, napkins, facial tissue, wax-coated cardboard, pizza boxes, frozen food boxes, label backing sheets; or paper coated with food, wax, foil or plastic.

Additional information:

With extra cardboard try to fit as much into your bin as possible by flattening and cutting into smaller pieces. When your bin is full, flatten extra cardboard, tape together in one bundle, and lean next to your blue recycling bin. Bundles should be no larger than 3 feet in any direction.'''
    else:
        materialText = 'ERROR: Please try again.'
    result = "The image resembles " + material + ".\n"
    result = str(result)
    result = result.replace('(', '')
    result = result.replace(')', '')
    result = result.replace("'", '')
    return result, materialText, material

import sys
import re
import json
import random

from requests import get
from tqdm import tqdm
from bs4 import BeautifulSoup as soup
from concurrent.futures import ThreadPoolExecutor

from pydotmap import DotMap


class PinterestImageScraper:

    def __init__(self):
        self.json_data_list = []

    @staticmethod
    def clear():
        if os.name == 'nt':
            _ = os.system('cls')
        else:
            _ = os.system('clear')

    @staticmethod
    def get_pinterest_links(body):
        searched_urls = []
        html = soup(body, 'html.parser')
        links = html.select('#main > div > div > div > a')
        for link in links:
            link = link.get('href')
            link = re.sub(r'/url\?q=', '', link)
            if link[0] != "/" and "pinterest" in link:
                searched_urls.append(link)

        return searched_urls

    def get_source(self, url):
        try:
            res = get(url)
        except Exception as e:
            return
        html = soup(res.text, 'html.parser')
        # get json data from script tag having id initial-state
        json_data = html.find_all("script", attrs={"id": "initial-state"})
        for a in json_data:
            self.json_data_list.append(a.string)

    def save_image_url(self):
        print('[+] saving image urls ...')
        url_list = [i for i in self.json_data_list if i.strip()]
        if not len(url_list):
            return url_list
        url_list = []
        for js in self.json_data_list:
            try:
                data = DotMap(json.loads(js))
                urls = []
                for response in data.resourceResponses:
                    if isinstance(response.response.data, list):
                        for u in range(len(response.response.data)):
                            if isinstance(response.response.data[u].images.get("474x"), list):
                                for i in response.response.data[u].images.get("474x"):
                                    urls.append(i)
                            else:
                                urls.append(response.response.data[u].images.get("474x"))
                    else:
                        if isinstance(response.response.data.images.get("474x"), list):
                            for i in response.response.data.images["474x"]:
                                urls.append(i)
                        else:
                            urls.append(response.response.data.images.get("474x"))

                for data in urls:
                    url_list.append(data.url)
            except Exception as e:
                continue

        return url_list

    @staticmethod
    def saving_op(var):
        url_list, folder_name = var
        if not os.path.exists(os.path.join(os.getcwd(), folder_name)):
                os.mkdir(os.path.join(os.getcwd(), folder_name))
        for img in tqdm(url_list):
            result = get(img, stream=True).content
            file_name = img.split("/")[-1]
            file_path = os.path.join(os.getcwd(), folder_name, file_name)
            with open(file_path, 'wb') as handler:
                handler.write(result)
            print("", end="\r")

    @staticmethod
    def download(url_list, keyword):
        folder_name = keyword
        num_of_workers = 10
        idx = len(url_list) // num_of_workers
        param = []
        for i in range(num_of_workers):
            param.append((url_list[((i*idx)):(idx*(i+1))], folder_name))
        with ThreadPoolExecutor(max_workers=num_of_workers) as executor:
            executor.map(PinterestImageScraper.saving_op, param)
        PinterestImageScraper.clear()

    @staticmethod
    def start_scraping(key=None):
        try:
            key = input("Enter keyword: ") if key == None else key
            keyword = key + " pinterest"
            keyword = keyword.replace("+", "%20")
            url = f'http://www.google.co.in/search?hl=en&q={keyword}'
            res = get(url)
            searched_urls = PinterestImageScraper.get_pinterest_links(res.content)
        except Exception as e:
            return []

        return searched_urls, key.replace(" ", "_")


    def make_ready(self, key=None):
        extracted_urls, keyword = PinterestImageScraper.start_scraping(key)
        for i in extracted_urls:
            self.get_source(i)

        # get all urls of images and save in a list
        url_list = self.save_image_url()
        #url_list = url_list[:9] 
        n = 9
  
        url_list = random.sample(url_list, n)

        # download images from saved images url
        print(f"[+] Total {len(url_list)} files available to download.")
        print()

        if len(url_list):
            try:
                return url_list
            except KeyboardInterrupt:
                return False
            return True
        
        return False



@anvil.server.callable
def searchPin(mat):
    searchTerm = 'recycled ' + mat
    p_scraper = PinterestImageScraper()
    is_downloaded = p_scraper.make_ready(searchTerm)
    return is_downloaded



