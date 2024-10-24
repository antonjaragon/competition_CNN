#%%

import torch
import torch.nn.functional as F
import torchvision
import os

from model import CustomCNN, CustomResNet

#%%


def calculate_accuracy(y_pred, y):
  '''
  Compute accuracy from ground-truth and predicted labels.

  Input
  ------
  y_pred: torch.Tensor [BATCH_SIZE, N_LABELS]
  y: torch.Tensor [BATCH_SIZE]

  Output
  ------
  acc: float
    Accuracy
  '''
  y_prob = F.softmax(y_pred, dim = -1)
  y_pred = y_pred.argmax(dim=1, keepdim = True)
  correct = y_pred.eq(y.view_as(y_pred)).sum()
  acc = correct.float()/y.shape[0]
  return acc


def evaluate(model, iterator, criterion, device):
  epoch_loss = 0
  epoch_acc = 0

  # Evaluation mode
  model.eval()

  # Do not compute gradients
  with torch.no_grad():

    for(x,y) in iterator:

      x = x.to(device)
      y = y.to(device)

      # Make Predictions
      y_pred = model(x)

      # Compute loss
      loss = criterion(y_pred, y)

      # Compute accuracy
      acc = calculate_accuracy(y_pred, y)

      # Extract data from loss and accuracy
      epoch_loss += loss.item()
      epoch_acc += acc.item()

  return epoch_loss/len(iterator), epoch_acc/len(iterator)


def model_testing(model, test_iterator, criterion, device, model_name):
  # Test model
  model.load_state_dict(torch.load(model_name, weights_only=True))
  test_loss, test_acc = evaluate(model, test_iterator, criterion, device)
  print(f"Test -- Loss: {test_loss:.3f}, Acc: {test_acc * 100:.2f} %")


# %%
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
test_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((64, 64)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean, std)
])

test_data = torchvision.datasets.ImageFolder(os.path.join('data', 'test'), transform=test_transforms)
test_iterator = torch.utils.data.DataLoader(test_data, batch_size=64)

# print number of dog and cat images in the test set
print(f"Number of cat images: {len(test_data.targets) - sum(test_data.targets)}")
print(f"Number of dog images: {sum(test_data.targets)}")


#%%

model = CustomResNet(num_classes=2)
model_name = 'custom_resnet.pt'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
criterion = torch.nn.CrossEntropyLoss()

criterion.to(device)
model.to(device)

model_testing(model, test_iterator, criterion, device, model_name)




# %%
