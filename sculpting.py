import torch

import torchvision
import torchvision.transforms as transforms

from ManifoldSculpting import ManifoldSculpting



## Import Dataset (MNIST)
transform = transforms.Compose([
    transforms.ToTensor(),
])

# train_test_data = torchvision.datasets.MNIST(root='./Data', train=True, download=True, transform=transform)
train_data = torchvision.datasets.MNIST(root='./Data', train=True, download=True, transform=transform)
val_data  = torchvision.datasets.MNIST(root='./Data', train=False, download=True, transform=transform)

#frac = 0.98
#train_data, test_data = torch.utils.data.random_split(train_test_data, [frac, 1-frac], generator=torch.Generator().manual_seed(42))
#print(len(train_data))
#print(len(  val_data))
#print(len( test_data))


# Filter out images with labels 1 and 7 from the training dataset
indices = (train_data.targets == 1) + (train_data.targets == 2) # this has the same length as train_data.targets, filled with 1 and 0s
indices[8000:] = False  # Limit the number of images with labels 1 and 7
train_data.data, train_data.targets = train_data.data[indices], train_data.targets[indices]

# Same thing for the Validation dataset
indices = (val_data.targets == 1) + (val_data.targets == 2)
indices[4000:] = False
val_data.data, val_data.targets = val_data.data[indices], val_data.targets[indices]

# Same thing for the Test dataset
#indices = (test_data.targets == 1) + (test_data.targets == 7)
#indices[4000:] = False
#test_data.data, test_data.targets = test_data.data[indices], test_data.targets[indices]

#print("Train set: ", len(train_data.targets), "samples \nVal   set: ", len(val_data.targets), "samples")
#print("Train set: ", len(train_data.targets), "samples \nVal set: ", len(val_data.targets), "samples \nTest set: ", len(test_data.targets), "samples")
train_loader = torch.utils.data.DataLoader(train_data, batch_size=len(train_data), shuffle=True)

sculptor = ManifoldSculpting(k=10, sigma=.97) # defaults: k=5, n_dim=2, niter=100, sigma=0.98, patience=20
for data, labels in train_loader:
    sculptor.transform(data)
    break # superfluo pké c'è 1 solo elemento
    
print(f'Epochs: {sculptor.elapsed_epochs}')
print(f'Final mean error: {sculptor.last_error}')
print(f'Best mean error: {sculptor.best_error}')


