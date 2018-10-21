import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from torchvision import datasets, models, transforms

def get_model(name, depth=16, pretrained=True):
    if (name == 'alexnet'):
        net = models.alexnet(pretrained=pretrained)
        file_name = 'alexnet'
    elif (name == 'vggnet'):
        if(depth == 11):
            net = models.vgg11(pretrained=pretrained)
        elif(depth == 13):
            net = models.vgg13(pretrained=pretrained)
        elif(depth == 16):
            net = models.vgg16(pretrained=pretrained)
        elif(depth == 19):
            net = models.vgg19(pretrained=pretrained)
        else:
            print('Error : VGGnet should have depth of either [11, 13, 16, 19]')
            sys.exit(1)
        file_name = 'vgg-%s' %(depth)
    elif (name == 'squeezenet'):
        net = models.squeezenet1_0(pretrained=pretrained)
        file_name = 'squeeze'
    elif (name == 'resnet'):
        net = models.resnet(pretrained=pretrained)
        file_name = 'resnet'
    else:
        print('Error : Network should be either [alexnet / squeezenet / vggnet / resnet]')
        sys.exit(1)

    return net, file_name

def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history

# class EyeModel(nn.Module):
#     def __init__(self, num_classes=4):
#         super(EyeModel, self).__init__()
#
#         self.resnet = models.resnet18(pretrained=True)
#         self.num_features = self.resnet.fc.in_features
#         self.resnet.fc = nn.Linear(self.num_features, num_classes)
#         self.input_size = 224
#
#     def forward(self, x):
#         return self.resnet(x)
#
#     def set_parameter_requires_grad(self):
