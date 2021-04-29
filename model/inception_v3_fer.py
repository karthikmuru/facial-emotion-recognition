import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from sklearn.metrics import accuracy_score

class InceptionV3FER(pl.LightningModule):
  
  def __init__(self):
    super().__init__()

    self.n_classes = 7
    self.inception_model = models.inception_v3(pretrained=True)

    # Freeze the weights of InceptionV3
    for param in self.inception_model.parameters():
      param.requires_grad = False

    num_ftrs = self.inception_model.AuxLogits.fc.in_features
    self.inception_model.AuxLogits.fc = nn.Linear(num_ftrs, self.n_classes)

    num_ftrs = self.inception_model.fc.in_features
    self.inception_model.fc = nn.Linear(num_ftrs, 1024)
    self.fc2 = nn.Linear(1024, 512)
    self.fc3 = nn.Linear(512, 256)
    self.fc4 = nn.Linear(256, 128)
    self.fc5 = nn.Linear(128, self.n_classes)

    self.loss = nn.CrossEntropyLoss()
  
  def forward(self, x):

    x = self.inception_model(x)
    x = self.fc2(x)
    x = self.fc3(x)
    x = self.fc4(x)
    x = self.fc5(x)

    return x
  
  def training_step(self, batch, batch_idx):
    # Auxilary output is also used for training

    inputs, labels = batch

    outputs, aux_outputs = self.inception_model(inputs)
    outputs = self.fc2(outputs)
    outputs = self.fc3(outputs)
    outputs = self.fc4(outputs)
    outputs = self.fc5(outputs)

    loss1 = self.loss(outputs, labels)
    loss2 = self.loss(aux_outputs, labels)
    loss = loss1 + 0.4*loss2

    _, preds = torch.max(outputs, 1)
    train_acc = accuracy_score(preds.cpu(), labels.cpu())

    self.log('train_acc', train_acc, prog_bar=True, logger=True, on_step=False, on_epoch=True)
    self.log('train_loss', loss, prog_bar=False, logger=True, on_step=False, on_epoch=True)

    return { "loss": loss }


  def validation_step(self, val_batch, batch_idx):

    inputs, labels = val_batch
    outputs = self.forward(inputs)
    loss = self.loss(outputs, labels)
    
    _, preds = torch.max(outputs, 1)
    val_acc = torch.tensor(accuracy_score(preds.cpu(), labels.cpu()))

    self.log('val_loss', loss, prog_bar=True)
    self.log("val_acc", val_acc, prog_bar=True, logger=True, on_step=True, on_epoch=True, )

  def configure_optimizers(self):
    # Custom Learning rate for each layer. Increases as we go deeper towards output. 
    optimizer = optim.Adam(
      [
          {"params": self.inception_model.Mixed_5b.parameters(), "lr": 1e-5},
          {"params": self.inception_model.Mixed_5c.parameters(), "lr": 1e-5},
          {"params": self.inception_model.Mixed_5d.parameters(), "lr": 1e-5},
          {"params": self.inception_model.Mixed_6a.parameters(), "lr": 1e-5},
          {"params": self.inception_model.Mixed_6b.parameters(), "lr": 1e-4},
          {"params": self.inception_model.Mixed_6c.parameters(), "lr": 1e-4},
          {"params": self.inception_model.Mixed_6d.parameters(), "lr": 1e-4},
          {"params": self.inception_model.Mixed_6e.parameters(), "lr": 1e-4},
          {"params": self.inception_model.Mixed_7a.parameters(), "lr": 1e-4},
          {"params": self.inception_model.Mixed_7b.parameters(), "lr": 1e-4},
          {"params": self.inception_model.Mixed_7c.parameters(), "lr": 1e-4},
          {"params": self.inception_model.AuxLogits.parameters()},
          {"params": self.inception_model.fc.parameters()},
          {"params": self.fc2.parameters()},
          {"params": self.fc3.parameters()},
          {"params": self.fc4.parameters()},
          {"params": self.fc5.parameters()}
      ],
      lr=1e-3,
    )

    return optimizer

  def on_epoch_start(self):
    # Start traning the Inception layers from Epoch 5
    # Unfreezing the weights    
    if(self.current_epoch == 5):
      for param in self.inception_model.parameters():
        param.requires_grad = True
      print("Unfreezed params!")  