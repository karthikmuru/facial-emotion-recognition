import argparse
import torch

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from data import KDEF
from model import InceptionV3FER

def _setup_parser():
  parser = argparse.ArgumentParser(add_help=False)

  trainer_parser = pl.Trainer.add_argparse_args(parser)
  trainer_parser._action_groups[1].title = "Trainer Args"

  data_group = parser.add_argument_group("Data Args")
  KDEF.add_to_argparse(data_group)

  parser.add_argument("--help", "-h", action="help")

  return parser

def main():
  parser = _setup_parser()
  args = parser.parse_args()

  data = KDEF(args)
  model = InceptionV3FER(args)

  logger = TensorBoardLogger('training/logs')
  model_checkpoint_callback = pl.callbacks.ModelCheckpoint(
    monitor="val_loss", 
    dirpath = 'training/logs',
    filename="{epoch:03d}-{val_loss:.3f}", 
    save_top_k = 3,
    mode="min"
  )

  print(data)
  
  args.weights_summary = "full"
  trainer = pl.Trainer.from_argparse_args(args, callbacks=callbacks, logger=logger)
  trainer.fit(model, data)

if __name__ == "__main__":
  main()