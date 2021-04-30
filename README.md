# Facial Emotion Recognition

Project demonstrating the use of transfer learning for the application of facial emotion recognition. The model has been trained based on the ImageNet pretrained Inception V3 model. This project is implemented using  `pytorch-lightning`.

## Data

Karolinska Directed Emotional Faces (KDEF) : https://www.kdef.se/

## Setup and Usage

1.  Create and activate virtual environment
```
python3 -m venv FER_env

source FER_env/bin/activate
```

2.  Install requirements
```
pip3 install -r requirements.txt
```

3. Preprocess the data:

Download the original dataset from the link mentioned above and preprocess it using the preprocessing Jupyter notebook in the notebooks folder.

4. Train the model
```
python3 facial-emotion-recognition/run_experiment.py \
  --batch_size=32 \
  --train_val_split=80 \
  --gpus=-1 \
  --data_dir="/content/drive/MyDrive/ML/FER/KDEF_resized_backup" \
  --progress_bar_refresh_rate=20 \
  --num_workers=2 \
  --max_epochs=20
```

The model weights from the epochs with the top 3 least validation losses will be saved in the `traning/logs` folder.
