import os 
from functools import partial
import torch.utils.data as data 
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import torch 

from mydata import ReverseDataset
from mymodel import ReversePredictor

# Path to the folder where the pretrained models are saved
CHECKPOINT_PATH = "./saved_models/"
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print("Device:", device)

def train_reverse(**kwargs): 
    root_dir = os.path.join(CHECKPOINT_PATH, "ReverseTask")
    # exist_ok (optional) : A default value False is used for this parameter. If the target directory already exists an OSError is raised if its value is False otherwise not. 
    # For value True leaves directory unaltered. 
    os.makedirs(root_dir, exist_ok=True)
    trainer = pl.Trainer(default_root_dir=root_dir,
                         callbacks=[ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc")],
                         accelerator="gpu" if str(device).startswith("cuda") else "cpu",
                         devices=1,
                         max_epochs=10,
                         gradient_clip_val=5)
    trainer.logger._default_hp_metric = None # Optional logging argument that we don't need

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(root_dir, "ReverseTask.ckpt")
    if os.path.isfile(pretrained_filename):
        print("Found pretrained model, loading...")
        model = ReversePredictor.load_from_checkpoint(pretrained_filename)
    else:
        model = ReversePredictor(max_iters=trainer.max_epochs*len(train_loader), **kwargs)
        trainer.fit(model, train_loader, val_loader)

    # Test best model on validation and test set
    val_result = trainer.test(model, val_loader, verbose=False)
    test_result = trainer.test(model, test_loader, verbose=False)
    result = {"test_acc": test_result[0]["test_acc"], "val_acc": val_result[0]["test_acc"]}

    model = model.to(device)
    return model, result

if __name__ == "__main__": 
    dataset = partial(ReverseDataset, 10, 16)
    train_loader = data.DataLoader(dataset(50000), batch_size=128,shuffle=True,drop_last=True,pin_memory=True)
    val_loader = data.DataLoader(dataset(1000), batch_size=128)
    test_loader = data.DataLoader(dataset(10000), batch_size=128)

    reverse_model, reverse_result = train_reverse(input_dim=train_loader.dataset.num_categories,
                                                model_dim=32,
                                                num_heads=1,
                                                num_classes=train_loader.dataset.num_categories,
                                                num_layers=1,
                                                dropout=0.0,
                                                lr=5e-4,
                                                warmup=50)

    print(f"Val accuracy:  {(100.0 * reverse_result['val_acc']):4.2f}%")
    print(f"Test accuracy: {(100.0 * reverse_result['test_acc']):4.2f}%")