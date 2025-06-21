from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from ECGDataset import ECGDataset
import lightning as L
from torch.utils.data import random_split
from fold import iterate_fold
from lightning.pytorch.callbacks import StochasticWeightAveraging
from ECGDataLoader import ECGDataLoader
# from MCDANN import MCDANN
import importlib
import torch
import random
import numpy as np

def main():
    batch_size = 64
    num_workers = 12
    num_epochs = 12
    learning_rate = 1e-3
    val_metric = 'val_acc'
    mode = 'max'
    num_classes = 2
    label_file = 'ptb_fold.csv'
    data_dir = '../'
    split_ratio = 0.8   
    model_name = 'MCDANN'  # <-- Set your model class name here
    model_module = importlib.import_module(f'ECGModel.{model_name}')
    ModelClass = getattr(model_module, model_name)

    random_seed = 30
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    for current, remaining in iterate_fold([0,1, 2, 3, 4]):

        dataloader = ECGDataLoader(
            csv_file=label_file,
            data_dir=data_dir,
            fold_train=remaining,
            fold_test=current,
            batch_size=batch_size,
            num_workers=num_workers,
            split_ratio=split_ratio
        )
        dataloader.setup()
        
        model = ModelClass(num_classes=num_classes, learning_rate=learning_rate)

        checkpoint_callback = ModelCheckpoint(
                    dirpath='checkpoints/',
                    filename=f'{current}-ecg-{{epoch:02d}}-{{{val_metric}:.4f}}',
                    save_top_k=1,
                    monitor=val_metric,
                    mode=mode
                )

        trainer = L.Trainer(
            max_epochs=num_epochs,
            accelerator='auto',
            devices=1,
            callbacks=[checkpoint_callback, StochasticWeightAveraging(swa_lrs=1e-2)]
        )

        trainer.fit(model, dataloader.train_dataloader(), dataloader.val_dataloader())

        # Load best model for testing
        best_model_path = checkpoint_callback.best_model_path
        print("best_model_path", best_model_path)
        model = ModelClass.load_from_checkpoint(best_model_path, num_classes=num_classes, learning_rate=learning_rate)
        trainer.test(model, dataloader.test_dataloader()) 

if __name__ == '__main__':
    main()