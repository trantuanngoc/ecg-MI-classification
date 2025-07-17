# ecg-MI-classification

## How to run
```bash
git clone https://github.com/trantuanngoc/ecg-MI-classification.git

mv processed_ptb ecg-MI-classification/processed_ptb 
# Bỏ thư mục processed_ptb vào thư mục chính để train

pip install lightning pandas

cd ecg-MI-classification/ecg

python main
```

## Project Structure

```
ecg-MI-classification/
├── ecg/
│   ├── main.py  (file to run)
│   ├── ECGDataLoader.py (Data loader for lightning)
│   ├── ECGDataset.py (Dataset for lightning)
│   ├── fold.py (create fold for k-fold)
│   ├── Transform.py (do data augmentation)
│   └── ECGModel ── MCDANN.py (model)
├── processed_ptb (PTB dataset)
├── ptb_fold.csv (label file)
```

