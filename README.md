# kaggle_human_protein_atlas_image_classification

Links:
- [My GitHub](https://github.com/avipartho)
- [Human Protein Atlas Image Classification](https://www.kaggle.com/c/human-protein-atlas-image-classification)

Public Score: `0.50487` 
Private Score: `0.47002`

# 1. Requirements

`pytorch-0.4.0` `pretrainedmodels-0.7.4` `imgaug` `iterative-stratification` `scikit-klearn` `python 3`

# 2. Usage

`step1: change config.py for different directories and parameters`

`step2: run main.py`

# 3. Saving models

Three models were saved from training process: best-loss, best-f1 & final-epoch

# 4. Results

After running for 50 epochs, an ensemble of 5-fold cv gave **0.50487** on public data (best-f1 model) and **0.47002** on test data (best-loss model)


# 5. NOTES

- Training data was stratified and oversampled according to class frequency
- Used pretrained network : bninception
- 512x512, 4 channel image was used
- Pretrained model's 1st layer 4th channel was initialized with green channel's weights
- Time took for a 80-20 train-val split on a 12 GB tesla k80: around 20 hrs
- Because some unknown problems,the result may vary a bit.

# 6. Observations

- Image normalization worsen the result a bit
- Weighted BCE, f1 loss and focal loss didn't work well
- Data stratification and oversampling helped a lot
- 5 fold cv helps in all cases
- Fixed threshold (0.15) did way better than thresholds calculated over entire training data

# 7. To-dos

- Apply other pretrained networks (i.e. resnet 34)
- Try with 3 channel (RGB) images
- Apply TTA (hflip and vflip in particular)