# StaDis: A novel apporach for computioanal pathology image ood detection

[Journal Link](https://www.sciencedirect.com/science/article/abs/pii/S1361841525003202) | [Cite](#Reference)


## Abstract
Modern Computational pathology (CPath) models aim to alleviate the burden on pathologists. However, once deployed, these models may generate unreliable predictions when encountering data types not seen during training, potentially causing a trust crisis within the computational pathology community. Out-of-distribution (OOD) detection, acting as a safety measure before model deployment, demonstrates significant promise in ensuring the reliable use of models in real clinical application. However, most existing computational pathology models lack OOD detection mechanisms, and no OOD detection method is specifically designed for this field. In this paper, we propose a novel OOD detection approach called Stability Distance (StaDis), uniquely developed for CPath. StaDis measures the feature gap between an image and its perturbed counterpart. As a plug-and-play module, it requires no retraining and integrates seamlessly with any model. Additionally, for the first time, we explore OOD detection at the whole-slide image (WSI) level within the multiple instance learning (MIL) framework. Then, we design different pathological OOD detection benchmarks covering three real clinical scenarios: patch- and slide-level anomaly tissue detection, rare case mining, and frozen section (FS) detection. Finally, extensive comparative experiments are conducted on these pathological OOD benchmarks. In 38 experiments, our approach achieves SOTA performance in 23 cases and ranks second in 10 experiments. Especially, the AUROC results of StaDis with CONCH as the backbone improve by 7.91% for patch-based anomaly tissue detection.

## Contribution
* StaDis is the first OOD detection method specifically designed for computational pathology.
* StaDis is the first work to explore pathology OOD detection under a Multiple Instance Learning (MIL) framework.
* We establish pathology-oriented OOD benchmarks that simulate realistic unseen environments in clinical practice.

## Environment
We recommend creating the environment directly from the provided stadis_env.yaml file:
```bash
conda env create -f stadis_env.yaml
conda activate stadis
```
This will install all required dependencies for running our STADIS framework, including PyTorch, CUDA support, WSI preprocessing tools, MIL models, and other utilities.

If you encounter issues related to conflicting CUDA or PyTorch versions, please ensure your local GPU driver is compatible with the CUDA version specified in stadis_env.yaml.

## Data Preprocess
We follow the WSI (Whole Slide Image) preprocessing pipeline provided by [CLAM](https://github.com/mahmoodlab/CLAM).

The first step is to obtain the patch coordinates from a WSI:

 `python create_patches_fp.py --save_dir ./clam_format_patch/ --experiment_mask C16 --seg --patch --stitch`

After generating the patch coordinates, we extract feature embeddings using a pretrained backbone:
 
 `python extract_features_fp.py --model_name resnet50 --experiment_mask C16`

## Create Split
### Step 1. Generate CSV files

Enter the metadata directory and create the dataset CSV files:

 ```bash
 cd dataset_csv
 python make_csv.py --experiment_mask C16 --task task_1_tumor_vs_normal
 ```

### Step 2. Split in-distribution dataset into train/val/test
```bash
python creat_splits_seq.py \
    --model_name resnet50 \
    --task task_1_tumor_vs_normal
```

## Train and Test Model

 ```bash
 python train_wsi_model.py --experiment_mask C16 --model_name resnet50 --model_type abmil --task task_1_tumor_vs_normal
 ```
 Run inference on the ID dataset:
 ```bash
python test_wsi_model.py --experiment_mask C16 --model_name resnet50 --model_type abmil --task task_1_tumor_vs_normal
```


## Train and Test OOD Detection Classifier (StaDis)

### Step 1. OOD detection dataset splitting
For the OOD detection task, we follow the standard setting commonly adopted in natural image OOD detection. The dataset is divided into **training**, **validation**, and **testing** subsets as follows:

* The **original classification training set** is directly used as the **training set** for the OOD detection task.
* The **original classification test set** is further split into validation and test subsets with a **2:8 ratio**.

  * 20% → OOD **validation** set
  * 80% → OOD **test** set

This splitting strategy ensures consistency with conventional OOD detection benchmarks while retaining sufficient evaluation samples for reliable performance measurement.

 ```bash
python create_splits_ood_seq.py --experiment_mask C16 --model_name resnet50 --model_type clam_sb --task task_1_tumor_vs_normal
```


### Step 2. Get perturbation feature of patch
To obtain the perturbation features of the original images, which serve as the initial input to StaDis, run the following command:
```bash
python extract_perturbation_features_fp.py --experiment_mask C16 --model_name resnet50 --task task_1_tumor_vs_normal
```

### Step 3. Train and Test StaDis

To select the optimal hyperparameters for StaDis, we run the following command:
```bash
python -u WSI_OOD_Stadis.py \
    --gpu=2 \
    --model_name=resnet50 \
    --model_type=abmil \
    --task=task_1_tumor_vs_normal \
    --experiment_mask=C16
```
Moreover, As the first work to investigate OOD detection in computational pathology under a multiple instance learning (MIL) framework, we additionally provide implementations of several baseline OOD detection methods adapted to MIL-based WSI models.
```bash
python -u WSI_OOD_baseline.py \
    --gpu=2 \
    --model_name=resnet50 \
    --model_type=abmil \
    --task=task_1_tumor_vs_normal \
    --experiment_mask=C16
```
# Reference
If our work or code is helpful for your research, please consider citing our [paper](https://www.sciencedirect.com/science/article/abs/pii/S1361841525003202).


```bibtex
@article{ZHANG2025103774,
  title   = {StaDis: Stability distance to detecting out-of-distribution data in computational pathology},
  journal = {Medical Image Analysis},
  volume  = {106},
  pages   = {103774},
  year    = {2025},
  issn    = {1361-8415},
  doi     = {https://doi.org/10.1016/j.media.2025.103774},
  author  = {Di Zhang and Jiusong Ge and Jiashuai Liu and Chunbao Wang and Tieliang Gong and Zeyu Gao and Chen Li},
}
```

