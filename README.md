
# ShortcutProbe
## Preparation

### Download datasets
- [Waterbirds](https://nlp.stanford.edu/data/dro/waterbird_complete95_forest2water2.tar.gz)
- [CelebA](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset) ([metadata](https://github.com/PolinaKirichenko/deep_feature_reweighting/blob/main/celeba_metadata.csv))
- ImageNet [(train](https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar) [,val)](https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar)
- [ImageNet-A](https://github.com/hendrycks/natural-adv-examples)
- [NICO](https://drive.google.com/drive/folders/17-jl0fF9BxZupG75BtpOqJaB6dJ2Pv8O?usp=sharing)
- CivilComments: The dataset will be automatically downloaded from`wilds`
- MultiNLI: Call the function `download` in `data/multinli.py` to download the dataset.

Unzip the dataset files into individual folders.

In the `config.py` file, set each value in `dataset_paths` to your corresponding dataset folder. 

### Prepare `metadata.csv` for each dataset
- Waterbirds, CelebA, CivilComments, and MultiNLI provide `metadata.csv` files.
- For the ImageNet-9 and ImageNet-A datasets, run the following code
    ```python
    from data.in9_data import prepare_imagenet9_metadata, prepare_imageneta_metadata
    base_dir = "path/to/imagenet/folder"
    prepare_imagenet9_metadata(base_dir)
    data_root = "path/to/imagenet-a/folder"
    prepare_imageneta_metadata(data_root)
    ````
- For the NICO dataset, run the following to prepare metadata:
    ```python 
    from data.nico_data import prepare_metadata
    prepare_metadata(NICO_DATA_FOLDER, NICO_CXT_DIC_PATH, NICO_CLASS_DIC_PATH)
    ```
## ERM training
Train an ERM model.
```python
python main.py --dataset waterbirds\
               --save_folder /p/spurious/spurious_exprs\
               --backbone resnet50\
               --batch_size 32\
               --pretrained True\
               --mode train\
               --epoch 100\
               --optimizer sgd\
               --optimizer_kwargs lr=0.003 weight_decay=0.0001 momentum=0.9\
               --scheduler cosine\
               --scheduler_kwargs T_max=100\
               --gpu 0\
               --seed 0\
               --split_train 1.0\
               --train_split train\
               --test_split test\
               --algorithm erm
```
## ShortcutProbe
Run NeuronTune using the ERM-trained model above. Choose different datasets and model architectures by setting dataset and backbone, respectively.
The mappings between the parameters here and the parameters in the paper are as follows:
- `n_base` corresponds to `K`
- `sem_reg` corresponds to $\eta$
- `spu_reg` corresponds to $\lambda$
- `mis_ratio` corresponds to $r$

`optimizer_vec_kwargs` specifies optimization parameters for learning the shortcut detector.
`optimizer_cls_kwargs` specifies optimization parameters for learning the last classification layer.
```python
python main.py --dataset waterbirds\
               --save_folder /p/spurious/spurious_exprs\
               --backbone resnet50\
               --batch_size 32\
               --pretrained True\
               --mode train\
               --gpu 0\
               --seed 0\
               --split_train 1.0\
               --split_val 0.5\
               --erm_model /p/spurious/spurious_exprs/erm_waterbirds_resnet50_train_32B_100E_seed0/best_val_acc_model.pt\
               --optimizer_cls sgd\
               --optimizer_cls_kwargs lr=0.001 weight_decay=0.0001 momentum=0.9\
               --optimizer_vec sgd\
               --optimizer_vec_kwargs lr=0.0001 weight_decay=0.0001 momentum=0.9\
               --shortcutprobe1_epochs 50\
               --shortcutprobe2_epochs 50\
               --train_split val_subset1\
               --test_split test\
               --n_base 2\
               --mis_ratio 0.3\
               --sem_reg 5\
               --spu_reg 5\
               --algorithm shortcutprobe
```