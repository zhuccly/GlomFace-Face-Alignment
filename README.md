
A Tensorflow implementation of the proposed GlomFace


# Installation Instructions

Menpo project
Menpo 0.8.1
Menpodetect 0.5.0
Menpo fit 0.5.0
TensorFlow 1.10.1

# Pretrained model 
Google Drive:[Download](https://drive.google.com/file/d/1Z9rx6aCBvRKB-00R08idV3xBV7QHROAx/view?usp=sharing)(Note that we saved the complete network and not just the parameters, so the model file is large. )

# Masked dataset
Masked 300W:[Download](https://drive.google.com/file/d/1598pCEdSmmubxjCuQ8OdxyG6E833Ybtx/view?usp=sharing)

# Test GlomFace
```
    # Activate the conda environment.
    source activate environment-name

    # Track the train process and evaluate the current checkpoint against the validation set
    python Glom_eval.py --dataset_path="./databases/Masked_300W/ibug/*.jpg" --num_examples=135 --eval_dir=ckpt/eval  --device='/gpu:0' --checkpoint_dir=$PWD/ckpt/GF_model
    
```
# Visualization
```
    # Run tensorboard to visualise the results
    tensorboard --logdir==$PWD/ckpt/eval
```
