# Prune Your Model Before Distill It

### Welcome

This is an PyTorch implement of the paper ``Prune Your Model Before Distill It''.

## Table of Contents

### [1 How to run](#how_to_run)
### [2 ResNet Optimization](#resnet_opt)

## <a name=how_to_run></a>1 How to run

### 1.1 Full framework
```
python main.py --pre_train --pruning --kd
```
1. code progress
* Train the model. (default: vgg-19 on cifar100)
* Proceed with pruning the previously trained model. (default: lr-rewinding, 0.79 pruning ratio)
* KD the model with previously pruned model. (default: vanilla KD|vgg19-rwd-st79|cifar100)
* To test the full framework: `python main.py --pre_train --pruning --kd`

* You can check all argument using 'help' command and change them to `.json` file. 
  (json file path should be `experiments/hyperparam`)

* Default setting is `vgg19` (79% weight pruned) teacher and `vgg19-rwd-st79` student.

* All training result will be stored in `result/data`.

* If you want to run `training | pruning | kd` separately, you can refer to 1.2/1.3/1.4 below.

* In this version, we provide the student model used in the experiment. (Structured pruning applied)

* If you want to use the end-to-end student model, use `--student_model vgg-custom`. This creates a VGG student model according to the teacher model (only VGG is currently available and ResNet version will be updated)

### 1.2 Train Model
The following command will train the model.
```
python main.py --pre_train
```
* Run model training only.

### 1.3 Prune the Teacher
```
python main.py --pruning
```
* Run model pruning only.
* Put the trained model to be pruned as `.pth` in the `experiments/trained_model` folder. Put the name of the model in the
  `--pre_model_name` argument. The model name is separated by `_`.
* If the name of the trained model is `vgg19_trainedmodel.pth` then you can directly edit the `prune_default.json` in
`experiments/hyperparam` or enter python commands to run it.
  ex) `python main.py --pruning --pre_model_name vgg19_traindmodel`

### 1.4 Distill the Teacher to Student
```
python main.py --kd
```
* Run KD only
* Put the teacher model to be pruned as `.pth` in the `experiments/teacher_model` folder.  Put the name of the model in the
  `--teacher_model_name` argument. The model name is separated by `_`.
* If the name of the teacher model is `vgg19_pruned.pth` then you can directly edit the `kd_default.json` in
`experiments/hyperparam` or enter python commands to run it.
  ex) `python main.py --kd --teacher_model_name vgg19_pruned`
* If you want to use the end-to-end student model, use `--student_model vgg-custom`. This creates a VGG student model according to the teacher model (only VGG is currently available and ResNet version will be updated)

### 1.5 Available models
* CIFAR100
```
    cifar100_models = {
        'vgg11',
        'vgg19',
        'vgg19-rwd-cl1',
        'vgg19-rwd-cl2',
        'vgg19-rwd-st36',
        'vgg19-rwd-st59',
        'vgg19-rwd-st79',
        'vgg19dbl',
        'vgg19dbl-rwd-st36',
        'vgg19dbl-rwd-st59',
        'vgg19dbl-rwd-st79',
        'vgg-custom'
    }
```

* Tiny-ImageNet
```
    tiny_imagenet_models = {
        'vgg16',
        'resnet18',
        'resnet18-rwd-st36',
        'resnet18-rwd-st59',
        'resnet18-rwd-st79',
        'resnet18dbl',
        'resnet18dbl-rwd-st36',
        'resnet18dbl-rwd-st59',
        'resnet18dbl-rwd-st79',
        'resnet50',
        'resnet50-rwd-st36',
        'resnet50-rwd-st59',
        'resnet50-rwd-st79',        
        'mobilenet-v2'
    }
```

## <a name=resnet_opt></a>2 ResNet Optimization

* We experimented in different ways in ResNet optimization for Tiny-ImageNet.

If the maxpooling layer is used in conv1, a larger batche_size and learning rate should be used, and the accuracy gain
obtained by using the pruned teacher may slightly decrease.
