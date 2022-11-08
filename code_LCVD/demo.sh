DATASET='CIFAR10' # ('CIFAR10', 'SVHN')
MODEL='ResNet18' # (ResNet18' 'VGG19' 'ShuffleNetV2' 'DenseNet121')
ORDER=20
NAME='0'

python main_baseline.py --model ${MODEL} --dataset ${DATASET} --name=${NAME}\

python main_LCVD.py --dataset ${DATASET} --model ${MODEL} --name=${NAME} --order ${ORDER} --pretrained --epoch 1