#!/bin/bash

modelname='TextCNN'
dataset='Subj'
negNum='3'
posNum='5'
seed='1'
resultnum="132"

modelnum='0'
l2_1='0.00001'
epoch1='20'
lr1='0.005'
bs1='64'
maxbound='1'
loadmodel='0'
loadmodelname='110_Epoch_12_model_1_12_0.708.pt'
startepoch='0'

l2_2='0.0001'
lr2='0.0001'
bs2='64'
epoch2='50'
gpu='1'
modelfilename='132_Epoch_19_model_5_3.pt'

echo "start pretraining "${modelname}
python train.py --modelname ${modelname} --modelnum ${modelnum} --resultnum ${resultnum} --loadmodel ${loadmodel} --modelfilename ${loadmodelname} --startepoch ${startepoch} --posNum ${posNum} --l2 ${l2_1} --seed ${seed} --epoch ${epoch1} --dataset ${dataset} --negNum ${negNum} --lr ${lr1} --bs ${bs1} --maxbound ${maxbound} 

echo "\nstart training "${modelname}" 1"
python classify1.py --modelname ${modelname} --modelnum ${modelnum} --resultnum ${resultnum} --dataset ${dataset} --l2 ${l2_2} --seed ${seed} --lr ${lr2} --batch_size ${bs2} --epoch ${epoch2} --gpu ${gpu}

echo "\nstart training "${modelname}" 2"
python classify1.py --modelname ${modelname} --modelnum ${modelnum} --resultnum ${resultnum} --dataset ${dataset} --l2 ${l2_2} --seed ${seed} --modelfilename ${modelfilename} --use_aug 1 --lr ${lr2} --batch_size ${bs2} --epoch ${epoch2} --gpu ${gpu}

echo "\nstart testing"
# python test.py --dataset ${dataset} --modelname ${modelname}