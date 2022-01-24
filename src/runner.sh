#!/bin/bash

echo 'Calling scripts!'

#rm -rf logs

# centralized
for i in {1..5}
do  

    python centralized.py --data=fmnist &
    python centralized.py --data=cifar10 --device=cuda:1
   
done


# fairness FL
for i in {1..5}
do
    python federated.py --data=fmnist --concent=1.0  &
    python federated.py --data=cifar10 --concent=1.0 --device=cuda:1 


    python federated.py --data=fmnist --concent=0.5 &
    python federated.py --data=cifar10 --concent=0.5 --device=cuda:1 &

    python federated.py --data=fmnist --concent=0.25  &
    python federated.py --data=cifar10 --concent=0.25 --device=cuda:1

done



# attack without defense
num_corrupt=1
attack=2

for i in {1..5}
do
    for poison_frac in 0.5 1.0
    do  
        python federated.py --data=fmnist --concent=1.0 --poison_frac=$poison_frac --attack=$attack --num_corrupt=$num_corrupt &
        python federated.py --data=fmnist --concent=0.5 --poison_frac=$poison_frac --attack=$attack --num_corrupt=$num_corrupt &
        python federated.py --data=fmnist --concent=0.25 --poison_frac=$poison_frac --attack=$attack --num_corrupt=$num_corrupt &

        python federated.py --data=cifar10 --concent=1.0 --poison_frac=$poison_frac --attack=$attack --num_corrupt=$num_corrupt --device=cuda:1 &
        python federated.py --data=cifar10 --concent=0.5 --poison_frac=$poison_frac --attack=$attack --num_corrupt=$num_corrupt --device=cuda:1 &
        python federated.py --data=cifar10 --concent=0.25 --poison_frac=$poison_frac --attack=$attack --num_corrupt=$num_corrupt --device=cuda:1 
    done
done


rlr_threshold=4
num_corrupt=1
attack=2

for i in {1..5}
do
    for poison_frac in 0.5
    do  
        python federated.py --data=fmnist --concent=1.0 --poison_frac=$poison_frac --attack=$attack --num_corrupt=$num_corrupt --rlr_threshold=$rlr_threshold &
        python federated.py --data=fmnist --concent=0.5 --poison_frac=$poison_frac --attack=$attack --num_corrupt=$num_corrupt --rlr_threshold=$rlr_threshold &
        python federated.py --data=fmnist --concent=0.25 --poison_frac=$poison_frac --attack=$attack --num_corrupt=$num_corrupt --rlr_threshold=$rlr_threshold &

        python federated.py --data=cifar10 --concent=1.0 --poison_frac=$poison_frac --attack=$attack --num_corrupt=$num_corrupt --rlr_threshold=$rlr_threshold --device=cuda:1 &
        python federated.py --data=cifar10 --concent=0.5 --poison_frac=$poison_frac --attack=$attack --num_corrupt=$num_corrupt --rlr_threshold=$rlr_threshold --device=cuda:1 &
        python federated.py --data=cifar10 --concent=0.25 --poison_frac=$poison_frac --attack=$attack --num_corrupt=$num_corrupt --rlr_threshold=$rlr_threshold --device=cuda:1 
    done
done

echo 'All experiments are running!'





