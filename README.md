# Fairness-Robustness-in-FedLearning

This is the code for the TPS-ISA 2021 paper titled **The Impact of Data Distribution on Fairness and Robustness in Federated Learning**. It has been tested with PyTorch 1.9.0.

See ```src/runner.sh``` for some example usage, and to replicate our results, as well as to run your own experiments.

For example, you can first run some experiments in the centralized setting to get some baselines values: 
```python centralized.py --data=cifar10 --device=cuda:1``` which will train a model on CIFAR10 dataset.

Then, you can measure fairness in the federated learning (FL) setting.
```python federated.py --data=cifar10 --concent=1.0``` where concent is the concentration parameter for Dirichlet distribution. The lower it is, the more non-IID the setting.

Finally, you can run an attack, and look at the interplay between robustness, and fairness in the FL setting.
```bash
python federated.py --data=cifar10 --concent=1.0 --poison_frac=$poison_frac --attack=$attack --num_corrupt=$num_corrupt
```

You can also activate [the RLR defense](https://ojs.aaai.org/index.php/AAAI/article/view/17118), and observe how fairness and robustness change under it.
```bash
python federated.py --data=cifar10 --concent=1.0 --poison_frac=$poison_frac --attack=$attack --num_corrupt=$num_corrupt --rlr_threshold=$rlr_threshold
```
where ```rlr_threshold``` is the threshold parameter of the defense (see the paper for more explanation).

You can also change the aggregation function, local training epoch/batch sizes etc. by configuring the appropriate argument.
```
usage: federated.py [-h] [--data DATA] [--num_agents NUM_AGENTS] [--agent_frac AGENT_FRAC] [--aggr AGGR] [--rlr_threshold RLR_THRESHOLD] [--num_corrupt NUM_CORRUPT] [--rounds ROUNDS] [--local_ep LOCAL_EP] [--bs BS] [--server_lr SERVER_LR]
                    [--class_per_agent CLASS_PER_AGENT] [--attack ATTACK] [--base_class BASE_CLASS] [--target_class TARGET_CLASS] [--poison_frac POISON_FRAC] [--pattern_type PATTERN_TYPE] [--device DEVICE] [--num_workers NUM_WORKERS]
                    [--snap SNAP] [--concent CONCENT]

optional arguments:
  -h, --help            show this help message and exit
  --data DATA           dataset we want to train on
  --num_agents NUM_AGENTS
                        number of agents:K
  --agent_frac AGENT_FRAC
                        fraction of agents per round:C
  --aggr AGGR           aggregation type
  --rlr_threshold RLR_THRESHOLD
                        break ties when votes sum to 0
  --num_corrupt NUM_CORRUPT
                        number of corrupt agents
  --rounds ROUNDS       number of communication rounds:R
  --local_ep LOCAL_EP   number of local epochs:E
  --bs BS               local batch size: B
  --server_lr SERVER_LR
                        servers learning rate
  --class_per_agent CLASS_PER_AGENT
                        default set to IID. Set to 1 for (most-skewed) non-IID.
  --attack ATTACK       0: no attack, 1: sign-flip, 2: backdoor
  --base_class BASE_CLASS
                        base class for backdoor attack
  --target_class TARGET_CLASS
                        target class for backdoor attack
  --poison_frac POISON_FRAC
                        fraction of dataset to corrupt for backdoor attack
  --pattern_type PATTERN_TYPE
                        shape of bd pattern
  --device DEVICE       To use cuda, set to a specific GPU ID.
  --num_workers NUM_WORKERS
                        num of workers for multithreading
  --snap SNAP           do inference in every num of snap rounds
  --concent CONCENT     concentration of dirichlet dist when doing niid sampling
 ```
 
 
 If you use our paper in a way, please consider citing it.
 
 ```bibtex
 @article{ozdayi2021impact,
  title={The Impact of Data Distribution on Fairness and Robustness in Federated Learning},
  author={Ozdayi, Mustafa Safa and Kantarcioglu, Murat},
  journal={arXiv preprint arXiv:2112.01274},
  year={2021}
}
```
