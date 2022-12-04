#!/bin/bash 
for dataset in MNIST CIFAR-10
do
	for attack in BadNet WaNet FIBA LIRA BlindBd
	do
		echo $dataset $attack
		if [[ $attack == "BadNet" ]] && [[ $dataset == "MNIST" ]]
		then
			cd BadNets
			source ~/anaconda3/etc/profile.d/conda.sh
			conda activate aml
			python main.py --dataset MNIST --load_local 
			conda deactivate
			cd ..
		elif [[ $attack == "BadNet" ]] && [[ $dataset == "CIFAR-10" ]]
		then
			cd BadNets
			source ~/anaconda3/etc/profile.d/conda.sh
			conda activate badnet
			python main.py --dataset CIFAR10 --load_local 
			conda deactivate
			cd ..
		elif [[ $attack == "WaNet" ]] && [[ $dataset == "MNIST" ]]
		then
			cd WaNet
			source ~/anaconda3/etc/profile.d/conda.sh
			conda activate aml
			python eval.py --dataset mnist --attack_mode all2one --bs 64 --lr_C 0.01 --target_label 1 
			conda deactivate
			cd ..
		elif [[ $attack == "WaNet" ]] && [[ $dataset == "CIFAR-10" ]]
		then
			cd WaNet
			source ~/anaconda3/etc/profile.d/conda.sh
			conda activate aml
			python eval.py --dataset cifar10 --attack_mode all2one --bs 64 --lr_C 0.01 --target_label 1 
			conda deactivate
			cd ..
		elif [[ $attack == "FIBA" ]] && [[ $dataset == "MNIST" ]]
		then
			cd FIBA
			source ~/anaconda3/etc/profile.d/conda.sh
			conda activate fiba
			python eval.py --device cuda:0 --dataset mnist --test_model ./checkpoints/mnist/all2one0/best_acc_bd_ckpt.pth.tar --lr_C 0.01 --target_label 1 --n_iters 100 --target_img trigger.jpg --cross_dir ./noise_img 
			conda deactivate
			cd ..
		elif [[ $attack == "FIBA" ]] && [[ $dataset == "CIFAR-10" ]]
		then
			cd FIBA
			source ~/anaconda3/etc/profile.d/conda.sh
			conda activate fiba
			python eval.py --device cuda:0 --dataset cifar10 --test_model ./checkpoints/cifar10/all2one0/best_acc_bd_ckpt.pth.tar --lr_C 0.01 --target_label 1 --n_iters 100 --target_img trigger.jpg --cross_dir ./noise_img 
			conda deactivate
			cd ..
		elif [[ $attack == "BlindBd" ]] && [[ $dataset == "MNIST" ]]
		then
			cd BlindBackdoor
			source ~/anaconda3/etc/profile.d/conda.sh
			conda activate aml
			python training.py --name mnist --params configs/mnist_params.yaml --commit none
			conda deactivate
			cd ..
		elif [[ $attack == "BlindBd" ]] && [[ $dataset == "CIFAR-10" ]]
		then
			cd BlindBackdoor
			source ~/anaconda3/etc/profile.d/conda.sh
			conda activate aml
			python training.py --name cifar10 --params configs/cifar10_params.yaml --commit none
			conda deactivate
			cd ..
		elif [[ $attack == "LIRA" ]] && [[ $dataset == "MNIST" ]]
		then
			cd LIRA
			source ~/anaconda3/etc/profile.d/conda.sh
			conda activate lira
			python paddle/lira_backdoor_injection.py --dataset mnist --clsmodel mnist_cnn --data_root ../../data/MNIST --path experiments/ --epochs 100 --mode all2one --target_label 1 --batch-size 64 --lr 0.01 --lr-atk 0.01 --test_epochs 100 
			conda deactivate
			cd ..
		elif [[ $attack == "LIRA" ]] && [[ $dataset == "CIFAR-10" ]]
		then
			cd LIRA
			source ~/anaconda3/etc/profile.d/conda.sh
			conda activate lira
			python paddle/lira_backdoor_injection.py --dataset cifar10 --data_root ../../data --clsmodel vgg11 --path experiments/ --epochs 100 --train-epoch 1 --mode all2one --target_label 1 --epochs_per_external_eval 10 --cls_test_epochs 5 --verbose 2 --batch-size 64 --alpha 0.5 --eps 0.01 --avoid_clsmodel_reinitialization --test_eps 0.01 --test_alpha 0.5 --test_epochs 100 --test_lr 0.01 --lr-atk 0.01  --schedulerC_lambda 0.1 --schedulerC_milestones 50,100,150,200 --test_use_train_best 
			conda deactivate
			cd ..
		fi
	done
done

