# Frank-Wolfe method in Neural Networks

This repository contains the code for my Bachelor's thesis. The implementation of the Frank-Wolfe method as neural 
network optimizer can be found in 'optimizers.py', and the implementation of different constraints (including their LMOs) 
in 'constraints.py'. The methods creating the data which is used in the different figures of the thesis are in the 
subdirectory 'data_construction'. For each figure, there is a method in the subdirectory 'figure_construction' which, 
given the output of its corresponding 'data_construction'-method as input, creates the figure.

To execute the code, do ``pip install -r requirements.txt``, and, to produce a specific graphic (or get the accuracy of 
SGD on different models), execute the main file. Afterward, you will be prompted to input the intended figure's number 
(or 'SGD').

There are pre-trained networks for all figures except 5.1 and 5.2 (networks trained using MSFW or SGD are in '/networks' 
and those trained with SFW are in '/networks-SFW'). By default, they will be used to produce figures quickly without 
having to train the networks. To train the networks anew, delete them from their corresponding directory.