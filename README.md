# Aging-Aware Training for Printed Neuromorphic Circuits

This github repository is for the paper at ICCAD'22 - Aging-Aware Training for Printed Neuromorphic Circuits

cite as
```
Aging-Aware Training for Printed Neuromorphic Circuits
Zhao, H.; Hefenbrock, M.; Beigl, M.; Tahoori, M.
2022 International Conference on Computer-Aided Design (ICCAD), October, 2022 IEEE/ACM.
```



Usage of the code:

1. Training of printed neural networks

~~~
$ sh experiment_ICCAD_2022.sh
~~~

Alternatively, the experiments can be conducted by running command lines in `experiment_ICCAD_2022.sh` separately, e.g.,

~~~
$ python3 experiment.py --DATASET 0  --SEED 0  --MODE nominal --projectname ICCAD_2022
$ python3 experiment.py --DATASET 0  --SEED 1  --MODE nominal --projectname ICCAD_2022
...
~~~



2.   After training printed neural networks, the trained networks are in `./ICCAD_2022/model/`, the log files for training can be found in `./ICCAD_2022/log/`. If there is still files in `./ICCAD_2022/temp/`, you should run the corresponding command line to train the networks further. Note that, each training is limited to 48 hours, you can change this time limitation in `configuration.py`



3.   Evaluation can be done by running the `evaluation_ICCAD_2022.sh` in `./ICCAD_2022/` folder with

~~~
$ sh evaluation_ICCAD_2022.sh
~~~

 Of course, each line in this file can be run separately as in step 1.



4.   For visualization, run

~~~
$ python3 visualization.py
~~~

