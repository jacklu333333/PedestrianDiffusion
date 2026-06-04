### Basic Usage

1. train cmd

```shell
python trainScript.py 2>&1 | tee diffusion_test_$(hostname).log
```

```shell
python trainScript.py --model PedestrianDiffusion  2>&1 | tee diffusion_test_$(hostname).log
```

- To replicate IoNet, RoNINResNet, RoNINLSTM, RoNINTCN, TLIO, LLIO.

```shell
python trainScript.py --model IoNetModule --encoding False -d RoNINs -vae_weight "" 2>&1 | tee diffusion_test_$(hostname).log
python trainScript.py --model RoNINResNetModule --encoding False -d RoNINs -vae_weight "" 2>&1 | tee diffusion_test_$(hostname).log
python trainScript.py --model RoNINLSTMModule --encoding False -d RoNINs -vae_weight "" 2>&1 | tee diffusion_test_$(hostname).log
python trainScript.py --model RoNINTCNModule --encoding False -d RoNINs -vae_weight "" 2>&1 | tee diffusion_test_$(hostname).log
python trainScript.py --model TLIOModule --encoding False -d RoNINs -vae_weight "" 2>&1 | tee diffusion_test_$(hostname).log
```

- To replicate PedestrianDiffusion multi-datasets version

```shell
python trainScript.py --config config.json 2>&1 | tee diffusion_test_$(hostname).log
```

- To replicate PedestrianDiffusion RoNIN_sole version

```shell
python trainScript.py --config config.json -d RoNIN 2>&1 | tee diffusion_test_$(hostname).log
```

-Notice: The final result is 3d version. please use the following cmd to compute the 2d version.

```shell
python recompute_metrics.py /[path to version/] --2d
```

- To reproduce VAESpectrum3D from scratch.

```shell
python trainScriptOdom_VAE_imu.py --config config.json --encoding False -b 256 --vae__weight_path ""  2>&1 | tee diffusion_test_$(hostname).log
```

1. test cmd

```shell
python testScript.py -d hybrid -b 256 --model PedestrianDiffusion  -w [weight_path] 2>&1 | tee diffusion_test_$(hostname).log
```

* To test the unknon v.s. uknown, please go to ./utils/mdatasets/utils.py and manually comment line 382:398 and comment out line 350:358. Then run the testScript accordingly.

### Notice

1. Don't modify the default configuration file since it is use as the template to identify required parameters. in the cmd.
2. If must want to modify the configuration file. Please copy one and specified the alternative configuration file path. The program will overwrite it.
3. If you use both cmd parameter in the command line and alternative file, the program will first overwrite the template with alternative file and then overwrite with the cmd specification.

### Environment Setup

```shell
./envsetup.sh
```

### Datasets

1. please download the datasets and organize into such order
2. Other than the RIDI and OxIOD, all datasets have the default train, val, test set indices. So please copy the train, val, and test list for RIDI to the datasets directory of RIDI.

```
./datasets/RIDI
├── datasets
├── note.md
├── test_list.txt
├── train_list.txt
└── val_list.txt

./datasets/RoNIN
├── CITATION.txt
├── Data
├── frdr-checksums-and-filetypes.md
├── LICENSE.txt
├── lists
├── Pretrained_Models
└── README.txt

./datasets/TLIO
└── tlio_golden

./datasets/OxIOD
├── datasets
└── lists
    ├── test_list.txt
    ├── train_list.txt
    └── val_list.txt

```


### Notice
* This version of code may constinas functions haven't verified yet. We will provide more complete, verified code on Github once the review process is over.
* Some functions which are not mentioned above will work. But for the purpose of safe usage, please stay within the scope of the above command. 
* Many configuration in the config.json file are deprecated. Adjusting it might not change anything, and some some parameter only works for certain models.

