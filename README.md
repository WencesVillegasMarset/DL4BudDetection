# Deep Learning for 2D Grapevine Bud Detection

Pre-print available at: https://arxiv.org/abs/2008.11872

Data: 

* Classification and segmentation data [here](http://dharma.frm.utn.edu.ar/vise/bc/) 

* Pre-processed data [here](https://drive.google.com/file/d/1e4Vmknt5hWaWGSOD5kfxuq6w_QdNCnrn/view?usp=sharing).

### Running training and inference scripts.

* Install dependencies 

```console
bash:~$ pip install -r requirements.txt
```

* Download and extract pre-processed data on /images directory or just run the following get_data.py script 

```console
bash:~$ python get_data.py 
```

* Run training scripts for all architectures (sample call shows default values)

```console
bash:~$ python train.py --epochs 200 --lr 0.0001 --bs 4 --savemodel --csv train.csv --imgpath ./images
```

* Run inference scripts for a desired model, generating prediction masks for each threshold and optionally (--valid flag) generating a csv component report (call shows sample values, --model must be specified)

```console
bash:~$ python inference.py --model FCMN8rmsprop_lr0.0001_prep_keras_dp0.001_ep200.h5 --output ./output/validation/ --csv test.csv --imgpath ./images --valid
```