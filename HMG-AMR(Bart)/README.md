# HMG-AMR(Bart)


## Installation
```shell script
cd spring
pip install -r requirements.txt
pip install -e .
```

The code only works with `transformers` < 3.0 because of a disrupting change in positional embeddings.
The code works fine with `torch` 1.5. We recommend the usage of a new `conda` env.

## Train
Modify `config.yaml` in `configs`. 

### Text-to-AMR
```shell script
python bin/train.py --config configs/config.yaml --direction amr
```
Results in `runs/`




## Evaluate
### Text-to-AMR
```shell script
python bin/predict_amrs.py \
    --datasets <AMR-ROOT>/data/amrs/split/test/*.txt \
    --gold-path data/tmp/amr2.0/gold.amr.txt \
    --pred-path data/tmp/amr2.0/pred.amr.txt \
    --checkpoint runs/<checkpoint>.pt \
    --beam-size 5 \
    --batch-size 500 \
    --device cuda \
    --penman-linearization --use-pointer-tokens
```
`gold.amr.txt` and `pred.amr.txt` will contain, respectively, the concatenated gold and the predictions.

To reproduce our paper's results, you will also need need to run the [BLINK](https://github.com/facebookresearch/BLINK) 
entity linking system on the prediction file (`data/tmp/amr2.0/pred.amr.txt` in the previous code snippet). 
To do so, you will need to install BLINK, and download their models:
```shell script
git clone https://github.com/facebookresearch/BLINK.git
cd BLINK
pip install -r requirements.txt
sh download_blink_models.sh
cd models
wget http://dl.fbaipublicfiles.com/BLINK//faiss_flat_index.pkl
cd ../..
```
Then, you will be able to launch the `blinkify.py` script:
```shell
python bin/blinkify.py \
    --datasets data/tmp/amr2.0/pred.amr.txt \
    --out data/tmp/amr2.0/pred.amr.blinkified.txt \
    --device cuda \
    --blink-models-dir BLINK/models
```







## Acknowledgements
Our code is based on [Spring](https://github.com/SapienzaNLP/spring). Thanks for their high quality open codebase.

