# HMG-AMR(T5)


## Installation

    pip install -r  requirements.txt

## Train the models and obtain prediction


### T5 Training


1. Preprocess the data to remove wiki tags. Wiki tags point to reference in Wikipedia, this models perform wikification using Blink as a postprocessing step.
   
   The following command shows how to preprocess AMR 2.0. It can be performed in a similar way on the other datasets.
        
        python -u -m amr_utils.preprocess.preprocess_amr -i LDC2017T10/data/amrs/split \
            -o LDC2017T10/preprocessed_data/

2. Train T5 models and obtaining predictions

        python -u -m amr_parsing.t5.cli.train --train "./LDC2017T10/preprocessed_data/train.txt.features.nowiki" \
            --validation ./LDC2017T10/preprocessed_data/dev.txt.features.nowiki \
            --report_test ./LDC2017T10/preprocessed_data/test.txt.features.nowiki \
            --max_source_length 512 --max_target_length 512 --batch 8 -e 30 -m t5-large \
            --model_type t5 --output ./t5_amr/ --data_type "amrdata" \
            --task_type "text2amr" --val_from_epoch 10

   

#### T5 obtaining predictions from a checkpoint
1. Scoring

         python -u -m amr_parsing.cli.parser --test LDC2017T10/preprocessed_data/test.txt.features.nowiki \
             --max_source_length 512 --max_target_length 512 --batch 4 -m t5-large --model_type t5  \
             --output LDC2017T10/preprocessed_data/t5_amr_prediction.txt --data_type "amrdata" --task_type "text2amr" \
             --checkpoint t5_amr/multitask.model
   
2. Wikification

    To reproduce our results, you will also need need to run the [BLINK](https://github.com/facebookresearch/BLINK) 
    entity linking system on the prediction file. To do so, you will need to install BLINK, and download their models:
    ```shell script
    git clone https://github.com/facebookresearch/BLINK.git
    cd BLINK
    pip install -r requirements.txt
    pip install -e .
    sh download_blink_models.sh
    cd models
    wget http://dl.fbaipublicfiles.com/BLINK//faiss_flat_index.pkl
    cd ../..
    ```
    Then, you will be able to launch the `run_blink_wiki_adder.py` script:
    ```shell
    python -u -m amr_utils.blinkify.run_blink_wiki_adder.py \
    -i LDC2017T10/preprocessed_data/t5_amr_prediction.txt \ 
    -o LDC2017T10/preprocessed_data/ \
    --blink-models-dir ../BLINK/models/ 

The output file with wikifications will be written to the output folder.