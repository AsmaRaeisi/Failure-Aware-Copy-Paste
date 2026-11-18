
#install

conda create -n instaboostfast python=3.10 -y
conda activate instaboostfast

cd instaboostfast_official_labelme_wrapper
pip install -r requirements.txt


## Run 


python generate_instaboostfast_labelme_multiclass.py \
    --images "/path/train/images" \
    --json "/path/train/json" \
    --output "/path/output_instaboost" \
    --class-map "/path/classes.json" \
    --overwrite
