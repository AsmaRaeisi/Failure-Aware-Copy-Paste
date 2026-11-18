
## Install

```bash
conda create -n instaboostfast python=3.10 -y
conda activate instaboostfast

cd instaboostfast_official_labelme_wrapper
pip install -r requirements.txt
```

## Run 



```bash
python generate_instaboostfast_labelme.py \
  --images "/path/to/train_images" \
  --json "/path/to/train_annotations" \
  --output "/path/to/instaboostfast_output" \
  --seed 42 \
  --heatmap \
  --overwrite
```
