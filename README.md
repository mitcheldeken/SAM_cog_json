## Instructions for pushing this model to replicate
First, find a Linux or MacOS instance with GPU.
On the instance, install cog:
```
sudo curl -o /usr/local/bin/cog -L https://github.com/replicate/cog/releases/latest/download/cog_`uname -s`_`uname -m`
sudo chmod +x /usr/local/bin/cog
```
Clone this directory with:
```
git clone https://github.com/mitcheldeken/SAM_cog_json.git
cd SAM_cog_json
```
<!-- Download model weights:
```
WEIGHTS_URL=https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth
curl -O $WEIGHTS_URL
``` -->
Push model to replicate:
```
cog login
cog push r8.im/[your-username]/[your-model]
```