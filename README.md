sudo curl -o /usr/local/bin/cog -L https://github.com/replicate/cog/releases/latest/download/cog_`uname -s`_`uname -m`
sudo chmod +x /usr/local/bin/cog
mkdir segment-anything
cd segment-anything
cog init
---> copy and paste text of predict.py and cog.yaml
WEIGHTS_URL=https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth
curl -O $WEIGHTS_URL
cog login
cog push r8.im/mitcheldeken/segmentanything_masks