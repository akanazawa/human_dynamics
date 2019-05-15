import json
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--savedir',help='directory to save downloaded InstaVariety videos')
args = parser.parse_args()

savedir = args.savedir
# create the save directory if it doesn't exist
os.system('mkdir -p {}'.format(savedir))

## NOTE: this assumes that you're running the script from the 
## directory in which it's located. if not, you need to change
## the path of the InstaVariety.json file 
with open('InstaVariety.json','rb') as f:
    insta_data = json.load(f)

for post in insta_data:
    # make the save directory for the current video, if not exist
    savedir_p = '{}/{}'.format(savedir,post['download_tag'])
    os.system('mkdir -p {}'.format(savedir_p))
    dl_link = post['video_link']
    dl_name = post['urls'][0]

    print('downloading {}'.format(dl_link))
    os.system('youtube-dl {} --output {}/{}'.format(dl_link,savedir_p,dl_name))
