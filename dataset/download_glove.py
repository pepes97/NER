import os
import gdown

def download_dataset():
    URL = "https://drive.google.com/uc?export=download&id=1-6gxK7kQODberN-XMP7VWH9QFFcC2nL5"
    gdown.download(URL, os.getcwd()+os.sep+"glove.6B.300d.txt", quiet=False)
    print("\n")

if __name__ == '__main__':
    download_dataset()