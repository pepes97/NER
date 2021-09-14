import os
import gdown

def download_dataset():
    URL = "https://drive.google.com/uc?export=download&id=1-FOOKAdTYFkSVKTpfAwQ7gL-BWi5SxbU"
    gdown.download(URL, os.getcwd()+os.sep+"wiki-news-300d-1M.vec", quiet=False)
    print("\n")

if __name__ == '__main__':
    download_dataset()