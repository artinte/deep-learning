import hashlib
import os
import requests

def download(url, folder='./data', sha1_hash=None):
    '''
    Download a file to folder and return the local filepath.
    '''
    os.makedirs(folder, exist_ok=True)
    fname = os.path.join(folder, url.split('/')[-1])
    # check cache
    if os.path.exists(fname) and sha1_hash:
        sha1 = hashlib.sha1()
        with open(fname, 'rb') as f:
            while True:
                data = f.read(1024 * 1024)
                if not data:
                    break
                sha1.update(data)
        if sha1.hexdigest() == sha1_hash:
            return fname
        else:
            print(sha1.hexdigest())
    # download
    print(f'Downloading {fname} from {url} ...')
    r = requests.get(url, stream=True, verify=True)
    with open(fname, 'wb') as f:
        f.write(r.content)
    return fname
