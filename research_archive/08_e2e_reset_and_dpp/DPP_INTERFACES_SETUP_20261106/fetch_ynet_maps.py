"""Read the author-linked ZIP selectively via verified HTTP byte ranges.

Never execute archive code/pickle or extract paths. Only map/H assets are allowed.
The original ZIP is ~949 MiB; selected assets only are downloaded.
"""
import argparse,hashlib,io,json,re,urllib.request,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent
URL='https://drive.usercontent.google.com/download?id=1u4hTk_BZGq1929IxMPLCrDzoG3wsZnsa&export=download&confirm=t'
SOURCE='https://raw.githubusercontent.com/HarshayuGirase/Human-Path-Prediction/master/ynet/README.md'

class RangeReader(io.RawIOBase):
    def __init__(self,url):
        self.url=url;self.pos=0;self.total_read=0
        with urllib.request.urlopen(urllib.request.Request(url,headers={'Range':'bytes=0-0'}),timeout=30) as r:
            header=r.headers.get('Content-Range','')
            if r.status!=206 or not re.fullmatch(r'bytes 0-0/\d+',header):raise ValueError('RANGE_NOT_SUPPORTED')
            self.length=int(header.split('/')[-1]);r.read(1)
    def readable(self):return True
    def seekable(self):return True
    def tell(self):return self.pos
    def seek(self,offset,whence=0):
        self.pos=offset if whence==0 else self.pos+offset if whence==1 else self.length+offset
        if not 0<=self.pos<=self.length:raise ValueError('ZIP_SEEK_OUTSIDE')
        return self.pos
    def read(self,size=-1):
        size=self.length-self.pos if size<0 else min(size,self.length-self.pos)
        if size==0:return b''
        if size>32*1024**2:raise ValueError('SINGLE_RANGE_LIMIT_32M')
        end=self.pos+size-1
        with urllib.request.urlopen(urllib.request.Request(self.url,headers={'Range':f'bytes={self.pos}-{end}'}),timeout=40) as r:
            if r.status!=206 or r.headers.get('Content-Range')!=f'bytes {self.pos}-{end}/{self.length}':raise ValueError('WRONG_RANGE')
            data=r.read(size+1)
        if len(data)!=size:raise ValueError('RANGE_LENGTH')
        self.pos+=size;self.total_read+=size;return data

def main(download=False):
    reader=RangeReader(URL)
    with zipfile.ZipFile(reader) as archive:
        entries=[dict(name=x.filename,size=x.file_size,compressed=x.compress_size,CRC=x.CRC) for x in archive.infolist() if 'eth' in x.filename.lower() or 'ucy' in x.filename.lower()]
        print(json.dumps(entries,indent=2))
        if not download:return
        dest=ROOT/'official_ynet_assets';dest.mkdir(exist_ok=False)
        selected=[]
        for info in archive.infolist():
            name=info.filename;low=name.lower()
            if not ('eth_ucy' in low and low.endswith(('.png','.jpg','.jpeg','_h.txt'))):continue
            if info.file_size>32*1024**2 or info.compress_size>32*1024**2:raise ValueError('MAP_ASSET_TOO_LARGE')
            # Flatten by content index, retain original archive path in receipt.
            out=dest/(f'{len(selected):03d}_'+Path(name).name)
            data=archive.read(info) # zipfile verifies CRC before returning.
            with out.open('xb') as f:f.write(data)
            selected.append(dict(archive_path=name,path=str(out.resolve()),sha256=hashlib.sha256(data).hexdigest(),bytes=len(data),CRC=info.CRC))
        if not selected:raise ValueError('NO_MAP_ASSETS_FOUND')
        receipt=dict(status='DOWNLOADED_OFFICIAL_ASSETS_NOT_SCIENTIFIC_MAP_ADMISSION',source_readme=SOURCE,url=URL,archive_bytes=reader.length,network_bytes=reader.total_read,files=selected,no_checkpoint_pickle_or_archive_code_executed=True)
        with (ROOT/'YNET_DOWNLOAD.json').open('x',encoding='utf-8') as f:json.dump(receipt,f,ensure_ascii=False,indent=2)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--download',action='store_true');main(p.parse_args().download)
