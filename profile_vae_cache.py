"""Offline A0 VAE-cache resource profile. No transformer or optimizer."""
import ast
import argparse
import hashlib
import json
import os
import resource
import time
from pathlib import Path
from typing import Optional
import numpy as np
import torch
import torchvision.transforms as TT
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize
from torch.utils.data import Dataset
from tqdm import tqdm
from diffusers import AutoencoderKLCogVideoX


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True)
    args=ap.parse_args();root=args.root
    output=root/'profile';output.mkdir(exist_ok=False)
    torch.set_num_threads(2)
    source=root/'trainer.py';tree=ast.parse(source.read_text())
    node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='VideoDataset')
    ns=dict(Dataset=Dataset,Optional=Optional,Path=Path,np=np,torch=torch,TT=TT,
        InterpolationMode=InterpolationMode,resize=resize,tqdm=tqdm)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),ns)
    data=ns['VideoDataset'](instance_data_root='/home/ubuntu/nkst_paired_data_v1/A0',
        caption_column='prompts.txt',video_column='videos.txt',height=480,width=720,fps=8,max_num_frames=49)
    assert len(data)==500
    model='/home/ubuntu/hf_cache/hub/models--THUDM--CogVideoX-2b/snapshots/1137dacfc2c9c012bed6a0793f4ecf2ca8e7ba01'
    vae=AutoencoderKLCogVideoX.from_pretrained(model,subfolder='vae',torch_dtype=torch.float16,local_files_only=True)
    vae.requires_grad_(False);vae.eval();vae.to('cuda');vae.enable_slicing();vae.enable_tiling()
    cache=[];start=time.monotonic();torch.cuda.reset_peak_memory_stats()
    cache_bytes=0;rows=[]
    with (output/'progress.jsonl').open('x') as log:
        for i in range(len(data)):
            # Guard the host because the cache remains resident, like training.
            available=int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))*1024
            if available<4*1024**3:raise RuntimeError('Host RAM reserve below 4GiB')
            step=time.monotonic()
            video=data._decode_video(i)
            with torch.no_grad():
                distribution=vae.encode(video.to('cuda',dtype=vae.dtype).unsqueeze(0).permute(0,2,1,3,4)).latent_dist
            for name,value in list(vars(distribution).items()):
                if isinstance(value,torch.Tensor):setattr(distribution,name,value.detach().cpu().half())
            tensors=[v for v in vars(distribution).values() if isinstance(v,torch.Tensor)]
            if not all(torch.isfinite(t).all() for t in tensors):raise RuntimeError('Nonfinite latent cache')
            storages={t.untyped_storage().data_ptr():t.untyped_storage().nbytes() for t in tensors}
            cache_bytes+=sum(storages.values());cache.append(distribution)
            torch.cuda.synchronize();del video
            row=dict(index=i,seconds=time.monotonic()-step,cache_bytes=cache_bytes,
                rss_peak_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                gpu_peak_allocated=torch.cuda.max_memory_allocated(),gpu_peak_reserved=torch.cuda.max_memory_reserved())
            rows.append(row);log.write(json.dumps(row)+'\n');log.flush()
            if (i+1)%25==0:print(i+1,'encoded',flush=True)
    report=dict(status='completed',videos_encoded=len(cache),seconds=time.monotonic()-start,
        **{k:v for k,v in rows[-1].items() if k not in ('index','seconds')},
        arm='A0',transformer_loaded=False,optimizer_started=False,cache_saved_to_disk=False,
        trainer_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),torch=torch.__version__)
    (output/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':main()
