"""Execute the trainer's unchanged Dataset AST without model imports/downloads."""
import ast
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Optional
from types import SimpleNamespace
import numpy as np
import torch
import torchvision.transforms as TT
from torchvision.transforms.functional import resize
from torchvision.transforms import InterpolationMode
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import decord

ROOT=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,default=ROOT/'p0_audit_results/training_data_dryrun_v1')
    args=parser.parse_args()
    torch.set_num_threads(2)
    source=ROOT/'phase3/train_cogvideox_lora.py'
    tree=ast.parse(source.read_text())
    node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='VideoDataset')
    namespace=dict(Dataset=Dataset,Optional=Optional,Path=Path,np=np,torch=torch,
        TT=TT,resize=resize,InterpolationMode=InterpolationMode,tqdm=tqdm)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),namespace)
    cls=namespace['VideoDataset']
    loader_call=next(n for n in ast.walk(tree) if isinstance(n,ast.Call)
        and isinstance(n.func,ast.Name) and n.func.id=='DataLoader')
    generator_node=next(k.value for k in loader_call.keywords if k.arg=='generator')
    generator_code=compile(ast.Expression(body=generator_node),str(source),'eval')
    out=args.out;out.mkdir(exist_ok=False)
    reports=[];orders={}
    for arm in ('A0','A1','A2','A3'):
        dataset=cls(instance_data_root=str(ROOT/'p0_audit_results/paired_video_smoke_v1'/arm),
            caption_column='prompts.txt',video_column='videos.txt',height=480,width=720,fps=8,max_num_frames=49)
        decoded=set();checks=[]
        for i,path in enumerate(dataset.instance_video_paths):
            assert dataset.instance_video_indices[i]==list(range(49))
            if str(path.resolve()) in decoded:continue
            decoded.add(str(path.resolve()))
            tensor=dataset._decode_video(i)
            assert tuple(tensor.shape)==(49,3,480,720)
            assert tensor.dtype==torch.float32 and tensor.is_contiguous() and torch.isfinite(tensor).all()
            assert tensor.min()>=-1 and tensor.max()<=1
            checks.append(dict(video=path.name,shape=list(tensor.shape),dtype=str(tensor.dtype),
                minimum=float(tensor.min()),maximum=float(tensor.max())))
            del tensor
        # __getitem__ expects VAE-produced latents. Use explicitly marked index
        # sentinels ONLY to exercise the original dataset and real shuffle order.
        dataset.instance_latents=list(range(len(dataset)))
        def order():
            loader=DataLoader(dataset,batch_size=1,shuffle=True,num_workers=0,
                generator=eval(generator_code,dict(torch=torch,args=SimpleNamespace(seed=42))))
            return [int(batch['instance_video'].item()) for batch in loader]
        indices=order();assert indices==order()
        assert sorted(indices)==list(range(len(dataset)))
        base_ids=[dataset.instance_video_paths[i].name.split('_')[0] for i in indices]
        orders[arm]=base_ids
        reports.append(dict(arm=arm,dataset_length=len(dataset),unique_video_decodes=checks,
            sample_indices=indices,base_order=base_ids,multiplicity=dict(Counter(base_ids))))
    assert orders['A0']==orders['A2'] and orders['A1']==orders['A3']
    report=dict(arms=reports,trainer_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        dataset_execution='unchanged class extracted by AST; not whole trainer import',
        sampler_execution='real CPU DataLoader with explicitly injected index sentinels, not VAE latents',
        generator_expression=ast.unparse(generator_node),
        versions=dict(torch=torch.__version__,decord=decord.__version__),
        vae_executed=False,accelerate_executed=False,training_started=False)
    (out/'results.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
