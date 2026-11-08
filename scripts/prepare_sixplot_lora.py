"""Add train/validation six-channel plots with the unchanged historical renderer."""
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import render_imu as renderer

ROOT=renderer.ROOT
OUT=ROOT/'dataset/vlm/supervised'


def init_worker():
    global X,PROTOCOL,FIG,LINES,POINTS
    X=np.load(ROOT/'dataset/cache/pet_windows.npz')['X']
    PROTOCOL=json.loads((renderer.BASE/'render_protocol.json').read_text())
    FIG,LINES,POINTS=renderer.figure(PROTOCOL)


def draw(chunk):
    output=[]
    for row in chunk:
        row=dict(row);index=row.pop('cache_index')
        path=OUT/'images'/('sample_'+hashlib.sha256(row['window_id'].encode()).hexdigest()[:20]+'.webp')
        info=renderer.render(X[index],path,PROTOCOL,FIG,LINES,POINTS)
        output.append({**row,'image_file':str(path.relative_to(ROOT)),**info})
    return output


def main(limit=None):
    OUT.mkdir(exist_ok=True);(OUT/'images').mkdir(exist_ok=True)
    w,x,p=renderer.load_windows()
    selected=w[w.split.isin(['train','validation'])].copy()
    selected.insert(0,'cache_index',selected.index)
    if limit:selected=selected.head(limit)
    dest=OUT/'images.csv';done={}
    if dest.exists():
        for r in pd.read_csv(dest).to_dict('records'):
            assert renderer.sha(ROOT/r['image_file'])==r['image_sha256']
            done[r['window_id']]=r
    jobs=[r for r in selected.to_dict('records') if r['window_id'] not in done]
    fields=[c for c in selected.columns if c!='cache_index']+['image_file','render_seconds','image_sha256']
    for c in p['channels']:fields.extend(c+s for s in ['_below','_above','_min','_max'])
    mode='a' if dest.exists() else 'w'
    with dest.open(mode,newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields)
        if mode=='w':writer.writeheader()
        chunks=[jobs[i:i+32] for i in range(0,len(jobs),32)]
        with ProcessPoolExecutor(max_workers=8,initializer=init_worker) as pool:
            for chunk in pool.map(draw,chunks):
                writer.writerows(chunk);f.flush();done.update({r['window_id']:r for r in chunk})
                print(f'SIXPLOT {len(done)}/21188',flush=True)
    if not limit:
        assert len(done)==21188 and set(done)==set(selected.window_id)
        check={'train_windows':16461,'validation_windows':4727,'manifest_sha256':renderer.sha(dest),
               'renderer_sha256':renderer.sha(Path(renderer.__file__)),
               'render_protocol_sha256':renderer.sha(renderer.BASE/'render_protocol.json'),
               'rule':'Historical six-plot renderer unchanged; training-fixed axes, 1280x1152, all200 points/6axes; no new test assets.'}
        (OUT/'validation.json').write_text(json.dumps(check,indent=2)+'\n')


if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--limit',type=int)
    main(a.parse_args().limit)
