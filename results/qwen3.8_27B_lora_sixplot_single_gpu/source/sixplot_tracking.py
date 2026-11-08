"""Offline W&B metrics and a local, continuously updated loss plot."""
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
(ROOT/'.cache/matplotlib').mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(ROOT/'.venv/wandb-packages'))


class Tracker:
    def __init__(self,directory,config,name,group):
        import wandb
        self.directory=Path(directory);self.directory.mkdir(parents=True,exist_ok=True)
        tracking=ROOT/'results/qwen3.8_27B_lora_sixplot_single_gpu/tracking'
        for key,suffix in [('WANDB_CACHE_DIR','cache'),('WANDB_CONFIG_DIR','config'),('WANDB_DATA_DIR','data')]:
            os.environ[key]=str(tracking/suffix);(tracking/suffix).mkdir(parents=True,exist_ok=True)
        # User explicitly selected offline recording. No images, data, weights or source uploads.
        self.run=wandb.init(project='pet-movement',group=group,name=name,mode='offline',dir=str(self.directory),
                config=config,settings=wandb.Settings(disable_code=True,disable_git=True,console='off',x_disable_stats=True))
        self.run.define_metric('optimizer_step')
        for pattern in ['train/*','validation/*','memory/*']:self.run.define_metric(pattern,step_metric='optimizer_step')
        self.rows=[]
        self.log_path=self.directory/'metrics.jsonl'
        (self.directory/'wandb_run.json').write_text(json.dumps({'mode':'offline','id':self.run.id,'project':'pet-movement',
           'group':group,'name':name,'files_dir':self.run.dir,'website_url':None,'sdk_version':wandb.__version__},indent=2)+'\n')

    def log(self,values):
        self.run.log(values)
        with self.log_path.open('a') as f:f.write(json.dumps(values)+'\n')
        if 'train/loss' in values:
            self.rows.append(values)
            if len(self.rows)==1 or len(self.rows)%20==0:self.plot()

    def plot(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        if not self.rows:return
        x=[r['optimizer_step'] for r in self.rows];y=[r['train/loss'] for r in self.rows]
        fig,ax=plt.subplots(figsize=(9,4),dpi=130)
        ax.plot(x,y,color='#0072B2',alpha=.4,label='Per-update answer-token CE')
        smooth=[];total=0.
        for i,v in enumerate(y):
            total+=v
            if i>=20:total-=y[i-20]
            smooth.append(total/min(i+1,20))
        ax.plot(x,smooth,color='#0072B2',label='20-update mean');ax.set(xlabel='Optimizer update',ylabel='Training loss',title=self.run.name)
        ax.legend();ax.grid(alpha=.2);fig.tight_layout()
        tmp=self.directory/'loss_curve.tmp.png';fig.savefig(tmp);tmp.replace(self.directory/'loss_curve.png');plt.close(fig)

    def finish(self,summary):
        self.plot();self.run.summary.update(summary);self.run.finish()
