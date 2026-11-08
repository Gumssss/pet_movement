"""By My Eyes-inspired single shared-axis IMU waveforms; training-only adaptation."""
import argparse,csv,hashlib,json,time
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import pandas as pd
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'dataset/vlm_acl'
COLORS=['#0072B2','#D55E00','#009E73','#CC79A7','#E69F00','#332288']
AUTHOR_COMMIT='c45ed732dc57d7823a311cab6e93d292b5ed32a1'

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def save_fixed(path,value):
    if path.exists():assert json.loads(path.read_text())==value,'Frozen configuration differs: '+str(path)
    else:path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')

def setup():
    w=pd.read_csv(ROOT/'dataset/splits/windows.csv')
    z=np.load(ROOT/'dataset/cache/pet_windows.npz');x=z['X']
    p=json.loads((ROOT/'results/protocol.json').read_text())
    assert len(w)==len(x)==24896
    assert np.array_equal(z['dogs'],w.dog_id) and np.array_equal(z['tests'],w.session)
    assert np.allclose(z['starts'],w.start_sec) and np.array_equal(np.array(p['classes'])[z['y']],w.label)
    train=x[w.split.eq('train')].reshape(-1,6).astype('float64')
    mean=train.mean(0);std=np.maximum(train.std(0),1e-6)
    normalized=(train-mean)/std
    q=np.quantile(normalized,[.001,.999],axis=0)
    low=float(q[0].min());high=float(q[1].max());pad=.05*(high-low)
    stats={'fit_split':'train','fit_windows':16461,'fit_dogs':27,'fit_samples_per_channel':len(train),
           'channels':p['channels'],'mean':mean.tolist(),'std':std.tolist(),'ddof':0,
           'std_floor':1e-6,'computation_dtype':'float64',
           'rule':'one global per-channel training mean/std; reused for all dogs and windows; no target-dog or per-window fitting'}
    save_fixed(BASE/'normalization.json',stats)
    protocol={
        'version':'imu_2s_acl_shared_axis_v1','experiment':'imu_2s_acl_shared_axis_zero_shot_en',
        'paper_url':'https://aclanthology.org/2024.emnlp-main.133/',
        'author_code_commit':AUTHOR_COMMIT,
        'author_code_url':f'https://github.com/diamond264/ByMyEyes/blob/{AUTHOR_COMMIT}/core/visualizer.py',
        'method_scope':'single shared-axis normalized waveform adaptation; no automatic visualization search or few-shot examples',
        'window_manifest_sha256':sha(ROOT/'dataset/splits/windows.csv'),
        'split_sha256':sha(ROOT/'dataset/splits/dog_splits.json'),
        'normalization_file':'normalization.json','normalization_sha256':sha(BASE/'normalization.json'),
        'research_file':'method.md','processor_check_file':'processor_check_qwen38.json',
        'channels':p['channels'],'colors':COLORS,'sampling_hz':100,'samples_per_window':200,
        'input_signal':'all six original channels, training-standardized before plotting; no magnitude or time-frequency conversion',
        'layout':'six colored traces overlaid on one shared x/y coordinate system with channel legend',
        'scale_fit_split':'train','scale_fit_windows':16461,
        'range_rule':'min of per-channel training z-score 0.001 quantiles, max of 0.999 quantiles; add 5% of the shared span on both sides',
        'per_channel_z_quantiles':q.tolist(),'shared_y_limits':[low-pad,high+pad],
        'out_of_range':'training-fixed shared limits; clipped display with colored boundary triangles; per-channel counts and standardized min/max logged',
        'width':512,'height':409,'dpi':100,'font':'DejaVu Sans','font_size_pt':10,
        'line_width_pt':1.5,'background':'white','time_axis_seconds':[0,2],
        'legend':{'location':'upper right','columns':2,'font_size_pt':7.5},
        'suptitle_font_size_pt':20,
        'image_format':'WebP lossless RGB; no private metadata',
        'test_windows':3708,'title':'target data','axes_title':'raw waveform',
        'filename_rule':'same opaque SHA256(window_id) identifier, in a separate image directory',
        'comparison_note':'Same Qwen3.8-27B, English behavioral priors, vLLM and test windows; changes include channel-wise training standardization, shared-axis overlay, paper-size image and matching visual-description text. This is not a full few-shot/VisGen reproduction.',
    }
    save_fixed(BASE/'render_protocol.json',protocol)
    return w,x,mean,std,protocol

def figure(p):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'figure.facecolor':'white','axes.facecolor':'white'})
    fig,ax=plt.subplots(figsize=(5.12,4.09),dpi=100)
    fig.subplots_adjust(left=.15,right=.98,bottom=.14,top=.82)
    fig.suptitle('target data',fontsize=20,y=.99)
    ax.set_title('raw waveform',fontsize=10)
    ax.set_xlabel('Time [sec]');ax.set_ylabel('Normalized value')
    ax.set_xlim(0,2);ax.set_ylim(*p['shared_y_limits'])
    ax.set_xticks(np.arange(0,2.01,.5));ax.set_yticks(np.linspace(*p['shared_y_limits'],5))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2g'))
    lines=[];markers=[]
    for channel,color in zip(p['channels'],COLORS):
        line,=ax.plot(np.arange(200)/100,np.zeros(200),color=color,linewidth=1.5,label=channel)
        lines.append(line)
        markers.append((ax.scatter([],[],marker='v',color=color,s=12,zorder=4),
                        ax.scatter([],[],marker='^',color=color,s=12,zorder=4)))
    ax.legend(loc='upper right',ncol=2,fontsize=7.5,framealpha=.85,columnspacing=.7,handlelength=1.5)
    return fig,lines,markers

def render(raw,path,mean,std,p,fig,lines,markers):
    started=time.perf_counter();values=(raw.astype('float64')-mean)/std
    low,high=p['shared_y_limits'];t=np.arange(200)/100;record={}
    for c,channel in enumerate(p['channels']):
        v=values[:,c];below=v<low;above=v>high
        lines[c].set_ydata(np.clip(v,low,high))
        markers[c][0].set_offsets(np.column_stack([t[below],np.full(below.sum(),low)]))
        markers[c][1].set_offsets(np.column_stack([t[above],np.full(above.sum(),high)]))
        record[channel+'_below']=int(below.sum());record[channel+'_above']=int(above.sum())
        record[channel+'_min']=float(v.min());record[channel+'_max']=float(v.max())
    fig.canvas.draw()
    image=Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy())
    assert image.size==(512,409)
    image.save(path,format='WEBP',lossless=True,method=4)
    record['render_seconds']=time.perf_counter()-started;record['image_sha256']=sha(path)
    return record

def prepare(preview_only=False):
    BASE.mkdir(exist_ok=True);(BASE/'images').mkdir(exist_ok=True)
    w,x,mean,std,p=setup();fig,lines,markers=figure(p)
    if not (BASE/'preview.webp').exists():
        first=int(np.flatnonzero(w.split.eq('train'))[0])
        render(x[first],BASE/'preview.webp',mean,std,p,fig,lines,markers)
    if preview_only:
        print('Training-only preview ready.',flush=True);plt.close(fig);return
    file=BASE/'images.csv';old=pd.read_csv(file).to_dict('records') if file.exists() else []
    done={r['window_id']:r for r in old}
    fields=['window_id','image_file','image_sha256','render_seconds']
    for c in p['channels']:fields.extend(c+s for s in ['_below','_above','_min','_max'])
    mode='a' if file.exists() else 'w'
    with file.open(mode,newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields)
        if mode=='w':writer.writeheader()
        for i in np.flatnonzero(w.split.eq('test')):
            wid=w.window_id.iloc[i];name='sample_'+hashlib.sha256(wid.encode()).hexdigest()[:20]+'.webp'
            path=BASE/'images'/name
            if wid in done:
                assert sha(path)==done[wid]['image_sha256'];continue
            info=render(x[i],path,mean,std,p,fig,lines,markers)
            row={'window_id':wid,'image_file':str(path.relative_to(ROOT)),**info}
            writer.writerow(row);stream.flush();done[wid]=row
            if len(done)%200==0:print(f'rendered ACL {len(done)}/3708',flush=True)
    assert len(done)==3708
    plt.close(fig);print('ACL-style rendering complete: 3708 fixed test windows.',flush=True)

def check_processor():
    from transformers import AutoProcessor
    proc=AutoProcessor.from_pretrained(ROOT/'models/Qwen3.8-27B',local_files_only=True)
    im=Image.open(BASE/'preview.webp').convert('RGB')
    batch=proc.image_processor(images=[im],return_tensors='pt')
    grid=batch['image_grid_thw'][0].tolist();_,gh,gw=grid
    patch=proc.image_processor.patch_size;merge=proc.image_processor.merge_size
    arr=batch['pixel_values'].numpy().reshape(gh//merge,gw//merge,merge,merge,3,2,patch,patch)
    arr=arr[:,:,:,:,:,0].transpose(4,0,2,5,1,3,6).reshape(3,gh*patch,gw*patch)
    mean=np.array(proc.image_processor.image_mean)[:,None,None];std=np.array(proc.image_processor.image_std)[:,None,None]
    rgb=np.rint(np.clip(arr*std+mean,0,1)*255).astype('uint8').transpose(1,2,0)
    Image.fromarray(rgb).save(BASE/'processor_preview_qwen38.webp',lossless=True)
    prompt=(BASE/'prompt.txt').read_text();assert prompt.isascii()
    c=[{'role':'user','content':[{'type':'image','image':im},{'type':'text','text':prompt}]}]
    encoded=proc.apply_chat_template(c,tokenize=True,return_dict=True,return_tensors='pt',add_generation_prompt=True,enable_thinking=False)
    image_id=proc.tokenizer.convert_tokens_to_ids('<|image_pad|>');tokens=int((encoded['input_ids']==image_id).sum())
    assert tokens==gh*gw//merge**2 and grid==[1,26,32]
    result={'3.8-27B':{'export_size':list(im.size),'processed_size':[gw*patch,gh*patch],
        'image_grid_thw':grid,'image_tokens':tokens,'total_input_tokens':int(encoded['input_ids'].shape[1]),
        'prompt_language':'en','prompt_sha256':sha(BASE/'prompt.txt'),
        'resize_note':'author-like 512x409 raster rounded by actual Qwen processor to 512x416; processed preview saved for visual inspection'}}
    save_fixed(BASE/'processor_check_qwen38.json',result);print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','check-processor'])
    parser.add_argument('--preview-only',action='store_true')
    args=parser.parse_args()
    prepare(args.preview_only) if args.command=='prepare' else check_processor()
