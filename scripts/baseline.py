"""Run: python baseline.py audit | train. Outputs only to explicitly configured paths."""
import os,sys,json,zipfile,hashlib,time,random,pickle,platform
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'dataset/犬只运动传感器数据集'
OUT=ROOT/'results'; MODELS=ROOT/'models'
CNN_OUT=OUT/'cnn'; RF_OUT=OUT/'random_forest'
SPLIT_DIR=ROOT/'dataset/splits'; CACHE_DIR=ROOT/'dataset/cache'
CACHE=CACHE_DIR/'pet_windows.npz'; AUDIT_CACHE=CACHE_DIR/'pet_audit.json'
SPLIT_CONFIG=json.loads((SPLIT_DIR/'dog_splits.json').read_text())
SEED=SPLIT_CONFIG['seed']; SPLITS=SPLIT_CONFIG['splits']
CHANNELS=['ANeck_x','ANeck_y','ANeck_z','GNeck_x','GNeck_y','GNeck_z']
CLASSES=['Galloping','Lying chest','Sitting','Sniffing','Standing','Trotting','Walking']

# Saved dog assignments are authoritative; never silently generate a new split.

def audit():
 CACHE_DIR.mkdir(parents=True,exist_ok=True)
 stats=Counter(); rawlabels=Counter(); support=Counter(); sessions=[]; xx=[]; yy=[]; dd=[]; tt=[]; ss=[]
 def process(df):
  dog=int(df.DogID.iloc[0]); test=int(df.TestNum.iloc[0]); t=df.t_sec.to_numpy(); x=df[CHANNELS].to_numpy(np.float32)
  a=df[['Behavior_1','Behavior_2','Behavior_3']].fillna('<undefined>').to_numpy(str)
  for row in a.T: rawlabels.update(row)
  hit=np.stack([(a==c).any(axis=1) for c in CLASSES],axis=1)
  for k,c in enumerate(CLASSES): support[(dog,c)]+=int(hit[:,k].sum())
  stats['sole_galloping_rows']+=int(((a=='Galloping').any(1)&((a!='<undefined>').sum(1)==1)).sum()); stats['rows']+=len(df); stats['no_target_rows']+=int((hit.sum(1)==0).sum()); stats['multiple_target_rows']+=int((hit.sum(1)>1).sum())
  stats['unannotated_rows']+=int((a=='<undefined>').all(1).sum()); stats['nonfinite_signal_rows']+=int((~np.isfinite(x).all(1)).sum())
  delta=np.diff(t); bad=np.where(~np.isclose(delta,.01,atol=1e-5))[0]+1
  stats['timestamp_breaks']+=len(bad); sessions.append({'dog':dog,'test':test,'rows':len(df),'breaks':len(bad)})
  for seg in np.split(np.arange(len(df)),bad):
   n=len(seg)//200
   stats['tail_rows']+=len(seg)%200
   if not n: continue
   inds=seg[:n*200].reshape(n,200); h=hit[inds]; v=x[inds]
   # Exactly one main behavior throughout 2 seconds. Auxiliary behaviors may coexist.
   valid=(h.sum(2)==1).all(1)&(h==h[:,:1,:]).all((1,2))&np.isfinite(v).all((1,2))
   stats['candidate_windows']+=n; stats['rejected_windows']+=int((~valid).sum())
   if valid.any():
    xx.append(v[valid]); yy.extend(h[valid,0].argmax(1)); dd.extend([dog]*int(valid.sum())); tt.extend([test]*int(valid.sum())); ss.extend(t[inds[valid,0]])
 with zipfile.ZipFile(DATA/'DogMoveData_csv_format.zip') as z:
  with z.open('DogMoveData.csv') as f:
   carry=None
   cols=['DogID','TestNum','t_sec']+CHANNELS+['Behavior_1','Behavior_2','Behavior_3']
   for chunk in pd.read_csv(f,usecols=cols,chunksize=400000):
    if carry is not None: chunk=pd.concat([carry,chunk],ignore_index=True)
    groups=list(chunk.groupby(['DogID','TestNum'],sort=False))
    for _,g in groups[:-1]: process(g)
    carry=groups[-1][1].copy()
   if carry is not None: process(carry)
 X=np.concatenate(xx); y=np.array(yy); dogs=np.array(dd); tests=np.array(tt); starts=np.array(ss)
 np.savez(CACHE,X=X,y=y,dogs=dogs,tests=tests,starts=starts)
 classes={c:{'rows_with_label':sum(n for (d,b),n in support.items() if b==c),'windows':int((y==k).sum()),'dogs':len(np.unique(dogs[y==k]))} for k,c in enumerate(CLASSES)}
 result={'stats':dict(stats),'classes':classes,'raw_labels':dict(rawlabels),'sessions':sessions,'csv_zip_sha256':hashlib.file_digest(open(DATA/'DogMoveData_csv_format.zip','rb'),'sha256').hexdigest()}
 AUDIT_CACHE.write_text(json.dumps(result,indent=2))
 print(json.dumps({'stats':dict(stats),'classes':classes,'dogs':len(np.unique(dogs)),'sessions':len(sessions)},indent=2),flush=True)


def features(x):
 # 9 statistics per channel, no target-dog standing calibration.
 return np.concatenate([x.mean(1),x.std(1),x.min(1),x.max(1),np.sqrt((x*x).mean(1)),np.quantile(x,.25,axis=1),np.median(x,axis=1),np.quantile(x,.75,axis=1),np.abs(np.diff(x,axis=1)).mean(1)],axis=1).astype('float32')

def train():
 import torch
 from torch import nn
 from sklearn.ensemble import RandomForestClassifier
 from sklearn.metrics import f1_score,accuracy_score,balanced_accuracy_score,classification_report,confusion_matrix
 import sklearn,joblib
 OUT.mkdir(exist_ok=True);MODELS.mkdir(exist_ok=True)
 CNN_OUT.mkdir(exist_ok=True);RF_OUT.mkdir(exist_ok=True)
 random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.set_num_threads(4)
 torch.use_deterministic_algorithms(True)
 z=np.load(CACHE);X=z['X'];y=z['y'];dogs=z['dogs'];tests=z['tests'];starts=z['starts']; K=len(CLASSES)
 masks={k:np.isin(dogs,v) for k,v in SPLITS.items()}
 assert all(set(SPLITS[a]).isdisjoint(SPLITS[b]) for a,b in [('train','test'),('train','validation'),('validation','test')])
 assert all(set(y[m])==set(range(K)) for m in masks.values())
 meta=pd.DataFrame({'dog_id':dogs,'session':tests,'start_sec':starts,'end_exclusive_sec':starts+2,'label':[CLASSES[i] for i in y]})
 meta['split']=np.select([masks['train'],masks['validation']],['train','validation'],default='test')
 meta.insert(0,'window_id',[f'dog{d}_session{s}_start{t:.12g}' for d,s,t in zip(dogs,tests,starts)])
 saved=pd.read_csv(SPLIT_DIR/'windows.csv')
 pd.testing.assert_frame_equal(meta,saved,check_dtype=False)
 # Unique, non-overlapping source windows within every session.
 assert not meta.duplicated(['dog_id','session','start_sec']).any()
 for _,g in meta.groupby(['dog_id','session']): assert (np.diff(np.sort(g.start_sec))>=1.9999).all()
 auditdata=json.loads(AUDIT_CACHE.read_text())
 protocol={'seed':SEED,'channels':CHANNELS,'classes':CLASSES,'sampling_hz':100,'window_samples':200,'stride_samples':200,'splits':SPLITS,'label_rule':'Exactly one of seven target behaviors in any annotation column at every sample, identical through entire window; auxiliary labels allowed; Task and PointEvent not inputs.','version':'Local files; release version unverified. Archive SHA256 is authoritative fingerprint.','units':'Raw stored units, no physical-unit conversion; source text does not unambiguously specify units. CSV header order retained (description swaps ANeck x/y).','cnn':'Conv1d(6,32,7),BN,ReLU,MaxPool2; Conv1d(32,64,5),BN,ReLU,MaxPool2; Conv1d(64,96,3),BN,ReLU,GlobalAvgPool,Dropout(.2),Linear(96,7)','cnn_training':{'max_epochs':35,'early_stop_patience':7,'batch_size':128,'lr':.001,'weight_decay':.0001,'class_weights':'inverse square root of training frequency','selection':'validation Macro-F1 only'},'rf':{'n_estimators':200,'min_samples_leaf':2,'max_features':'sqrt','class_weight':'balanced_subsample'},'audit':auditdata,'versions':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'sklearn':sklearn.__version__,'torch':torch.__version__},'evaluation':'One fixed dog holdout, one training seed; clean-window seven-class task, not whole continuous stream or clinical validation.'}
 protocol.update(split_file='dataset/splits/dog_splits.json',windows_file='dataset/splits/windows.csv',predictions_files={'cnn':'results/cnn/predictions.csv','random_forest':'results/random_forest/predictions.csv'},metrics_files={'cnn':'results/cnn/metrics.json','random_forest':'results/random_forest/metrics.json'})
 (OUT/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2))
 print('RF training',flush=True); t=time.perf_counter();F=features(X); prep=time.perf_counter()-t
 rf=RandomForestClassifier(n_estimators=200,min_samples_leaf=2,max_features='sqrt',class_weight='balanced_subsample',random_state=SEED,n_jobs=4)
 t=time.perf_counter();rf.fit(F[masks['train']],y[masks['train']]);rf_seconds=time.perf_counter()-t
 joblib.dump({'model':rf,'channels':CHANNELS,'classes':CLASSES,'feature_order':['mean','std','min','max','rms','q25','median','q75','mean_absolute_difference'],'window_samples':200,'seed':SEED},MODELS/'random_forest.joblib',compress=3)
 print('RF trained',rf_seconds,flush=True)
 mean=X[masks['train']].mean((0,1));std=X[masks['train']].std((0,1)).clip(1e-6)
 T=torch.from_numpy(((X-mean)/std).transpose(0,2,1).copy());Y=torch.from_numpy(y).long()
 class CNN(nn.Module):
  def __init__(self):
   super().__init__();self.net=nn.Sequential(nn.Conv1d(6,32,7,padding=3),nn.BatchNorm1d(32),nn.ReLU(),nn.MaxPool1d(2),nn.Conv1d(32,64,5,padding=2),nn.BatchNorm1d(64),nn.ReLU(),nn.MaxPool1d(2),nn.Conv1d(64,96,3,padding=1),nn.BatchNorm1d(96),nn.ReLU(),nn.AdaptiveAvgPool1d(1),nn.Flatten(),nn.Dropout(.2),nn.Linear(96,K))
  def forward(self,x):return self.net(x)
 model=CNN();opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
 freq=np.bincount(y[masks['train']],minlength=K);weights=1/np.sqrt(freq);weights/=weights.mean()
 lossfn=nn.CrossEntropyLoss(weight=torch.tensor(weights,dtype=torch.float32))
 loader=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(T[masks['train']],Y[masks['train']]),batch_size=128,shuffle=True,generator=torch.Generator().manual_seed(SEED))
 def predict(mask):
  model.eval()
  with torch.inference_mode():return torch.cat([model(b).softmax(1) for b in T[mask].split(256)]).numpy()
 history=[];best=-1;stale=0;t=time.perf_counter()
 for epoch in range(1,36):
  model.train();tot=0
  for bx,by in loader:
   opt.zero_grad();loss=lossfn(model(bx),by);loss.backward();opt.step();tot+=loss.item()*len(by)
  vp=predict(masks['validation']);score=f1_score(y[masks['validation']],vp.argmax(1),labels=list(range(K)),average='macro',zero_division=0)
  history.append({'epoch':epoch,'train_loss':tot/int(masks['train'].sum()),'validation_macro_f1':score})
  print(f'Epoch {epoch}: val Macro-F1={score:.4f}',flush=True)
  if score>best+1e-6:
   best=score;stale=0;bestepoch=epoch
   torch.save({'state_dict':model.state_dict(),'mean':mean.tolist(),'std':std.tolist(),'channels':CHANNELS,'classes':CLASSES,'window_samples':200,'seed':SEED,'epoch':epoch},MODELS/'cnn.pt')
  else:stale+=1
  if stale>=7:break
 cnn_seconds=time.perf_counter()-t
 ck=torch.load(MODELS/'cnn.pt',weights_only=True);model.load_state_dict(ck['state_dict']);model.eval()
 pd.DataFrame(history).to_csv(CNN_OUT/'training_history.csv',index=False)
 metrics={};testmask=masks['test'];yt=y[testmask];dt=dogs[testmask]; unique=np.unique(dt)
 for name,pr,seconds in [('random_forest',rf.predict_proba(F[testmask]),rf_seconds),('cnn',predict(testmask),cnn_seconds)]:
  yp=pr.argmax(1); meta.loc[testmask,name+'_prediction']=[CLASSES[i] for i in yp]
  meta.loc[testmask,name+'_confidence']=pr.max(1)
  cms=np.stack([confusion_matrix(yt[dt==d],yp[dt==d],labels=list(range(K))) for d in unique])
  def macro(cm):
   den=cm.sum(0)+cm.sum(1);return float(np.divide(2*np.diag(cm),den,out=np.zeros(K),where=den>0).mean())
  rng=np.random.default_rng(SEED);boots=[macro(cms[rng.integers(0,len(unique),len(unique))].sum(0)) for _ in range(2000)]
  # Single-window latency includes feature extraction / normalization; excludes sensor acquisition.
  lat=[]
  for i in range(110):
   sample=X[testmask][i%len(yt):i%len(yt)+1];beg=time.perf_counter()
   if name=='random_forest':rf.predict_proba(features(sample))
   else:
    with torch.inference_mode():model(torch.from_numpy(((sample-mean)/std).transpose(0,2,1).copy())).softmax(1)
   if i>=10:lat.append((time.perf_counter()-beg)*1000)
  per_dog={str(d):{'windows':int((dt==d).sum()),'macro_f1_fixed_7_classes':macro(cms[j]),'accuracy':float(accuracy_score(yt[dt==d],yp[dt==d])),'balanced_accuracy_present_classes':float(balanced_accuracy_score(yt[dt==d],yp[dt==d])),'class_support':cms[j].sum(1).tolist()} for j,d in enumerate(unique)}
  metrics[name]={'macro_f1':float(f1_score(yt,yp,labels=list(range(K)),average='macro',zero_division=0)),'accuracy':float(accuracy_score(yt,yp)),'balanced_accuracy':float(balanced_accuracy_score(yt,yp)),'dog_cluster_bootstrap_95ci':np.quantile(boots,[.025,.975]).tolist(),'classification_report':classification_report(yt,yp,labels=list(range(K)),target_names=CLASSES,output_dict=True,zero_division=0),'confusion_matrix':cms.sum(0).tolist(),'per_dog':per_dog,'training_seconds':seconds,'latency_ms_p50':float(np.median(lat)),'latency_ms_p95':float(np.quantile(lat,.95))}
 metrics['cnn']['best_epoch']=bestepoch;metrics['cnn']['parameters']=sum(p.numel() for p in model.parameters());metrics['rf_feature_extraction_all_windows_seconds']=prep
 for name,destination in [('cnn',CNN_OUT),('random_forest',RF_OUT)]:
  prediction_cols=['window_id']+[c for c in meta if c.startswith(name+'_')]
  meta[prediction_cols].to_csv(destination/'predictions.csv',index=False)
  model_metrics={name:metrics[name]}
  if name=='random_forest':model_metrics['rf_feature_extraction_all_windows_seconds']=prep
  (destination/'metrics.json').write_text(json.dumps(model_metrics,ensure_ascii=False,indent=2))
 # Re-load artifacts and verify identical predictions.
 rf2=joblib.load(MODELS/'random_forest.joblib')['model'];assert np.array_equal(rf.predict(F[testmask][:32]),rf2.predict(F[testmask][:32]))
 clone=CNN();clone.load_state_dict(torch.load(MODELS/'cnn.pt',weights_only=True)['state_dict']);clone.eval()
 with torch.inference_mode():assert torch.equal(model(T[testmask][:8]),clone(T[testmask][:8]))
 print('DONE',json.dumps({k:{a:v[a] for a in ['macro_f1','accuracy','training_seconds']} for k,v in metrics.items() if isinstance(v,dict)}),flush=True)

if __name__=='__main__':
 import argparse
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('command',choices=['audit','train'])
 args=parser.parse_args()
 audit() if args.command=='audit' else train()
