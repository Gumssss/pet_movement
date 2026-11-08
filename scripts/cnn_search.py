"""20 predeclared CNN training configurations; select by validation Macro-F1 only.
Run with the same environment as baseline.py. Resumes completed trials.
"""
import json,time,random,copy,hashlib
from pathlib import Path
import numpy as np,pandas as pd,torch
from torch import nn
from sklearn.metrics import f1_score,accuracy_score,balanced_accuracy_score,classification_report,confusion_matrix
ROOT=Path(__file__).resolve().parents[1];SHARED=ROOT/'results';R=SHARED/'cnn';RF=SHARED/'random_forest';M=ROOT/'models'
SPLIT_DIR=ROOT/'dataset/splits'
SPLIT_CONFIG=json.loads((SPLIT_DIR/'dog_splits.json').read_text())
SEED=SPLIT_CONFIG['seed'];DEVICE='mps' if torch.backends.mps.is_available() else 'cpu'
torch.set_num_threads(2)
BASE=dict(lr=.001,weight_decay=.0001,dropout=.2,batch_size=128,class_weight_power=.5,scheduler='none')
CHANGES=[{}, {'lr':.0003},{'lr':.0006},{'lr':.002},{'lr':.003},{'weight_decay':.00001},{'weight_decay':.001},{'weight_decay':.01},{'dropout':.1},{'dropout':.35},{'dropout':.5},{'batch_size':64},{'batch_size':256},{'class_weight_power':0.},{'class_weight_power':1.},{'scheduler':'cosine'},{'lr':.0006,'dropout':.35,'weight_decay':.001,'scheduler':'cosine'},{'lr':.002,'dropout':.1,'batch_size':256,'scheduler':'cosine'},{'lr':.001,'dropout':.35,'weight_decay':.001,'class_weight_power':1.,'scheduler':'cosine'},{'lr':.0006,'batch_size':256,'class_weight_power':0.,'scheduler':'cosine'}]
CONFIGS=[dict(BASE,**c) for c in CHANGES]
def sync():
 if DEVICE=='mps':torch.mps.synchronize()
def atomic_json(path,data):
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2));tmp.replace(path)
def scores(y,p):
 return {'macro_f1':float(f1_score(y,p,labels=list(range(7)),average='macro',zero_division=0)),'balanced_accuracy':float(balanced_accuracy_score(y,p)),'accuracy':float(accuracy_score(y,p))}
class CNN(nn.Module):
 def __init__(self,dropout):
  super().__init__();self.net=nn.Sequential(nn.Conv1d(6,32,7,padding=3),nn.BatchNorm1d(32),nn.ReLU(),nn.MaxPool1d(2),nn.Conv1d(32,64,5,padding=2),nn.BatchNorm1d(64),nn.ReLU(),nn.MaxPool1d(2),nn.Conv1d(64,96,3,padding=1),nn.BatchNorm1d(96),nn.ReLU(),nn.AdaptiveAvgPool1d(1),nn.Flatten(),nn.Dropout(dropout),nn.Linear(96,7))
 def forward(self,x):return self.net(x)
def predict(model,x):
 model.eval()
 with torch.inference_mode():return torch.cat([model(b).softmax(1) for b in x.split(512)]).cpu().numpy()
def main():
 p=json.loads((SHARED/'protocol.json').read_text())
 metrics={**json.loads((RF/'metrics.json').read_text()),**json.loads((R/'metrics.json').read_text())}
 assert p['splits']==SPLIT_CONFIG['splits'] and p['seed']==SEED
 windows=pd.read_csv(SPLIT_DIR/'windows.csv')
 rf_predictions=pd.read_csv(RF/'predictions.csv');cnn_predictions=pd.read_csv(R/'predictions.csv')
 assert windows.window_id.tolist()==rf_predictions.window_id.tolist()==cnn_predictions.window_id.tolist()
 predictions=rf_predictions.merge(cnn_predictions,on='window_id',validate='one_to_one',sort=False)
 assert windows.window_id.tolist()==predictions.window_id.tolist()
 w=windows.merge(predictions,on='window_id',validate='one_to_one',sort=False);test=w.split=='test'
 # Random-forest results are read only here; search updates CNN outputs only.
 text=(SHARED/'README.md').read_text();marker='\n## Balanced accuracy 与20次CNN超参数搜索\n'
 if marker not in text:
  text+=marker+f"\n原始随机森林 balanced accuracy：{metrics['random_forest']['balanced_accuracy']:.4f}；原始CNN：{metrics['cnn']['balanced_accuracy']:.4f}。定义为真实标签中各类别召回率的平均值；总体测试集包含全部七类。每犬仅平均该犬真实出现的类别。\n\n20组预声明超参数均使用相同七类、窗口、犬只划分和种子。仅按验证Macro-F1选择checkpoint及最终模型，balanced accuracy同时记录；每次试验结果在cnn/cnn_search.csv，每轮记录在cnn/cnn_search_history.csv。最佳权重在models/cnn_best.pt，原始cnn.pt保留。\n"
  (SHARED/'README.md').write_text(text)
 search_protocol={'configurations':CONFIGS,'max_epochs':35,'patience':7,'min_delta':1e-6,'selection':'validation Macro-F1; ties choose earlier trial','seed':SEED,'device':DEVICE,'threads':2,'torch_version':torch.__version__,'note':'Fresh training per trial, not continuing from baseline weights. Same architecture, changed training hyperparameters/dropout. Fixed seed, MPS bitwise reproducibility not guaranteed. Test not used in search; prior baseline test was already reported.'}
 if 'cnn_search' in p:assert p['cnn_search']['configurations']==CONFIGS
 p['cnn_search']=search_protocol;atomic_json(SHARED/'protocol.json',p)
 z=np.load(ROOT/'dataset/cache/pet_windows.npz');X=z['X'];y=z['y'];dogs=z['dogs']
 assert len(X)==len(w) and np.array_equal(dogs,w.dog_id) and np.array_equal(z['tests'],w.session) and np.allclose(z['starts'],w.start_sec)
 assert np.array_equal(np.array(p['classes'])[y],w.label)
 masks={k:np.isin(dogs,ids) for k,ids in SPLIT_CONFIG['splits'].items()}
 for k,v in masks.items():assert np.array_equal(v,(w.split==k).to_numpy())
 mean=X[masks['train']].mean((0,1));std=X[masks['train']].std((0,1)).clip(1e-6)
 # Only train/validation copied to device during search.
 tensors={k:torch.from_numpy(((X[v]-mean)/std).transpose(0,2,1).copy()).to(DEVICE) for k,v in masks.items() if k!='test'}
 ty=torch.tensor(y[masks['train']],device=DEVICE,dtype=torch.long);vy=y[masks['validation']]
 freq=np.bincount(y[masks['train']],minlength=7)
 csv=R/'cnn_search.csv';histfile=R/'cnn_search_history.csv'
 records=pd.read_csv(csv).to_dict('records') if csv.exists() else [];history=pd.read_csv(histfile).to_dict('records') if histfile.exists() else []
 done={int(r['trial']) for r in records};best_global=max([r['validation_macro_f1'] for r in records],default=-1)
 print('DEVICE',DEVICE,'baseline BA',metrics['random_forest']['balanced_accuracy'],metrics['cnn']['balanced_accuracy'],flush=True)
 for trial,c in enumerate(CONFIGS,1):
  if trial in done:continue
  history=[h for h in history if int(h['trial'])!=trial]
  random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED)
  model=CNN(c['dropout']).to(DEVICE);opt=torch.optim.AdamW(model.parameters(),lr=c['lr'],weight_decay=c['weight_decay'])
  sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=35,eta_min=c['lr']*.05) if c['scheduler']=='cosine' else None
  weight=freq.astype(float)**(-c['class_weight_power']);weight/=weight.mean();lossfn=nn.CrossEntropyLoss(weight=torch.tensor(weight,dtype=torch.float32,device=DEVICE))
  gen=torch.Generator().manual_seed(SEED);best=-1;stale=0;start=time.perf_counter();trialhistory=[]
  print('TRIAL',trial,c,flush=True)
  for epoch in range(1,36):
   model.train();order=torch.randperm(len(ty),generator=gen).to(DEVICE);total=torch.zeros((),device=DEVICE)
   for ix in order.split(c['batch_size']):
    opt.zero_grad(set_to_none=True);loss=lossfn(model(tensors['train'][ix]),ty[ix]);loss.backward();opt.step();total+=loss.detach()*len(ix)
   pr=predict(model,tensors['validation']);sc=scores(vy,pr.argmax(1))
   row={'trial':trial,'epoch':epoch,'train_loss':float(total.cpu())/len(ty),'lr':opt.param_groups[0]['lr'],**{'validation_'+k:v for k,v in sc.items()}}
   trialhistory.append(row)
   if sc['macro_f1']>best+1e-6:
    best=sc['macro_f1'];bestsc=sc;bestepoch=epoch;stale=0;state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
   else:stale+=1
   if sched:sched.step()
   if epoch%5==0:print(' trial',trial,'epoch',epoch,'valF1',round(sc['macro_f1'],4),'best',round(best,4),flush=True)
   if stale>=7:break
  sync();elapsed=time.perf_counter()-start
  row={'trial':trial,**c,'best_epoch':bestepoch,'epochs_run':epoch,**{'validation_'+k:v for k,v in bestsc.items()},'training_seconds':elapsed,'seed':SEED,'device':DEVICE}
  if best>best_global+1e-6:
   best_global=best
   ck={'state_dict':state,'mean':mean.tolist(),'std':std.tolist(),'channels':p['channels'],'classes':p['classes'],'window_samples':200,'seed':SEED,'epoch':bestepoch,'trial':trial,'config':c,'validation_metrics':bestsc,'device_trained':DEVICE}
   tmp=M/'cnn_best.tmp';torch.save(ck,tmp);tmp.replace(M/'cnn_best.pt')
  records.append(row);history.extend(trialhistory)
  pd.DataFrame(records).to_csv(csv,index=False);pd.DataFrame(history).to_csv(histfile,index=False)
  print('FINISHED',json.dumps(row),flush=True)
  del model,opt
  if DEVICE=='mps':torch.mps.empty_cache()
 # Select has completed before accessing test predictions.
 assert len(records)==20
 ck=torch.load(M/'cnn_best.pt',weights_only=True);best_model=CNN(ck['config']['dropout']).to(DEVICE);best_model.load_state_dict(ck['state_dict'])
 assert ck['trial']==max(records,key=lambda r:r['validation_macro_f1'])['trial']
 testx=torch.from_numpy(((X[masks['test']]-mean)/std).transpose(0,2,1).copy()).to(DEVICE)
 pr=predict(best_model,testx);yp=pr.argmax(1);yt=y[masks['test']];ds=dogs[masks['test']]
 entry={**scores(yt,yp),'trial':ck['trial'],'best_epoch':ck['epoch'],'config':ck['config'],'validation_metrics':ck['validation_metrics'],'classification_report':classification_report(yt,yp,labels=list(range(7)),target_names=p['classes'],output_dict=True,zero_division=0),'confusion_matrix':confusion_matrix(yt,yp,labels=list(range(7))).tolist(),'per_dog':{str(d):scores(yt[ds==d],yp[ds==d]) for d in np.unique(ds)},'device_trained':DEVICE,'parameters':sum(t.numel() for t in best_model.parameters()),'search_total_training_seconds':sum(r['training_seconds'] for r in records)}
 rng=np.random.default_rng(SEED);unique=np.unique(ds);boots=[]
 for _ in range(2000):
  inds=np.concatenate([np.flatnonzero(ds==d) for d in rng.choice(unique,len(unique),replace=True)]);boots.append(list(scores(yt[inds],yp[inds]).values()))
 entry['dog_bootstrap_95ci']={k:np.quantile(np.array(boots)[:,i],[.025,.975]).tolist() for i,k in enumerate(['macro_f1','balanced_accuracy','accuracy'])}
 metrics['cnn_best']=entry;atomic_json(R/'metrics.json',{k:v for k,v in metrics.items() if k in ['cnn','cnn_best']})
 w.loc[test,'cnn_best_prediction']=np.array(p['classes'])[yp];w.loc[test,'cnn_best_confidence']=pr.max(1);w[['window_id']+[c for c in w if c.startswith('cnn_')]].to_csv(R/'predictions.csv',index=False)
 # Verify saved prediction metrics and model reload consistency.
 assert abs(balanced_accuracy_score(w.loc[test,'label'],w.loc[test,'cnn_best_prediction'])-entry['balanced_accuracy'])<1e-12
 loaded=CNN(ck['config']['dropout']).to(DEVICE);loaded.load_state_dict(torch.load(M/'cnn_best.pt',weights_only=True)['state_dict'])
 assert np.allclose(predict(best_model,testx[:32]),predict(loaded,testx[:32]),atol=1e-6)
 table='\n|模型|测试Macro-F1|测试Balanced accuracy|测试Accuracy|\n|---|---:|---:|---:|\n'
 for k,n in [('random_forest','随机森林'),('cnn','原始CNN'),('cnn_best','调参CNN（验证集选择）')]:
  v=metrics[k];table+=f"|{n}|{v['macro_f1']:.4f}|{v['balanced_accuracy']:.4f}|{v['accuracy']:.4f}|\n"
 report=(SHARED/'README.md').read_text();report=report.split('\n### 20次搜索完成结果')[0]
 report+='\n### 20次搜索完成结果\n'+table+f"\n20/20组完成，最佳为第{ck['trial']}组、第{ck['epoch']}轮。参数：`{json.dumps(ck['config'])}`。按验证Macro-F1选择，并非按测试集挑最好者。每组从头训练而非连续续训，架构与数据任务保持一致；固定种子，MPS可能存在数值非确定性。设备为{DEVICE}，因此训练耗时不可直接与此前CPU训练相比。所有组有验证结果，只有最终选中组额外做测试评价。搜索总训练时间{entry['search_total_training_seconds']/60:.1f}分钟。\n\n搜索脚本位于scripts/cnn_search.py；训练记录位于results/cnn/cnn_search.csv、results/cnn/cnn_search_history.csv；最佳模型为models/cnn_best.pt。20次搜索可能过拟合9只验证犬，这不是20次独立泛化验证。测试集已在先前基线中查看过，正式研究仍需额外独立验证。\n"
 (SHARED/'README.md').write_text(report)
 print('ALL_DONE',json.dumps({k:entry[k] for k in ['trial','macro_f1','balanced_accuracy','accuracy']}),flush=True)
if __name__=='__main__':
 import argparse
 parser=argparse.ArgumentParser(description=__doc__)
 parser.parse_args()
 main()
