"""CUDA engineering pilot: real source records, full inputs, no-XP MAE and six-D NSF.

A small freshly initialized network checks integration. It does not replace or
rank pretrained production models; staged new alpha/age products are not admitted.
"""
import argparse,json,time,hashlib
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
import torch
from torch import nn
from masked_stellar_autoencoder.training.full_gaia_inputs import PHOTOMETRY,XP,AUXILIARY,full_gaia_policy,auxiliary_context,visible_observations
from masked_stellar_autoencoder.training.consensus_survey_map import ConsensusSurveyMap
from masked_stellar_autoencoder.training.six_survey_map import SixSurveyMap
from masked_stellar_autoencoder.training.remediation_joint6 import JointSixHead

ROOT=Path('/arc/projects/k-pop/msa_runs/stellar-improvement-20261015')

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--updates',type=int,default=600);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
 m=dict(status='waiting_for_source_parts',training_scope='Fresh small-network engineering pilot, not continuation of SSL and not final architecture selection',arms={},reconstruction_weight=.03,new_age_alpha_products_admitted=False)
 def save():
  t=a.output/'manifest.tmp';t.write_text(json.dumps(m,indent=2,default=str)+'\n');t.replace(a.output/'manifest.json')
 save()
 try:
  if not torch.cuda.is_available():raise RuntimeError('CANFAR CUDA required')
  torch.set_num_threads(4);device='cuda'
  photo=ROOT/'full_gaia_photometry_stage_20261108_v1/raw-observations';gaia=ROOT/'full_gaia_gaia_stage_20261108_v2/ssl-reservoir'
  deadline=time.monotonic()+6*3600
  while not (list(photo.glob('*.parquet')) and list(gaia.glob('*.parquet'))):
   for folder in [photo.parent,gaia.parent]:
    path=folder/'manifest.json'
    if path.exists():
     d=json.loads(path.read_text())
     if d.get('status')=='failed':raise RuntimeError('Source dependency failed: '+str(folder)+' '+d.get('error',''))
   if time.monotonic()>deadline:raise TimeoutError('Required source parts not ready after six hours')
   time.sleep(30)
  parts=sorted(photo.glob('*.parquet'))[:8]
  raw=pd.concat([pd.read_parquet(x) for x in parts],ignore_index=True).drop_duplicates('source_id').head(50000)
  if raw.source_id.dtype.kind not in 'iu':raise ValueError('Noninteger Gaia IDs')
  names=PHOTOMETRY+XP+AUXILIARY;policy=full_gaia_policy(names)
  x=raw.reindex(columns=names).to_numpy(dtype='f4')
  error_names=json.loads(Path('configs/full-gaia-observation-columns-20261108.json').read_text());errlookup=dict(zip(error_names['feature_names'],error_names['error_names']))
  errors=np.column_stack([pd.to_numeric(raw[errlookup[n]],errors='coerce').to_numpy() if errlookup.get(n) in raw else np.full(len(raw),np.nan) for n in names])
  with h5py.File(ROOT/'expanded_six_source_20261107_v5/prepared.h5') as f:
   ids=f['source_id'][:];roles=f['role'][:];pos=np.searchsorted(ids,raw.source_id.to_numpy());hit=pos<len(ids);hit[hit]&=ids[pos[hit]]==raw.source_id.to_numpy()[hit]
   oldrows=pos[hit];x=x[hit];errors=errors[hit];sid=raw.source_id.to_numpy()[hit];starrole=roles[oldrows]
   order=np.argsort(oldrows);oldrows=oldrows[order];x=x[order];errors=errors[order];sid=sid[order];starrole=starrole[order]
   records=[];sources=[]
   for name,g in f['native_observations'].items():
    rr=g['star_row'][:];p0=np.searchsorted(oldrows,rr);h=p0<len(oldrows);h[h]&=oldrows[p0[h]]==rr[h];idx=np.flatnonzero(h)
    if not len(idx):continue
    v=g['value'][idx];obs=g['observed'][idx].astype(bool);e=g['error'][idx]
    records.append((p0[h],v,obs,e));sources.append(name)
  if len(x)<256 or not records:raise RuntimeError('Too few matched staged observations for engineering pilot')
  row=np.concatenate([r[0] for r in records]);y=np.concatenate([r[1] for r in records]);mask=np.concatenate([r[2] for r in records]);sigma=np.concatenate([r[3] for r in records]);source=np.concatenate([np.full(len(r[0]),i) for i,r in enumerate(records)])
  for j in [4,5]:
   mask[:,j]&=y[:,j]>0
   sigma[:,j]=np.where(mask[:,j],sigma[:,j]/np.maximum(y[:,j],1e-20),np.nan)
   y[:,j]=np.where(mask[:,j],np.log(np.maximum(y[:,j],1e-20)),np.nan)
  fitstar=starrole==b'fit';fit=fitstar[row]&mask.any(1);tune=(starrole[row]==b'tune')&mask.any(1)
  if fit.sum()<128 or tune.sum()<16:raise RuntimeError('Insufficient FIT/TUNE records')
  center=np.array([np.median(y[fit&mask[:,j],j]) if (fit&mask[:,j]).any() else 0 for j in range(6)])
  scale=np.array([np.std(y[fit&mask[:,j],j]) if (fit&mask[:,j]).sum()>2 else 1 for j in range(6)]);scale=np.maximum(scale,1e-3)
  y=np.where(mask,(y-center)/scale,0).astype('f4');sigma=np.where(mask&np.isfinite(sigma)&(sigma>0),sigma/scale,.1).astype('f4');sigma=np.minimum(sigma,10)
  visible=visible_observations(x);x[~visible]=np.nan
  xc=np.array([np.nanmedian(x[fitstar,j]) if np.isfinite(x[fitstar,j]).any() else 0 for j in range(len(names))]);xs=np.array([np.nanstd(x[fitstar,j]) if np.isfinite(x[fitstar,j]).sum()>2 else 1 for j in range(len(names))]);xs=np.maximum(xs,1e-3)
  values=np.nan_to_num((x-xc)/xs,nan=0).astype('f4');aux,auxmask,auxnames=auxiliary_context(x,errors,names);ac=np.array([np.median(aux[fitstar&auxmask[:,j],j]) if (fitstar&auxmask[:,j]).any() else 0 for j in range(aux.shape[1])]);ast=np.array([np.std(aux[fitstar&auxmask[:,j],j]) if (fitstar&auxmask[:,j]).sum()>2 else 1 for j in range(aux.shape[1])]);aux=np.where(auxmask,(aux-ac)/np.maximum(ast,1e-3),0).astype('f4')
  # Real no-XP rows use measured Gaia photometry only and the same FIT normalization.
  reservoir=pd.concat([pd.read_parquet(p) for p in sorted(gaia.glob('*.parquet'))[:8]],ignore_index=True).drop_duplicates('source_id')
  if 'has_xp_continuous' not in reservoir:raise RuntimeError('Cannot certify genuine no-XP reservoir')
  reservoir=reservoir.loc[~reservoir.has_xp_continuous.astype(bool)].copy()
  z=reservoir.source_id.to_numpy(dtype='u8').copy();z^=z>>np.uint64(30);z*=np.uint64(0xBF58476D1CE4E5B9);z^=z>>np.uint64(27);z*=np.uint64(0x94D049BB133111EB);z^=z>>np.uint64(31)
  blocked=np.asarray(json.loads(Path('configs/expanded-six-source-20261107.json').read_text())['benchmark_fit_exclusion_ids'],dtype='i8')
  ok=(z%100<75)&~np.isin(reservoir.source_id.to_numpy(),blocked)
  # Inherit prior held-out roles wherever IDs already exist; do not rehash them into FIT.
  rp=np.searchsorted(ids,reservoir.source_id.to_numpy());rh=rp<len(ids);rh[rh]&=ids[rp[rh]]==reservoir.source_id.to_numpy()[rh];ok[rh]&=roles[rp[rh]]==b'fit'
  reservoir=reservoir.loc[ok]
  rx=np.full((len(reservoir),len(names)),np.nan,dtype='f4')
  for n,c in [('G','phot_g_mean_mag'),('BP','phot_bp_mean_mag'),('RP','phot_rp_mean_mag')]:
   if c in reservoir:rx[:,names.index(n)]=reservoir[c]
  rv=visible_observations(rx);rx=np.nan_to_num((rx-xc)/xs,nan=0).astype('f4')
  if len(rx)<16:raise RuntimeError('Insufficient genuine no-XP FIT reconstruction rows')
  support=np.array([mask[fit&(source==i)].any(0) for i in range(len(sources))]);active=support.any(0)
  m.update(status='training',cuda=torch.cuda.get_device_name(0),stars=len(x),fit_records=int(fit.sum()),tune_records=int(tune.sum()),sources=sources,support=support.tolist(),active_targets=active.tolist(),real_no_xp_ssl_rows=len(rx),raw_parts=[str(p) for p in parts],scope_limit='New ages/alpha still pending admission. Small fresh network, bounded current native records. Additional catalogue joins and cluster exclusions beyond inherited roles remain production requirements.');save()
  tensor=lambda a,dtype=torch.float32:torch.as_tensor(a,dtype=dtype,device=device)
  v=tensor(values);vm=tensor(visible);au=tensor(np.concatenate([aux,auxmask],1));yy=tensor(y);mm=tensor(mask);ee=tensor(sigma);sr=tensor(source,torch.long);rr=tensor(row,torch.long);recon=tensor(rx);reconmask=tensor(rv)
  encidx=np.flatnonzero(policy.encoder);oldidx=np.array([names.index(n) for n in ('W1','W2','G','BP','RP')+XP]);xpi=np.array([names.index(n) for n in XP])
  for arm in ['restricted_inputs','restored_inputs','restored_consensus']:
   torch.manual_seed(20261108);torch.cuda.manual_seed_all(20261108);rng=np.random.default_rng(20261108)
   encoder=nn.Sequential(nn.Linear(2*len(names),256),nn.ELU(),nn.Linear(256,128),nn.ELU()).to(device);decoder=nn.Linear(128,len(encidx)).to(device)
   context=nn.Sequential(nn.Linear(128+au.shape[1],256),nn.ELU()).to(device);head=JointSixHead('nsf',latent_dim=256,width=128,particles=32,proposal_weight=0).to(device);head.gaussian.requires_grad_(False)
   # Unsupported targets are irrelevant to the loss; keep a declared dummy support solely for gauge construction.
   if not active.all():raise RuntimeError('All six supported targets required before common-gauge pilot')
   maps=ConsensusSurveyMap(support,support.astype(float),kind='spline').to(device) if arm=='restored_consensus' else SixSurveyMap(support,[int(np.flatnonzero(support[:,j])[0]) for j in range(6)],kind='spline').to(device)
   params=list(encoder.parameters())+list(decoder.parameters())+list(context.parameters())+list(head.parameters())+list(maps.parameters());opt=torch.optim.Adam(params,lr=3e-4)
   allowed=torch.zeros(len(names),device=device);allowed[oldidx if arm=='restricted_inputs' else encidx]=1
   def latent(rows,drop=False):
    values0=v[rows];mask0=vm[rows]*allowed
    if drop:
     mask0=mask0*(torch.rand_like(mask0)>.2)
     if arm!='restricted_inputs':mask0[:,xpi]*=(torch.rand((len(rows),1),device=device)>.3)
    z=encoder(torch.cat([values0*mask0,mask0],1));a0=au[rows] if arm!='restricted_inputs' else torch.zeros_like(au[rows]);return z,context(torch.cat([z,a0],1))
   for step in range(1,a.updates+1):
    pick=rng.choice(np.flatnonzero(fit),size=min(64,int(fit.sum())),replace=False);ix=tensor(pick,torch.long);r=rr[ix];opt.zero_grad(set_to_none=True);z,c=latent(r,True)
    draw=maps(head.sample(c,16),sr[ix]);noisy=draw+torch.randn_like(draw)*ee[ix][None];mask1=mm[ix][None];d1=torch.sqrt(((noisy-yy[ix][None])*mask1).square().sum(-1)+1e-8).mean();d2=torch.sqrt(((noisy[:8]-noisy[8:])*mask1).square().sum(-1)+1e-8).mean();energy=d1-.5*d2
    recmask=vm[r][:,encidx]*allowed[encidx];mae=((decoder(z)-v[r][:,encidx]).abs()*recmask).sum()/recmask.sum().clamp_min(1)
    if arm!='restricted_inputs':
     rp=tensor(rng.choice(len(recon),size=min(64,len(recon)),replace=False),torch.long);rvm=reconmask[rp];masked=rvm*(torch.rand_like(rvm)>.2);rz=encoder(torch.cat([recon[rp]*masked,masked],1));mae=mae+((decoder(rz)-recon[rp][:,encidx]).abs()*rvm[:,encidx]).sum()/rvm[:,encidx].sum().clamp_min(1)
    loss=energy+.03*mae+.001*maps.regularization();loss.backward()
    if not torch.isfinite(loss) or any(p.grad is not None and not torch.isfinite(p.grad).all() for p in params):raise FloatingPointError('Nonfinite integration loss/gradient')
    torch.nn.utils.clip_grad_norm_(params,10);opt.step()
    if step%100==0:m['arms'][arm]=dict(completed_updates=step,energy=float(energy.detach()),mae=float(mae.detach()));save()
   with torch.no_grad():
    pick=np.flatnonzero(tune)[:512];ix=tensor(pick,torch.long);_,c=latent(rr[ix]);draw=maps(head.sample(c,128),sr[ix]);median=draw.median(0).values;err=(median-yy[ix]).abs();maes=[float(err[:,j][mm[ix,j].bool()].mean()) if mm[ix,j].any() else None for j in range(6)]
    np.savez_compressed(a.output/(arm+'-tune.npz'),source_id=sid[row[pick]],source=source[pick],target=y[pick],observed=mask[pick],draws=draw.cpu().numpy(),center=center,scale=scale)
   torch.save(dict(encoder=encoder.state_dict(),decoder=decoder.state_dict(),context=context.state_dict(),head=head.state_dict(),maps=maps.state_dict(),feature_names=names,aux_names=auxnames,feature_center=xc,feature_scale=xs,target_center=center,target_scale=scale),a.output/(arm+'.pth'))
   m['arms'][arm].update(tune_native_mae_standardized=maes,completed_updates=a.updates);save()
  m['status']='engineering_pilot_complete';save()
 except Exception as e:m.update(status='failed',error=repr(e));save();raise

if __name__=='__main__':main()
