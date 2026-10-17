import base64
import json
import zlib
from pathlib import Path

from canfar.sessions import Session

record = Path("experiments/xp/final-verification-session.json")
previous = json.loads(record.read_text())
s = Session()
for ident in previous["session_ids"]:
    info = s.info(ident)
    assert info and all(i["status"] == "Failed" for i in info), (
        "Existing verifier not confirmed failed"
    )
program = Path("experiments/xp/monitor/verify_benchmark.py").read_text()
manifest = '[workspace]\nname="xp-verify"\nchannels=["conda-forge"]\nplatforms=["linux-64"]\n[dependencies]\npython="3.13.*"\nnumpy="*"\nh5py="*"\n'
code = f"""import os,subprocess,time,sys
from pathlib import Path
os.environ.update(PIXI_HOME='/scratch/.pixi',PIXI_CACHE_DIR='/scratch/.cache/pixi');os.environ.pop('PYTHONPATH',None)
p=Path('/scratch/verify.py');p.write_text({program!r});m=Path('/scratch/env/pixi.toml');m.parent.mkdir(exist_ok=True);m.write_text({manifest!r})
probe=subprocess.run([sys.executable,'-c','import numpy,h5py;print(numpy.__file__,h5py.__file__)'])
if probe.returncode==0:
 subprocess.run([sys.executable,'-u',str(p)],check=True)
 sys.exit(0)
for attempt in range(3):
 r=subprocess.run(['pixi','install','--manifest-path',str(m)])
 if r.returncode==0: break
 if attempt==2: break
 time.sleep(5)
if r.returncode:
 subprocess.run([sys.executable,'-m','venv','/scratch/verification-venv'],check=True)
 interpreter='/scratch/verification-venv/bin/python'
 subprocess.run([interpreter,'-c',"import socket,runpy,sys; resolver=socket.getaddrinfo; hosts={{'pypi.org': '151.101.64.223', 'files.pythonhosted.org': '151.101.192.223'}}; socket.getaddrinfo=lambda h,p,*a,**k:resolver(hosts.get(h,h),p,*a,**k); sys.argv=['pip','install','numpy','h5py']; runpy.run_module('pip',run_name='__main__')"],check=True)
 subprocess.run([interpreter,'-u',str(p)],check=True)
else:
 subprocess.run(['pixi','run','--manifest-path',str(m),'python','-u',str(p)],check=True)
"""
a = (
    "exec(__import__('zlib').decompress(__import__('base64').b64decode('"
    + base64.b64encode(zlib.compress(code.encode())).decode()
    + "')))"
)
ids = s.create(
    name="msa-xp-final-verification-r2",
    image="images.canfar.net/astroai/notebook:latest",
    kind="headless",
    cmd="python",
    args="-u -c " + a,
    cores=4,
    ram=16,
    env={"PYTHONNOUSERSITE": "1", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"},
)
previous["failed_session_ids"] = (
    previous.get("failed_session_ids", []) + previous["session_ids"]
)
previous["session_ids"] = ids
record.write_text(json.dumps(previous, indent=2))
print(ids)
