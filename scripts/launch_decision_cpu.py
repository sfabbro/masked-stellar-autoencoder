"""Minimal Pixi CPU workers for unattended decision-wave QA and source preparation."""

import base64
import datetime as dt
import hashlib
import json
import zlib
from pathlib import Path

from launch_decision_wave import ROOT, addresses
from launch_stellar_improvement import create_form_session, submission_error_evidence


def submit_cpu(
    run_id, files, command, *, cpu=4, ram=32, extra_dependencies=None,
    image="images.canfar.net/astroai/terminal:latest",
):
    from canfar.sessions import Session

    directory = Path("experiments/xp/decision-wave-20261106")
    directory.mkdir(parents=True, exist_ok=True)
    receipt = directory / (run_id + ".json")
    if (cpu is None) != (ram is None):
        raise ValueError("CPU and RAM must be supplied together, or both omitted")
    if receipt.exists():
        raise RuntimeError("Reconcile existing receipt before submission")
    manifest = '[workspace]\nname="msa-decision-qa"\nchannels=["conda-forge"]\nplatforms=["linux-64"]\n[dependencies]\npython="3.13.*"\nnumpy=">=2.2,<3"\nh5py=">=3.10"\nastropy=">=6"\npandas=">=2.1"\nscipy=">=1.14"\nmatplotlib-base=">=3.8"\npyarrow=">=18"\n'
    if extra_dependencies:
        manifest += extra_dependencies
    files = dict(files, **{"pixi.toml": manifest})
    output = ROOT + run_id
    bundle = ROOT + "decision_wave_source_bundles_20261106_v1/" + run_id
    entry = (
        Path("scripts/canfar_dns_proxy.py").read_text()
        + "\nimport pathlib,subprocess,json,hashlib,shutil\n"
    )
    entry += "start_dns_proxy(" + repr(addresses()) + ")\n"
    entry += "root=pathlib.Path('/scratch/msa-decision-cpu');root.mkdir(parents=True,exist_ok=False)\n"
    entry += (
        "bundle=pathlib.Path("
        + repr(bundle)
        + ");bundle.mkdir(parents=True,exist_ok=False)\nfiles="
        + repr(files)
        + "\n"
    )
    entry += "for name,content in files.items():\n for target in [root/name,bundle/name]:\n  target.parent.mkdir(parents=True,exist_ok=True);target.write_text(content)\n"
    entry += "(bundle/'source-hashes.json').write_text(json.dumps({name:hashlib.sha256(content.encode()).hexdigest() for name,content in files.items()},indent=2)+'\\n')\n"
    entry += "os.environ.update(PYTHONNOUSERSITE='1',PIXI_HOME='/scratch/.pixi',PIXI_CACHE_DIR='/scratch/.cache/pixi',UV_CACHE_DIR='/scratch/.cache/uv',MPLCONFIGDIR='/scratch/.cache/matplotlib');os.environ.pop('PYTHONPATH',None)\n"
    entry += (
        "subprocess.run("
        + repr(
            [
                "pixi",
                "run",
                "--manifest-path",
                "/scratch/msa-decision-cpu/pixi.toml",
                "python",
                "-u",
                "-c",
                command,
            ]
        )
        + ",cwd=root,check=True)\n"
    )
    entry += "shutil.copyfile(root/'pixi.lock',bundle/'pixi.lock')\n"
    payload = (
        "-u -c exec(__import__('zlib').decompress(__import__('base64').b64decode("
        + repr(base64.b64encode(zlib.compress(entry.encode())).decode())
        + ")))"
    )
    r = dict(
        run_id=run_id,
        utc=dt.datetime.now(dt.UTC).isoformat(),
        output=output,
        source_bundle=bundle,
        status="submission_pending",
        session_ids=[],
        resources=dict(cpu=cpu, ram=ram, gpu=0),
        image=image,
        command=command,
        source_sha256={
            name: hashlib.sha256(content.encode()).hexdigest()
            for name, content in files.items()
        },
        payload_sha256=hashlib.sha256(entry.encode()).hexdigest(),
    )
    receipt.write_text(json.dumps(r, indent=2) + "\n")
    try:
        r["session_ids"] = create_form_session(
            Session(timeout=40),
            name="msa-" + run_id.replace("_", "-"),
            image=image,
            kind="headless",
            cmd="python",
            args=payload,
            env={"PYTHONNOUSERSITE": "1", "WORK": "/scratch"},
            **({"cores": cpu, "ram": ram} if cpu is not None else {}),
        )
        r["status"] = "accepted" if r["session_ids"] else "submission_ambiguous"
    except Exception as exc:
        r.update(status="submission_ambiguous", **submission_error_evidence(exc))
    receipt.write_text(json.dumps(r, indent=2) + "\n")
    print(json.dumps(r), flush=True)
    if r["status"] != "accepted":
        raise RuntimeError("Submission requires reconciliation")
    return r
