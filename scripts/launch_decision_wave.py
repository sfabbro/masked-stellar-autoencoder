"""Submit immutable CANFAR decision jobs with durable source bundles and receipts."""

import argparse
import base64
import datetime as dt
import hashlib
import json
import os
import socket
import urllib.parse
import urllib.request
import zlib
from pathlib import Path

from launch_stellar_improvement import create_form_session, submission_error_evidence

ROOT = "/arc/projects/k-pop/msa_runs/stellar-improvement-20261015/"
# Headless image; the entrypoint installs pixi and the job's environment itself.
IMAGE = os.environ.get("MSA_CANFAR_IMAGE", "images.canfar.net/astroai/base:latest")


def addresses():
    result = {}
    for host in [
        "github.com",
        "conda.anaconda.org",
        "conda-mapping.prefix.dev",
        "raw.githubusercontent.com",
        "pypi.org",
        "files.pythonhosted.org",
        "download.pytorch.org",
        "download-r2.pytorch.org",
        "data.sdss.org",
        "zenodo.org",
        "vizier.cfa.harvard.edu",
        "dc.zah.uni-heidelberg.de",
        "dc.g-vo.org",
        "gea.esac.esa.int",
        "sagadatabase.jp",
        "simbad.cfa.harvard.edu",
        # Public survey catalogues staged by stage_master_survey_files.py.
        "www.lamost.org",
        "data.desi.lbl.gov",
        "www.rave-survey.org",
    ]:
        try:
            result[host] = socket.gethostbyname(host)
        except socket.gaierror:
            url = "https://dns.google/resolve?" + urllib.parse.urlencode(
                dict(name=host, type="A")
            )
            with urllib.request.urlopen(url, timeout=15) as response:
                data = json.load(response)
            result[host] = next(a["data"] for a in data["Answer"] if a["type"] == 1)
    return result


def submit(
    run_id,
    files,
    command,
    *,
    gpu,
    cpu=4,
    ram=64,
    branch="codex/multisurvey-abundance-union-20261028",
    commit="60c47fcf609f5ee80a1f985cc95ab8ca648793e9",
    record_directory=Path("experiments/xp/decision-wave-20261106"),
    attempt=None,
):
    from canfar.sessions import Session

    record_directory.mkdir(parents=True, exist_ok=True)
    if attempt is not None and (
        not isinstance(attempt, int) or isinstance(attempt, bool) or attempt < 2
    ):
        raise ValueError("A retry attempt must be an explicit integer >=2")
    attempt_name = run_id if attempt is None else f"{run_id}-attempt-{attempt}"
    receipt = record_directory / (attempt_name + ".json")
    if receipt.exists():
        raise RuntimeError(
            "Reconcile existing receipt; never resubmit: " + str(receipt)
        )
    output = ROOT + run_id
    bundle = ROOT + "decision_wave_source_bundles_20261106_v1/" + attempt_name
    resolved = addresses()
    entry = (
        Path("scripts/canfar_dns_proxy.py").read_text()
        + "\nimport pathlib,subprocess,json,hashlib\n"
    )
    entry += "start_dns_proxy(" + repr(resolved) + ")\n"
    entry += "root=pathlib.Path('/scratch/src/sfabbro/masked-stellar-autoencoder'); root.parent.mkdir(parents=True,exist_ok=True)\n"
    entry += (
        "bundle=pathlib.Path("
        + repr(bundle)
        + ");bundle.mkdir(parents=True,exist_ok=False)\n"
    )
    entry += "files=" + repr(files) + "\n"
    entry += "for name,content in files.items():\n p=bundle/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(content)\n"
    entry += "(bundle/'source-hashes.json').write_text(json.dumps({name:hashlib.sha256(content.encode()).hexdigest() for name,content in files.items()},indent=2)+'\\n')\n"
    entry += (
        "os.environ.update(GIT_CONFIG_COUNT='1',GIT_CONFIG_KEY_0='http.curloptResolve',GIT_CONFIG_VALUE_0="
        + repr("github.com:443:" + resolved["github.com"])
        + ")\n"
    )
    entry += (
        "subprocess.run(['git','clone','--depth','100','--branch',"
        + repr(branch)
        + ",'https://github.com/sfabbro/masked-stellar-autoencoder.git',str(root)],check=True)\n"
    )
    entry += (
        "subprocess.run(['git','checkout','--detach',"
        + repr(commit)
        + "],cwd=root,check=True)\n"
    )
    entry += "(bundle/'base-commit.txt').write_text(subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True))\n"
    entry += "for name,content in files.items():\n p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(content)\n"
    route = (
        '    if [[ "${MSA_EXPERIMENT_TASK:-training}" == decision-wave ]]; then\n      "${pixi_run[@]}" '
        + command
        + '\n      exit "$?"\n    fi\n'
    )
    needle = '    if [[ "${MSA_EXPERIMENT_TASK:-training}" == age-data-audit ]]; then'
    entry += (
        "path=root/'batch_scripts/canfar_entrypoint.sh';s=path.read_text();assert "
        + repr(needle)
        + " in s;path.write_text(s.replace("
        + repr(needle)
        + ","
        + repr(route + needle)
        + ",1))\n"
    )
    entry += "(bundle/'entrypoint.sh').write_bytes(path.read_bytes())\n"
    entry += "os.environ.pop('PYTHONPATH',None)\n"
    entry += "subprocess.run(['bash',str(path),'pretrain-experiments'],cwd=root,check=True)\n"
    payload = (
        "-u -c exec(__import__('zlib').decompress(__import__('base64').b64decode("
        + repr(base64.b64encode(zlib.compress(entry.encode())).decode())
        + ")))"
    )
    r = dict(
        run_id=run_id,
        attempt=attempt or 1,
        output=output,
        source_bundle=bundle,
        source_commit=commit,
        branch=branch,
        command=command,
        utc=dt.datetime.now(dt.UTC).isoformat(),
        status="submission_pending",
        session_ids=[],
        resources=dict(gpu=gpu, cores=cpu, ram=ram),
        image=IMAGE,
        source_sha256={
            name: hashlib.sha256(content.encode()).hexdigest()
            for name, content in files.items()
        },
        payload_sha256=hashlib.sha256(entry.encode()).hexdigest(),
    )
    receipt.write_text(json.dumps(r, indent=2) + "\n")
    try:
        env = {
            "PYTHONNOUSERSITE": "1",
            "WORK": "/scratch/src",
            "MSA_XP_EXPERIMENTS": "0",
            "MSA_PRETRAIN_RESUME": output + "/unused-bootstrap.pth",
            "MSA_EXPERIMENT_TASK": "decision-wave",
            "MSA_EXPERIMENT_RUNS": run_id,
            "MSA_EXPERIMENT_OUTPUT": output,
            "MPLCONFIGDIR": "/scratch/.cache/matplotlib",
        }
        if not gpu:
            env["MSA_EXPERIMENT_CPU_ONLY"] = "1"
        r["session_ids"] = create_form_session(
            Session(timeout=40),
            name="msa-" + run_id.replace("_", "-"),
            image=IMAGE,
            kind="headless",
            cmd="python",
            args=payload,
            env=env,
            cores=cpu,
            ram=ram,
            **({"gpu": gpu} if gpu else {}),
        )
        r["status"] = "accepted" if r["session_ids"] else "submission_ambiguous"
    except Exception as exc:
        r.update(status="submission_ambiguous", **submission_error_evidence(exc))
    receipt.write_text(json.dumps(r, indent=2) + "\n")
    print(json.dumps(r), flush=True)
    if r["status"] != "accepted":
        raise RuntimeError("Reconcile ambiguous submission before continuing")
    return r


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", nargs="+", required=True)
    args = p.parse_args()
    import yaml

    config = Path("configs/decision-six-wave-20261106.yaml")
    cfg = yaml.safe_load(config.read_text())
    allowed = {a["run_id"] for a in cfg["arms"]}
    for run_id in args.runs:
        if run_id not in allowed:
            raise ValueError("Unknown declared arm")
        submit(
            run_id,
            {
                str(config): config.read_text(),
                "scripts/run_decision_six_pilot.py": Path(
                    "scripts/run_decision_six_pilot.py"
                ).read_text(),
            },
            "python -u scripts/run_decision_six_pilot.py --config "
            + str(config)
            + ' --run-id "$MSA_EXPERIMENT_RUNS" --output "$experiment_output"',
            gpu=1,
        )


if __name__ == "__main__":
    main()
