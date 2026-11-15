"""Submit the master spectroscopic label census as one CANFAR CPU headless job.

The job re-acquires every native source on CANFAR (SDSS SAS, VizieR, ESA Gaia
archive), writes the observation/star tables and manifest to ARC, and joins
against every already staged ``native_observations`` sidecar under the
stellar-improvement root to report new unique and new metal-poor stars.

Credentials (never committed), first match wins:

* ``CANFAR_TOKEN`` (+ optional ``CANFAR_URL``): bearer token;
* ``CANFAR_CERTIFICATE``: path to a cadcproxy.pem;
* ``CADC_PROXY_PEM``: PEM text, written to ``~/.ssl/cadcproxy.pem`` (0600);
* ``CADC_USERNAME`` + ``CADC_PASSWORD``: a fresh 10-day proxy certificate is
  fetched with ``cadc-get-cert`` through a temporary 0600 netrc that is
  deleted immediately afterwards.
"""

import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))

RUN_ID = "master_label_census_20261110_v1"
FILES = {
    "scripts/run_master_label_census.py": REPO / "scripts/run_master_label_census.py",
    "src/masked_stellar_autoencoder/training/master_labels.py": REPO
    / "src/masked_stellar_autoencoder/training/master_labels.py",
}
EMPTY = (
    "src/masked_stellar_autoencoder/__init__.py",
    "src/masked_stellar_autoencoder/training/__init__.py",
)


SKAHA_URL = "https://ws-uv.canfar.net/skaha/v1"  # v0 advertised but 404s


def prepare_credentials() -> str:
    """Pick a credential; runtime credentials also need an explicit server URL."""
    source = _credential_source()
    if source != "saved canfar auth":
        os.environ.setdefault("CANFAR_URL", SKAHA_URL)
    return source


CRED_URL = "https://cadc-west-01.canfar.net/cred/generate"


def _valid_pem(text: str, margin_s: int = 3600) -> bool:
    """True when the PEM's certificate is still valid ``margin_s`` from now."""
    import subprocess

    result = subprocess.run(
        ["openssl", "x509", "-noout", "-checkend", str(margin_s)],
        input=text,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def _generate_pem(user: str, password: str, days: int = 10) -> str:
    """Proxy certificate from the CADC credential service (basic auth).

    Uses ``requests``, which follows the environment's HTTPS proxy and CA
    bundle; ``cadc-get-cert`` does not in sandboxed sessions.
    """
    import requests

    r = requests.get(
        CRED_URL, params={"daysValid": days}, auth=(user, password), timeout=60
    )
    r.raise_for_status()
    if "BEGIN" not in r.text:
        raise SystemExit("CADC credential service returned no certificate")
    return r.text


def _credential_source() -> str:
    if os.environ.get("CANFAR_TOKEN"):
        return "CANFAR_TOKEN"
    if os.environ.get("CANFAR_CERTIFICATE"):
        if not Path(os.environ["CANFAR_CERTIFICATE"]).is_file():
            raise SystemExit("CANFAR_CERTIFICATE does not name a file")
        return "CANFAR_CERTIFICATE"
    path = Path.home() / ".ssl" / "cadcproxy.pem"
    path.parent.mkdir(mode=0o700, exist_ok=True)

    def use(text, source):
        path.write_text(text if text.endswith("\n") else text + "\n")
        path.chmod(0o600)
        os.environ["CANFAR_CERTIFICATE"] = str(path)
        return source

    # An expired CADC_PROXY_PEM is skipped, never written over a newer file.
    pem = os.environ.get("CADC_PROXY_PEM")
    if pem and _valid_pem(pem):
        return use(pem, "CADC_PROXY_PEM")
    if path.is_file() and _valid_pem(path.read_text()):
        os.environ["CANFAR_CERTIFICATE"] = str(path)
        return "~/.ssl/cadcproxy.pem"
    user, password = os.environ.get("CADC_USERNAME"), os.environ.get("CADC_PASSWORD")
    if user and password:
        return use(_generate_pem(user, password), "CADC_USERNAME/CADC_PASSWORD")
    if pem:
        raise SystemExit("CADC_PROXY_PEM has expired; replace it in the environment")
    saved = Path.home() / ".config" / "canfar" / "config.yaml"
    if saved.is_file():
        return "saved canfar auth"
    raise SystemExit(
        "No CANFAR credential: set CANFAR_TOKEN, CANFAR_CERTIFICATE, CADC_PROXY_PEM "
        "or CADC_USERNAME + CADC_PASSWORD"
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-id", default=RUN_ID)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--cpu", type=int, default=4)
    p.add_argument("--ram", type=int, default=32)
    p.add_argument(
        "--check-auth",
        action="store_true",
        help="prepare credentials and list CANFAR sessions, then exit",
    )
    a = p.parse_args()
    files = {name: path.read_text() for name, path in FILES.items()}
    files.update({name: "" for name in EMPTY})
    from launch_decision_wave import ROOT  # same ARC tree as every decision job

    work = ROOT + "master_label_sources_20261110_v1"
    output = ROOT + a.run_id
    argv = [
        "scripts/run_master_label_census.py",
        "--work",
        work,
        "--output",
        output,
        "--existing-root",
        ROOT.rstrip("/"),
    ]
    command = (
        "import runpy,sys;sys.argv="
        + repr(argv)
        + ";runpy.run_path('scripts/run_master_label_census.py',run_name='__main__')"
    )
    print(dict(run_id=a.run_id, output=output, work=work, files=sorted(files)))
    if a.dry_run:
        return
    print("credential:", prepare_credentials())
    if a.check_auth:
        from canfar.sessions import Session

        print("CANFAR sessions visible:", len(Session(timeout=40).fetch()))
        return
    from launch_decision_cpu import submit_cpu

    submit_cpu(
        a.run_id,
        files,
        command,
        cpu=a.cpu,
        ram=a.ram,
        extra_dependencies='pyvo=">=1.5"\nrequests="*"\n',
    )


if __name__ == "__main__":
    main()
