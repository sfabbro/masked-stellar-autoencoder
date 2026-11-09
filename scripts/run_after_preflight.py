"""Run a gate and its experiment in one Pixi interpreter, with durable errors."""

import argparse
import json
import runpy
import sys
from pathlib import Path


def execute(preflight, script, arguments):
    sys.argv = [str(preflight)]
    runpy.run_path(str(preflight), run_name="__main__")
    sys.argv = [str(script), *arguments]
    runpy.run_path(str(script), run_name="__main__")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-script", type=Path, required=True)
    parser.add_argument("--script", type=Path, required=True)
    args, remaining = parser.parse_known_args()
    try:
        execute(args.preflight_script, args.script, remaining)
    except Exception as exc:
        if "--output" in remaining:
            output = Path(remaining[remaining.index("--output") + 1])
            failure = output.with_name(output.name + "-launch-error.json")
            failure.write_text(
                json.dumps(
                    dict(
                        status="launch_failed",
                        error=repr(exc),
                        interpreter=sys.executable,
                    ),
                    indent=2,
                )
                + "\n"
            )
        raise


if __name__ == "__main__":
    main()
