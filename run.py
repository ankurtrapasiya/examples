"""Run the scripts behind the blog's numbers.

    python run.py                    list every script
    python run.py scoring-a-model    run the scripts in one folder
    python run.py bagging/bagging.py run one script
    python run.py all                run everything, and say which failed

Charts are written to charts/<post>/ next to this file.
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SKIP = {"fetch_prices.py"}       # downloads fresh prices and overwrites data/: run it on purpose


def scripts():
    return sorted(p for p in HERE.glob("*/*.py") if p.name not in SKIP and not p.parent.name.startswith("."))


def run(paths):
    failed = []
    for p in paths:
        print(f"\n=== {p.relative_to(HERE)} ===", flush=True)
        if subprocess.run([sys.executable, str(p)], cwd=HERE).returncode != 0:
            failed.append(p.relative_to(HERE))
    if len(paths) > 1:
        print(f"\n{len(paths) - len(failed)} of {len(paths)} scripts ran" + (f"; failed: {', '.join(map(str, failed))}" if failed else ""))
    return 1 if failed else 0


def main(args):
    if not args:
        print(__doc__)
        for p in scripts():
            print("  ", p.relative_to(HERE))
        return 0
    if args == ["all"]:
        return run(scripts())
    chosen = []
    for a in args:
        target = HERE / a
        if target.is_dir():
            chosen += sorted(target.glob("*.py"))
        elif target.is_file():
            chosen.append(target)
        else:
            print(f"no script or folder called {a!r}; run with no arguments to list them")
            return 2
    return run(chosen)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
