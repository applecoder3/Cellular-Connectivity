import argparse 

from pathlib import Path
from src.config import ROOT, load_config
from src.pipeline import prepare

def main():
    parser = argparse.ArgumentParser(description= __doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("prepare", help="Prepare the data for analysis")
    command.add_argument("--config", type=Path, default=ROOT / "config.example.json", help="Path to the configuration file")
    command.add_argument("--output", type=Path, default=ROOT / "data" / "processed", help="Path to the output directory")
    args = parser.parse_args()

    try:
        manifest = prepare(load_config(args.config), args.output)
    except(OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"Preparation failed: {exc}\n")
    print(f"Prepared Franklin County dataset {manifest['version']}")
    print(f"Layer counts: {manifest['counts']}")
    print("Combined FCC gap analysis: " + ("enabled" if manifest["fcc"]["complete"] else "disabled"))

    for reason in manifest["fcc"]["reasons"]:
        print(f" {reason}")
    if manifest["temporal_mismatch"]:
        print(manifest["temporal_mismatch"])


if __name__ == "__main__":
    main()
