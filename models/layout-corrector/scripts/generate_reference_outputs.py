from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--starter-dir", required=True)
    parser.add_argument("--corrector-job-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sampling", default="deterministic")
    parser.add_argument("--corrector-t-list", nargs="*", type=int, default=[10, 20, 30])
    parser.add_argument("--no-gumbel-noise", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.parse_args()
    raise NotImplementedError(
        "Reference generation requires the original starter kit and vendor runtime."
    )


if __name__ == "__main__":
    main()
