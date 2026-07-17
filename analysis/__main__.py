"""Allow ``python -m analysis <subcommand> ...`` in addition to
``python -m analysis.summarize_beta <subcommand> ...``."""

from analysis.summarize_beta import main

if __name__ == "__main__":
    raise SystemExit(main())
