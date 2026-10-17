import os

# Before anything imports pyplot: the suite renders figures and never shows them.
os.environ.setdefault("MPLBACKEND", "Agg")
