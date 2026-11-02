"""Shared job loop for stages that hold their models across a --batch table of tab-separated paths."""
import os
import traceback


def run_jobs(table, single, work):
    """Calls work(*paths) per row, the last path being its output. A batch continues past a failed or finished row."""
    if not table:
        work(*single)
        return
    with open(table) as rows:
        jobs = [row.rstrip("\n").split("\t") for row in rows]
    for job in jobs:
        if os.path.isfile(job[-1]) and os.path.getsize(job[-1]):
            print("JOB_REUSED", job[-1], flush=True)
        elif not all(os.path.isfile(path) for path in job[:-1]):
            print("JOB_SKIPPED", job[-1], flush=True)
        else:
            try:
                work(*job)
            except Exception:
                traceback.print_exc()
                print("JOB_FAILED", job[-1], flush=True)
