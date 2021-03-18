import json
import os
import os.path as osp
import subprocess

# TODO: update the json path
with open("s2_tiles_eastern_aez_tasks_s2.json") as fh_in:
    taskstr_list_s2 = json.load(fh_in)

with open("s2_tiles_eastern_aez_tasks.json") as fh_in:
    taskstr_list_s1 = json.load(fh_in)

# TODO: confirm with core team if S3 write access was implemented
tifs_fld = "file:///home/jovyan/wa/u23/data/tifs20"

cached_db_s1 = "/home/jovyan/wa/u23/data/dscache/africa-20-2019-01--P6M.db"
cached_db_s2 = "/home/jovyan/wa/u23/data/dscache/africa-20-2019-07--P6M.db"


def worker_run(tifs_fld: str, task_str: str, cached_db: str):
    target_folder = osp.join(tifs_fld, task_str)

    if osp.exists(target_folder) and len(os.listdir(target_folder)) >= 15:
        print(f"folder {target_folder} exists, skip")
    else:
        cmd = [
            "odc-stats",
            "run",
            "--threads",
            "-1",
            "--overwrite",
            "--plugin",
            "gm-s2",
            "--location",
            tifs_fld,
            cached_db,
            f"{task_str}",
        ]
        # list(TR.run([task_str]))
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    for task_s1, task_s2 in zip(taskstr_list_s1, taskstr_list_s2):
        worker_run(tifs_fld, task_s1, cached_db_s1)
        worker_run(tifs_fld, task_s2, cached_db_s2)
