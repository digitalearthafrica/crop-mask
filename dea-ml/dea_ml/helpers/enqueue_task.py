# TODO: 1. publish taskstr or json into AWS sqs queue
import subprocess

# TODO: do we allow the s3 db url ?
db = "/g/data/u23/..."  # or "s3://"
queue = "crop-mask-dev"
cmd = ["odc-stats", "publish-tasks", f"{db}", f"{queue}", "--verbose"]

subprocess.run(cmd, check=True)
