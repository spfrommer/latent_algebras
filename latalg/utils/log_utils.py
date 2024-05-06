from __future__ import annotations

import os
import re
import signal
import socket
import threading
import wandb

from latalg.utils import dirs, file_utils



def get_wandb_id(project: str, run_name: str) -> str:
    wandb_id_path = dirs.out_path(project, run_name, 'wandb_id.txt')
    return file_utils.read_file(wandb_id_path)


def upload_dataset_wandb(
        project: str,
        artifact: str,
        directory: str,
        run_name='data_upload'
    ):

    run = wandb.init(
        project=project,
        job_type='data',
        name=run_name,
        dir='/tmp',
        group='data_group'
    )
    artifact = wandb.Artifact(artifact, type='dataset')
    artifact.add_dir(directory)
    run.log_artifact(artifact)
    run.finish()


def force_finish_wandb(run_path: str):
    # Adapted from: https://github.com/wandb/wandb/issues/4929
    wandb_path = 'wandb/latest-run/logs/debug-internal.log'
    with open(os.path.join(run_path, wandb_path), 'r') as f:
        last_line = f.readlines()[-1]

    match = re.search(r'(HandlerThread:|SenderThread:)\s*(\d+)', last_line)
    if match:
        pid = int(match.group(2))
        print(f'wandb pid: {pid}')
    else:
        print('Cannot find wandb process-id.')
        return
    
    try:
        os.kill(pid, signal.SIGKILL)
        print(f"Process with PID {pid} killed successfully.")
    except OSError:
        print(f"Failed to kill process with PID {pid}.")


def launch_tensorboard(tensorboard_dir, port):
    # Use threading so tensorboard is automatically closed on process end
    command = f'tensorboard --bind_all --port {port} '\
              f'--logdir {tensorboard_dir} > /dev/null '\
              f'--samples_per_plugin images=100000 '\
              f'--window_title {socket.gethostname()} 2>&1'
    t = threading.Thread(target=os.system, args=(command,))
    t.start()

    print(f'Launching tensorboard on http://localhost:{port}')