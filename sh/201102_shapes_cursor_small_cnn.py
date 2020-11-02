import os
import subprocess
import discovery_utils


def run_train(i):

    cmd = ["python", "train.py", "--dataset", "data/shapes_cursor_train.h5", "--encoder", "small",
           "--name", "shapes_cursor_{:d}".format(i), "--action-dim", "8", "--copy-action", "--num-objects", "6"]

    subprocess.call(cmd)


def run_eval(i, log_file):

    for steps in [1, 5, 10]:

        f = open(log_file, "a+")
        cmd = ["python", "eval.py", "--dataset", "data/shapes_cursor_eval.h5", "--save-folder",
               "checkpoints/shapes_cursor_{:d}".format(i), "--num-steps", "{:d}".format(steps)]

        subprocess.call(cmd, stdout=f, stderr=f)
        f.close()


def main():

    name = "201102_shapes_cursor_small_cnn"
    executor = discovery_utils.setup_executor(name)

    log_file = "res/{:s}".format(name)
    log_folder = os.path.dirname(log_file)
    if not os.path.isdir(log_folder):
        os.makedirs(log_folder)

    #jobs = []
    #for i in range(10):
    #    jobs.append(executor.submit(run_train, i))
    #discovery_utils.check_jobs_done(jobs)

    jobs = []
    for i in range(10):
        tmp_log_file = "{:s}_{:d}.txt".format(log_file, i)
        jobs.append(executor.submit(run_eval, i, tmp_log_file))
    discovery_utils.check_jobs_done(jobs)


main()
