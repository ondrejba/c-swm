import subprocess
import discovery_utils


def run(i):

    cmd = ["python", "data_gen/env.py", "--env_id", "ShapesCursorTrain-v0",
           "--fname", "data/shapes_cursor_train_{:d}.h5".format(i),
           "--num_episodes", "1000", "--seed", "1"]

    subprocess.call(cmd)

    cmd = ["python", "data_gen/env.py", "--env_id", "ShapesCursorEval-v0",
           "--fname", "data/shapes_cursor_eval_{:d}.h5".format(i),
           "--num_episodes", "10000", "--seed", "2"]

    subprocess.call(cmd)


def main():

    name = "201102_shapes_cursor_dsets"
    executor = discovery_utils.setup_executor(name)

    jobs = []
    for i in range(10):
        jobs.append(executor.submit(run, i))
    discovery_utils.check_jobs_done(jobs)


main()
