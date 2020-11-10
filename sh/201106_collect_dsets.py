import subprocess
import discovery_utils


def collect():

    cmd = "python data_gen/env.py --env_id ShapesTrain-v0 --fname data/shapes_train.h5 --num_episodes 1000 --seed 1"
    subprocess.call(cmd, shell=True)

    cmd = "python data_gen/env.py --env_id ShapesEval-v0 --fname data/shapes_eval.h5 --num_episodes 10000 --seed 2"
    subprocess.call(cmd, shell=True)


def collect_imm():

    cmd = "python data_gen/env.py --env_id ShapesImmovableTrain-v0 --fname data/shapes_imm_train.h5 --num_episodes 1000 --seed 1"
    subprocess.call(cmd, shell=True)

    cmd = "python data_gen/env.py --env_id ShapesImmovableEval-v0 --fname data/shapes_imm_eval.h5 --num_episodes 10000 --seed 2"
    subprocess.call(cmd, shell=True)


def main():

    name = "201106_collect_dsets"
    executor = discovery_utils.setup_executor(name)

    jobs = [
        executor.submit(collect),
        executor.submit(collect_imm)
    ]

    discovery_utils.check_jobs_done(jobs)


main()
