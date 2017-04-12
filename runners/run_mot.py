from runners import RunnerParser
from madrl_environments.mot.mot_kitti import MotKittiEnv
from runners.rurllab import RLLabRunner

ENV_OPTIONS = [
    ('n_trackers', int, 3, ''),
]

def main(parser):
    args = parser.args
    env_config = dict(n_trackers=args.n_trackers)
    env = MotKittiEnv(**env_config)
    RLLabRunner(env, args)()

if __name__ == '__main__':
    main(RunnerParser(ENV_OPTIONS))
