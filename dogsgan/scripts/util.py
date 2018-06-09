from pathlib import Path


def execute(runner, load_from=None, mode='train', batch_size=128):
    if load_from is not None:
        runner.load_snapshot(Path(load_from))

    if mode == 'train':
        runner.train(batch_size=batch_size)
    elif mode == 'eval':
        runner.evaluate()
    else:
        raise Exception(f'Unknown mode {mode}')