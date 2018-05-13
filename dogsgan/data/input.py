import tensorflow as tf

from dogsgan.data.preprocess import preprocessed_dir
from pathlib import Path


def get_dataset():
    cache_location = Path('cache') / 'dataset.cache'
    cache_location.parent.mkdir(exist_ok=True)

    return tf.data.Dataset.list_files(str(preprocessed_dir) + '/*.jpeg') \
            .shuffle(buffer_size=100000) \
            .map(tf.read_file) \
            .map(tf.image.decode_jpeg) \
            .cache(filename=str(cache_location))


if __name__ == '__main__':
    tf.enable_eager_execution()
    dataset = get_dataset()

    it = dataset.make_one_shot_iterator()

    while True:
        try:
            it.get_next()
        except tf.errors.OutOfRangeError:
            break

    print('Done')


