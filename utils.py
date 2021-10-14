from functools import partial
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from IPython.core.display import display
from imageio import imread
from joblib import delayed
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm import tqdm
from tqdm.auto import tqdm


def read_images(file_dir, n_frames=None):
    file_dir = Path(file_dir)
    df = pd.Series(file_dir.glob('*.tif')).to_frame('file')
    df['name'] = df['file'].apply(lambda x: Path(x).name)
    df['image_index'] = df['name'].str.extract('(.*).tif').astype(int)
    df = df.sort_values('image_index', ignore_index=True)
    assert np.all(df['image_index'] == np.arange(len(df)))

    df = df[:n_frames]

    # images = ParallelCalls(prefer='threads').apply_df_rows(df, lambda row: imread(row['file']))
    images = df['file'].apply(imread)
    image = np.stack(images)
    return image


class ParallelCalls:
    def __init__(self, parallel=True, desc=None, **kwargs):
        self.desc = desc
        self.parallel = parallel

        kwargs = dict(dict(prefer='threads', verbose=False, n_jobs=-1), **kwargs)
        self.joblib_parallel = joblib.Parallel(**kwargs)

    def apply_df_rows(self, df, func):
        return self.call_funcs([partial(func, row) for _, row in df.iterrows()])

    def map(self, func, args):
        return self.call_funcs([partial(func, _) for _ in args])

    def call_funcs(self, funcs):
        funcs = tqdm(funcs, leave=False, desc=self.desc)
        if self.parallel:
            return self.joblib_parallel(delayed(_)() for _ in funcs)
        else:
            return [_() for _ in funcs]


def func_animation(funcs, **kwargs):
    return FuncAnimation(plt.gcf(), lambda _: _(), frames=funcs, **kwargs)


def images_animation(images, func=None, interval=100, **kwargs):
    # ax = plt.gca()
    a = plt.imshow(images[0], **kwargs)
    if func is None:
        def func(im):
            a.set_data(im)
            plt.clim(im.min(), im.max())
            plt.draw()
    ani = func_animation([partial(func, im) for im in images], interval=interval)
    display(ani)
    plt.close()
