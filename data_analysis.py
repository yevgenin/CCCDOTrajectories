import math
from functools import partial
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.neighbors import LocalOutlierFactor
from statsmodels.stats.weightstats import DescrStatsW
from tqdm import tqdm
from tqdm.auto import tqdm

from utils import read_images, ParallelCalls, images_animation


class FileAnalysis:
    def __init__(self):
        self.alpha_dt_range = 20, 100
        self.max_mean_jump_xy = 100.
        self.max_mean_jump_xyz = 100.
        self.min_length = 50
        self.t_weight = 10.
        self.min_pairs_count = 50
        self.smooth = .99
        self.num_frames = None
        self.intensity_quantile = .5

    def get_analysis_for_file(self, file, pixel_size, **analysis):
        tq = tqdm(leave=False, total=5, disable=False)

        def step(name):
            tq.set_description(name)
            tq.update()

        if 'msds_df' not in analysis:
            step('get_localizations')
            locs_df = get_localizations(file=file, num_frames=self.num_frames)['localizations']

            step('get_registration')
            get_registration_result = get_registration(file=file, num_frames=self.num_frames, smooth=self.smooth,
                                                       warped_image=False)
            homography = get_registration_result['homography']
            original_image = get_registration_result['extra_info']['original_image']

            step('get_warp_localizations')
            locs_warped_df = get_warp_localizations(
                locs_df=locs_df,
                homographies=homography,
                pixel_size=pixel_size,
            )

            step('get_trajectories')
            result = self.get_and_filter_trajectories(locs_warped_df)
            trajectories_df = result['trajectories_df']
            trajectories_info_df = result['trajectories_info_df']

            step('get_msds')
            msds_df = self.get_and_filter_msds(trajectories_df)

            analysis = dict(
                analysis,
                msds_df=msds_df,
                locs_df=locs_df,
                original_image=original_image,
                homography=homography,
                trajectories_df=trajectories_df,
                trajectories_info_df=trajectories_info_df,
                locs_warped_df=locs_warped_df,
            )

        step('get_alphas')
        analysis.update(alphas_df=self.get_alphas(analysis['msds_df']))
        return analysis

    def get_and_filter_msds(self, trajectories_df):
        msds_df = get_msds(trajectories_df)['msds_df']
        msds_df = self.get_filter_msds(msds_df)
        return msds_df

    def get_and_filter_trajectories(self, locs_df):
        df = locs_df
        if self.intensity_quantile:
            min_intensity = np.quantile(df['I'], self.intensity_quantile)
            df = df[df['I'] >= min_intensity]

        result = get_cluster_trajectories(df, t_weight=self.t_weight)
        trajectories_df = result['trajectories_df']
        trajectories_info_df = get_trajectories_info(trajectories_df)

        return dict(
            trajectories_df=self.get_filter_trajectories(trajectories_df, trajectories_info_df),
            trajectories_info_df=trajectories_info_df
        )

    def get_filter_trajectories(self, trajectories_df, trajectories_info_df):
        def func(mean_jump_xy, mean_jump_xyz, frame_range, length):
            return (
                    (self.max_mean_jump_xy is not None and mean_jump_xy <= self.max_mean_jump_xy)
                    and (self.min_length is not None and length >= self.min_length)
                    and (self.max_mean_jump_xyz is not None and mean_jump_xyz <= self.max_mean_jump_xyz)
            )

        keep = trajectories_info_df.apply(lambda _: func(**_), axis=1)
        trajectories_df = trajectories_df.groupby('traj').filter(lambda _: keep[_.name])
        return trajectories_df

    def get_filter_msds(self, msds_df):
        df = msds_df
        if self.min_pairs_count:
            df = df[df['pairs_count'] >= self.min_pairs_count]
        return df

    def get_alphas(self, msds_df):
        def func(df):
            if len(df) < 2:
                return
            df = df.reset_index()
            df = df[df['dt'] > 0]
            df_fit = df[df['dt'].between(*self.alpha_dt_range)]
            x, y = df_fit['dt'].values, df_fit['msd'].values
            p = np.polyfit(np.log(x), np.log(y), 1)

            x_pred = df['dt'].values
            y_pred = np.exp(np.polyval(p, np.log(x_pred)))
            alpha, intercept = p
            return pd.Series(dict(alpha=alpha, intercept=intercept, x_fit=x, y_fit=y, x_pred=x_pred, y_pred=y_pred))

        alphas_df = msds_df.groupby('traj').apply(func)
        alphas_df.attrs = dict(dt_range=self.alpha_dt_range)
        return alphas_df


def get_localizations(file, num_frames):
    csv_df: pd.DataFrame = pd.read_csv(file)
    df = csv_df

    if 'frame' not in df or not len(df):
        raise ValueError

    df = df.rename(columns={'x [nm]': 'x', 'y [nm]': 'y', 'z [nm]': 'z', 'frame': 'frame', 'intensity [au]': 'I'})
    df = df[['frame', *'xyz', 'I']]
    df = df.astype({'frame': int})
    df['frame'] = df['frame'] - 1

    if num_frames:
        df = df[df['frame'] < num_frames]

    return dict(
        localizations=df,
        extra_info=dict(
            csv_df=csv_df
        )
    )


def get_image(file, num_frames):
    return dict(
        image=read_images((file_dir := Path(file).parent.resolve().as_posix()),
                          n_frames=num_frames),
        extra_info=dict(
            file_dir=file_dir
        )
    )


class SequenceRegistration:
    def __init__(self, images, smooth=.99, parallel=True, **kwargs):
        self.parallel = parallel
        self.images = images
        self.smooth = smooth

    def make_registration_df(self, dst_index=0, warp=False):
        im0 = self.images[dst_index]
        r0 = RigidRegistrationOpenCV(image1=im0)
        features1 = r0.get_features_for_image(im0)
        r0.features1 = features1

        def get(im):
            analysis = RigidRegistrationOpenCV(im, im0, registration=r0)
            analysis.make_registration(warp=warp)
            return analysis.as_dict()

        res = ParallelCalls(parallel=self.parallel, prefer='threads', desc='registration').map(get, self.images)
        df = pd.DataFrame.from_records(res)
        df['smooth_homography'] = self.get_smooth_homography(df['homography'])
        self.registration_df = df

    def make_smooth_warp(self):
        df = self.registration_df
        df['warped'] = ParallelCalls(prefer='threads', desc='warp').call_funcs(
            [partial(warp_image, im, h) for im, h in df[['image0', 'smooth_homography']].values])

    def get_homography_analysis(self, homographies):
        def get(M):
            if M is None:
                return dict()
            return dict(
                theta=-math.atan2(M[0, 1], M[0, 0]) * 180 / math.pi,
                Tx=M[0, 2],
                Ty=M[1, 2],
                scale=math.sqrt(M[0, 0] ** 2 + M[1, 0] ** 2)
            )

        return pd.DataFrame.from_records(homographies.apply(get))

    def get_smooth_homography(self, homographies):
        h = homographies
        value = h[0]
        result = [value]
        for prev_value in h[1:]:
            if prev_value is not None and not np.isnan(prev_value.sum()):
                value = self.smooth * value + (1 - self.smooth) * prev_value
            result.append(value)
        return result

    def show_points(self, frame):
        points, h = self.registration_df.loc[frame, ['points', 'homography']]
        p = points[0][:, 0, :], points[1][:, 0, :]
        p1 = p[0] @ h[:, :2].T + h[:, 2]
        plt.plot(*p[0].T, '.', )
        plt.plot(*p[1].T, '.')
        plt.plot(*p1.T, 'x')

    def show_homography(self):
        df = self.registration_df
        func = self.get_homography_analysis

        func(df['homography']).plot()
        func(df['smooth_homography']).plot()

    def show(self, skip=None):
        warped = list(self.registration_df['warped'].dropna())

        image0 = list(self.registration_df['image0'].dropna())
        images_animation(warped[::skip])
        images_animation(image0[::skip])

    def show_pair(self, frame):
        RigidRegistrationOpenCV(**self.registration_df.loc[frame]).show()


class RigidRegistrationOpenCV:
    def __init__(self, image0=None, image1=None, registration: 'RigidRegistrationOpenCV' = None, **kwargs):
        self.image0 = image0
        self.image1 = image1
        self.matches_thresh = 0.5

        if registration is None:
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
            self.detector = cv2.SIFT_create(contrastThreshold=1e-2)
            self.features1 = None
        else:
            self.matcher = registration.matcher
            self.detector = registration.detector
            self.image1 = registration.image1
            self.features1 = registration.features1

    def as_dict(self):
        return {k: getattr(self, k, None) for k in
                ['homography', 'image0', 'image1', 'features', 'points'
                    , 'matches', 'warped']}

    def show(self):
        (kp0, d0), (kp1, d1) = self.features
        kp0_image = cv2.drawKeypoints(self.preproc_for_features(self.image0), kp0, None)
        kp1_image = cv2.drawKeypoints(self.preproc_for_features(self.image1), kp1, None)
        images_animation((self.warped, self.image1))
        images_animation((self.image0, self.image1))
        images_animation((kp0_image, kp1_image))

    def preproc_for_features(self, im):
        return cv2.normalize(im, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    def get_features(self):
        func = self.get_features_for_image
        features0 = func(self.image0)

        if self.features1 is None:
            features1 = func(self.image1)
        else:
            features1 = self.features1

        features = features0, features1
        self.features = features
        return features

    def get_features_for_image(self, im):
        im = self.preproc_for_features(im)
        kp, d = self.detector.detectAndCompute(im, None)
        return kp, d

    def get_images_homography(self):
        features = self.get_features()
        homography = self.get_features_homography(*features)
        return homography

    def get_warped(self, homography):
        im = self.image0
        warped = warp_image(im, homography)
        self.warped = warped
        return warped

    def get_features_homography(self, *features):
        (kp0, d0), (kp1, d1) = features
        matches = self.matcher.knnMatch(d0, d1, k=2)
        points = self.get_points_from_matches(kp0, kp1, matches)

        if not len(points[0]) or not len(points[1]):
            return

        homography, _ = cv2.estimateAffinePartial2D(*points)
        self.matches = matches
        self.homography = homography
        return homography

    def get_points_from_matches(self, kp1, kp2, matches):
        filtered_matches = [m for m, n in matches if m.distance < self.matches_thresh * n.distance]
        points = [kp1[_.queryIdx].pt for _ in filtered_matches], [kp2[_.trainIdx].pt for _ in filtered_matches]
        *points, = map(lambda _: np.float32(_).reshape(-1, 1, 2), points)
        self.points = points
        return points

    def make_registration(self, warp=False):
        homography = self.get_images_homography()
        if warp and homography is not None:
            self.get_warped(homography)


def warp_image(im, homography):
    warped = cv2.warpAffine(im, homography, (im.shape[1], im.shape[0]))
    return warped


def get_registration(file, num_frames, smooth, warped_image):
    image = get_image(file=file, num_frames=num_frames)['image']
    assert isinstance(image, np.ndarray)

    sr = SequenceRegistration(image, smooth=smooth, parallel=False)
    sr.make_registration_df(warp=False)

    extra_info = dict(
        original_image=image
    )
    if warped_image:
        sr.make_smooth_warp()
        extra_info.update(
            warped_image=np.stack(sr.registration_df['warped']),
        )

    return dict(
        homography=np.stack(sr.registration_df['smooth_homography']),
        extra_info=extra_info
    )


def get_warp_localizations(locs_df, homographies, pixel_size):
    df = locs_df.copy()
    items = tqdm(locs_df.groupby('frame').groups.items(), leave=False, desc='warp_locs')
    for (name, indices), h in zip(items, homographies):
        xy = df.loc[indices, ['x', 'y']].values
        xy_new = xy @ h[:, :2].T + h[:, 2] * pixel_size
        df.loc[indices, ['x', 'y']] = xy_new
    return df


def get_cluster_trajectories(locs_df, use_t=True, t_weight=10.):
    df = locs_df.copy()

    clustering = DBSCAN(eps=.5e3)
    if use_t:
        df['tw'] = df['frame'] * t_weight
        fit_columns = ['tw', *'xyz']
    else:
        fit_columns = [*'xyz']

    clf_outlier = LocalOutlierFactor()
    outlier = clf_outlier.fit_predict(df[fit_columns])
    df = df[outlier >= 0]
    df['traj'] = clustering.fit_predict(df[fit_columns])
    df = df[df['traj'] >= 0]
    df = df[['traj']].join(locs_df)
    df = df.groupby(['traj', 'frame']).mean().sort_index()

    return dict(
        trajectories_df=df
    )


def get_msds(trajectories_df):
    def func(trajectories_group_df: pd.DataFrame):
        df = trajectories_group_df.reset_index()

        t = df['frame'].values
        xyz = df[[*'xyz']].values

        dt = -t[:, None] + t[None, :]
        squared_distance = np.sum((xyz[:, None] - xyz[None, :]) ** 2, axis=-1)

        pairs_df = pd.DataFrame({'dt': dt.flatten(), 'squared_distance': squared_distance.flatten()}).query('dt>0')

        df = (
            pairs_df.groupby('dt')['squared_distance'].aggregate([
                ('msd', 'mean'), ('pairs_count', 'count'),
                ('std', 'std'), ('ci', lambda _: DescrStatsW(_).tconfint_mean())]
            )
        )
        return df

    msds_df = trajectories_df.groupby('traj').apply(func)
    assert msds_df.index.is_unique
    return dict(
        msds_df=msds_df,
    )


def get_trajectories_info(trajectories_df: pd.DataFrame):
    df = trajectories_df

    def get_mean_jump_xy(df, axes='xy'):
        df = df.reset_index()
        df = df[['frame', *axes]]
        df = df.diff()
        df = df[df['frame'] == 1]
        xy = df[[*axes]]
        return np.sqrt(np.mean((xy ** 2).sum(axis=1)))

    df = pd.DataFrame().assign(
        mean_jump_xy=df.groupby('traj').apply(get_mean_jump_xy, axes='xy'),
        mean_jump_xyz=df.groupby('traj').apply(partial(get_mean_jump_xy, axes='xyz')),
        frame_range=df.reset_index().groupby('traj').apply(lambda _: _['frame'].max() - _['frame'].min()),
        length=df.groupby('traj').apply(len)
    )
    return df


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='input file')
    parser.add_argument('--output', default='.', help='output dir')
    parser.add_argument('--pixel_size', type=int, default=110, help='image pixel size for the analysis')
    args = parser.parse_args()
    input_file = Path(args.input).absolute().as_posix()
    output_dir = Path(args.output).absolute().as_posix()

    file_analysis = FileAnalysis()
    result = file_analysis.get_analysis_for_file(file=input_file, pixel_size=args.pixel_size)

    result['locs_df'].to_csv(Path(output_dir, 'localizations.csv'))
    result['locs_warped_df'].to_csv(Path(output_dir, 'localizations_warped.csv'))
    result['trajectories_df'].to_csv(Path(output_dir, 'trajectories.csv'))
    result['msds_df'].to_csv(Path(output_dir, 'msds.csv'))
    homographies_df = pd.DataFrame(np.asarray(result['homography']).reshape(-1, 6))
    homographies_df.to_csv(Path(output_dir, 'homographies.csv'))
