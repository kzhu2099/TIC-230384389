'''Pulsation periods of the ZZ Ceti star TIC 230384389 in all 39 TESS sectors with 120-s data (2019-2024).

Measures the six periods of Romero et al. (2022) and a new signal in each sector and observing block, finds every significant peak near 750 s in each sector and TESS orbit, and checks the target pixel files. Writes Figure_1.pdf and the five tables of the Zenodo deposit (doi:10.5281/zenodo.23439428) to output/. The TESS files (about 580 MB) are downloaded into tess_cache/ on the first run.

Usage: python main.py
'''

import glob
import json
import os
import re
import urllib.request

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.time import Time
from astropy.timeseries import LombScargle
from astropy.wcs import WCS
from matplotlib.ticker import MaxNLocator
from scipy import fft
from scipy.optimize import curve_fit

TIC = 230384389
NAME = 'TIC230384389'
CACHE = 'tess_cache'
OUT = 'output'
S3 = 'https://stpubdata.s3.amazonaws.com'   # STScI's copy of the MAST TESS archive

# Every sector with 120-s data as of October 2026, one line per observing block
SECTORS = [14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
           40, 41, 47, 48, 49, 50, 51, 52, 53, 54, 56, 57, 58, 59, 60,
           73, 74, 76, 78, 79, 80, 81, 83, 84, 85, 86]
BLOCKS = {'2019-20': range(14, 27), '2021-22': range(40, 61), '2023-24': range(73, 87)}
PIXEL_SECTORS = [20, 47, 48, 49, 50, 51, 52, 53, 57, 74, 81]   # sectors whose target pixel files are checked

ROMERO2022 = [457.17, 493.86, 707.92, 749.61, 1284.58, 1633.57]   # periods (s), Romero et al. (2022), sectors 14-26
ROMERO2025 = [749.6259, 753.0697, 756.8263]   # periods near 750 s, Romero et al. (2025), sectors 14-60

CADENCE = 120.0          # s
TREND_WINDOW = 0.5       # d, running median divided out of each sector
CLIP_SIGMA = 5.0
FAP = 1e-3               # false-alarm probability of the significance threshold
N_SHUFFLE = 10000        # shuffled copies per sector
N_SHUFFLE_BLOCK = 1000   # shuffled copies per block
SEED = 230384389
OVERSAMPLE = 10
SEARCH_BAND = (300.0, 1e6 / (2 * CADENCE))   # uHz, 300 uHz to Nyquist
NOISE_BAND = (2500.0, 4100.0)                # uHz, no pulsations here
HALF_WIDTH = 10.0        # uHz, each signal is measured within +-10 uHz of its center
TRACK_STEP = 8.0         # uHz, how far a center can follow the peaks from one sector to the next
VIEW_HALF_WIDTH = 35.0   # uHz, spectra around each signal in Figure 1
BAND_750 = (1290.0, 1345.0)   # uHz, searched peak by peak in every sector and orbit
PIXEL_SIZE = 21.0        # arcsec

FIG_WIDTH = 6.9   # in, 0.97 of the AASTeX text width, so the font sizes are the printed sizes
FIGURE_RC = {
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'Tinos', 'Liberation Serif', 'Nimbus Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 10,
    'axes.titlesize': 10,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'legend.title_fontsize': 9,
    'legend.frameon': False,
    'axes.titlepad': 3.5,
    'axes.labelpad': 2.5,
    'axes.linewidth': 0.6,
    'lines.linewidth': 0.9,
    'lines.markeredgewidth': 0.6,
    'patch.linewidth': 0.6,
    'axes.edgecolor': '.25',
    'axes.labelcolor': '.15',
    'text.color': '.15',
    'xtick.color': '.25',
    'ytick.color': '.25',
    'xtick.labelcolor': '.15',
    'ytick.labelcolor': '.15',
    'xtick.major.size': 2.5,
    'ytick.major.size': 2.5,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.major.pad': 2.5,
    'ytick.major.pad': 2.5,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02,
    'pdf.fonttype': 42,   # embed TrueType fonts (AAS)
    'ps.fonttype': 42,
}
BLOCK_COLORS = ['#332288', '#44AA99', '#CC6677']   # Paul Tol's muted palette
BLOCK_MARKERS = ['o', 's', 'D']                    # so the blocks stay distinct in grayscale
INK = '#262B33'
BAND_COLOR = '#EE7733'
EDGE_COLOR = '#0077BB'
LIMIT_COLOR = '0.55'
TICK_SECTORS = [14, 20, 41, 50, 57, 76, 83]
AREA_PER_PPT = 3.0   # symbol area (pt^2) per ppt in panel (g)

def list_files():
    '''Light-curve files of every sector, listed from the archive the first time and then read from the cache.'''
    path = os.path.join(CACHE, 'files.json')
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)

    tid = f'{TIC:016d}'
    keys = []
    for s in SECTORS:
        prefix = f'tess/public/tid/s{s:04d}/{tid[:4]}/{tid[4:8]}/{tid[8:12]}/{tid[12:]}/'
        with urllib.request.urlopen(f'{S3}/?list-type=2&prefix={prefix}', timeout=60) as r:
            keys += re.findall(r'<Key>([^<]*-s_lc\.fits)</Key>', r.read().decode())
    with open(path, 'w') as f:
        json.dump(keys, f, indent=0)
    return keys

def download(key):
    '''Download one file into the cache, unless it is already there.'''
    path = os.path.join(CACHE, os.path.basename(key))
    if not os.path.exists(path):
        print(f'Downloading {os.path.basename(key)}')
        urllib.request.urlretrieve(f'{S3}/{key}', path + '.part')
        os.replace(path + '.part', path)
    return path

def running_median(t, x):
    '''Running median over 0.5 d (361 cadences) on the regular 120-s grid, with the gaps left empty.'''
    k = np.round((t - t[0]) * 86400 / CADENCE).astype(int)
    n = int(TREND_WINDOW * 86400 / CADENCE) + 1
    grid = np.full(k[-1] + 1, np.nan)
    grid[k] = x
    return pd.Series(grid).rolling(n, center=True, min_periods=n // 4).median().to_numpy()[k]

def read_sector(path):
    '''Light curve of one sector in ppt: PDCSAP flux with QUALITY = 0, divided by the running median.'''
    with fits.open(path) as hdu:
        data = hdu[1].data
        head = hdu[0].header
        good = (data['QUALITY'] == 0) & np.isfinite(data['PDCSAP_FLUX']) & np.isfinite(data['TIME'])
        t = np.asarray(data['TIME'][good], float)
        flux = np.asarray(data['PDCSAP_FLUX'][good], float)
        info = {
            'sector': head['SECTOR'],
            'file': os.path.basename(path),
            'camera': head['CAMERA'],
            'ccd': head['CCD'],
            'crowdsap': hdu[1].header['CROWDSAP'],
            'tessmag': head['TESSMAG'],
        }

    ppt = 1e3 * (flux / running_median(t, flux) - 1)

    # Clip at 5 sigma, with sigma from the median absolute deviation, until nothing changes
    keep = np.isfinite(ppt)
    for _ in range(20):
        med = np.median(ppt[keep])
        mad = np.median(np.abs(ppt[keep] - med))
        new_keep = np.isfinite(ppt) & (np.abs(ppt - med) < CLIP_SIGMA * 1.4826 * mad)
        if (new_keep == keep).all():
            break
        keep = new_keep

    t = t[keep]
    y = ppt[keep] - ppt[keep].mean()
    info['n_points'] = len(y)
    info['btjd_start'] = t[0]
    info['btjd_end'] = t[-1]
    info['rms_ppt'] = float(np.std(y))
    return pd.DataFrame({'sector': info['sector'], 'btjd': t, 'ppt': y}), info

def load_data():
    '''All light curves in one table, and one row of information per sector.'''
    os.makedirs(CACHE, exist_ok=True)
    keys = list_files()
    for s in PIXEL_SECTORS:
        key = next(k for k in keys if f'-s{s:04d}-' in k)
        download(key.replace('-s_lc.fits', '-s_tp.fits'))

    curves, info = [], []
    for key in keys:
        curve, row = read_sector(download(key))
        curves.append(curve)
        info.append(row)
    return pd.concat(curves, ignore_index=True), pd.DataFrame(info)

def series(curves):
    '''Times in seconds and fluxes in ppt.'''
    return curves.btjd.to_numpy() * 86400, curves.ppt.to_numpy()

def amp_spectrum(t, y, f):
    '''Lomb-Scargle amplitude spectrum (ppt) at frequencies f (uHz); a sinusoid of amplitude A gives a peak of height A.'''
    power = LombScargle(t, y, fit_mean=False, center_data=False).power(np.asarray(f) * 1e-6, method='fast', normalization='psd')
    return np.sqrt(4 * power / len(y))

def significance_threshold(t, y, rng, n_shuffle):
    '''Amplitude that the highest peak at 300 uHz to Nyquist exceeds in only 0.1% of flux-shuffled copies.'''
    # The data are placed on the regular 120-s grid so that each copy needs only one FFT.
    k = np.round((t - t.min()) / CADENCE).astype(int)
    m = int(2 ** np.ceil(np.log2((k.max() + 1) * OVERSAMPLE)))
    f = fft.rfftfreq(m, CADENCE) * 1e6
    band = (f >= SEARCH_BAND[0]) & (f <= SEARCH_BAND[1])
    grid = np.zeros(m)
    highest = np.empty(n_shuffle)
    for i in range(n_shuffle):
        grid[k] = rng.permutation(y)
        highest[i] = np.abs(fft.rfft(grid, workers=-1)[band]).max()
    return 2 / len(y) * np.quantile(highest, 1 - FAP)

def fit_sine(t, y, f0):
    '''Least-squares sinusoid starting at f0 (uHz). Returns its frequency and error (uHz), amplitude and error (ppt).'''
    tc = t - t.mean()

    def model(t, f, a, b):
        w = 2e-6 * np.pi * f
        return a * np.sin(w * t) + b * np.cos(w * t)

    w0 = 2e-6 * np.pi * f0
    a0, b0 = np.linalg.lstsq(np.column_stack([np.sin(w0 * tc), np.cos(w0 * tc)]), y, rcond=None)[0]
    (f, a, b), cov = curve_fit(model, tc, y, p0=[f0, a0, b0])
    amp = np.hypot(a, b)
    g = np.array([a, b]) / amp
    return f, np.sqrt(cov[0, 0]), amp, np.sqrt(g @ cov[1:, 1:] @ g)

def band_power(t, y, center):
    '''Power within +-10 uHz of center minus the noise, as the squared amplitude A^2 (ppt^2) of one sinusoid.'''
    # The noise and the scatter come from 80 bands of the same width at 2500-4100 uHz.
    df = 1e6 / np.ptp(t)
    n = int(2 * HALF_WIDTH / df)
    offsets = (np.arange(n) - (n - 1) / 2) * df
    window = np.sum(amp_spectrum(t, np.cos(2e-6 * np.pi * center * t), center + offsets) ** 2)   # a unit sinusoid

    n_ref = int((NOISE_BAND[1] - NOISE_BAND[0]) // (n * df))
    ref = amp_spectrum(t, y, NOISE_BAND[0] + np.arange(n_ref * n) * df).reshape(n_ref, n) ** 2
    noise = np.median(ref) / np.log(2)   # mean noise power per frequency bin
    ref_a2 = (ref.sum(axis=1) - n * noise) / window
    scatter = 1.4826 * np.median(np.abs(ref_a2 - np.median(ref_a2)))

    a2 = (np.sum(amp_spectrum(t, y, center + offsets) ** 2) - n * noise) / window
    a2_err = np.sqrt(scatter ** 2 + 2 * noise * max(a2, 0.0) / window)   # adds the signal-noise cross term
    return a2, a2_err, noise

def significant_peaks(f, a, threshold):
    '''Peaks above the threshold, highest first, at least 1.5 uHz apart (this drops the side lobes from the orbit gap).'''
    above = np.flatnonzero(a >= threshold)
    kept = []
    for i in above[np.argsort(a[above])[::-1]]:
        if all(abs(f[i] - f[j]) >= 1.5 for j in kept):
            kept.append(i)
    return kept

def track(t, y, period, center, threshold):
    '''Moves the band center to the A^2-weighted mean frequency of the significant peaks within 8 uHz of it.'''
    f_ref = 1e6 / period
    f = np.arange(f_ref - VIEW_HALF_WIDTH, f_ref + VIEW_HALF_WIDTH, 1e6 / np.ptp(t) / OVERSAMPLE)
    a = amp_spectrum(t, y, f)
    near = [i for i in significant_peaks(f, a, threshold) if abs(f[i] - center) <= TRACK_STEP]
    if not near:
        return center
    w = a[near] ** 2
    return float(np.sum(w * f[near]) / w.sum())

def measure(t, y, period, center, threshold, label):
    '''One signal in one sector or block: the highest peak within +-10 uHz of center, and the power in that band.'''
    f_ref = 1e6 / period
    grid = np.arange(center - HALF_WIDTH, center + HALF_WIDTH, 1e6 / np.ptp(t) / OVERSAMPLE)
    f, f_err, amp, amp_err = fit_sine(t, y, grid[np.argmax(amp_spectrum(t, y, grid))])
    a2, a2_err, noise = band_power(t, y, center)
    return {
        'sector_or_block': label,
        'source': 'Romero2022' if period in ROMERO2022 else 'new',
        'ref_period_s': period,
        'threshold_ppt': threshold,
        'mean_noise_ppt': np.sqrt(np.pi * noise) / 2,
        'track_freq_uHz': center,
        'track_offset_uHz': center - f_ref,
        'peak_freq_uHz': f,
        'peak_freq_err_uHz': f_err,
        'peak_period_s': 1e6 / f,
        'peak_amp_ppt': amp,
        'peak_amp_err_ppt': amp_err,
        'band_amp_ppt': np.sqrt(max(a2, 0.0)),
        'band_amp_err_ppt': a2_err / (2 * np.sqrt(a2)) if a2 > 0 else np.nan,
        'band_amp_3sigma_upper_ppt': np.sqrt(max(a2, 0.0) + 3 * a2_err),
        'band_detected': bool(a2 >= 3 * a2_err),
    }

def other_peaks(t, y, label, threshold, centers):
    '''Significant peaks of a block spectrum outside the bands of the six signals.'''
    f = np.arange(*SEARCH_BAND, 1e6 / np.ptp(t) / OVERSAMPLE)
    a = amp_spectrum(t, y, f)
    outside = np.all([np.abs(f - c) > HALF_WIDTH for c in centers.values()], axis=0)
    peaks = significant_peaks(f, np.where(outside, a, 0.0), threshold)
    return [{'block': label, 'freq_uHz': f[i], 'period_s': 1e6 / f[i], 'amp_ppt': a[i]} for i in peaks]

def prewhiten(t, y, threshold):
    '''Fits and subtracts the highest peak near 750 s, up to eight times, while it is above the threshold.'''
    f = np.arange(*BAND_750, 1e6 / np.ptp(t) / OVERSAMPLE)
    residual = y.copy()
    tc = t - t.mean()
    fits_found = []
    for _ in range(8):
        a = amp_spectrum(t, residual, f)
        if a.max() < threshold:
            break
        fit = fit_sine(t, residual, f[np.argmax(a)])
        w = 2e-6 * np.pi * fit[0]
        x = np.column_stack([np.sin(w * tc), np.cos(w * tc)])
        residual = residual - x @ np.linalg.lstsq(x, residual, rcond=None)[0]
        fits_found.append(fit)
    return fits_found

def peaks_750(curves, thresholds):
    '''Every significant peak near 750 s in each sector, and in each orbit (half sector) longer than 8 days.'''
    # An orbit has half the points, so its threshold is sqrt(2) times the sector threshold.
    rows = []
    for s, seg in curves.groupby('sector'):
        t, y = series(seg)
        btjd = seg.btjd.to_numpy()
        mid = 0.5 * (t.min() + t.max())
        parts = [('sector', np.ones(len(t), bool), thresholds[s])]
        for name, m in [('orbit1', t < mid), ('orbit2', t >= mid)]:
            if np.ptp(t[m]) > 8 * 86400:
                parts.append((name, m, np.sqrt(2) * thresholds[s]))

        for part, m, threshold in parts:
            for f, f_err, a, a_err in prewhiten(t[m], y[m] - y[m].mean(), threshold):
                rows.append({'sectors': s, 'part': part, 'btjd_mid': btjd[m].mean(), 'threshold_ppt': threshold,
                             'freq_uHz': f, 'freq_err_uHz': f_err, 'period_s': 1e6 / f, 'amp_ppt': a, 'amp_err_ppt': a_err})
    return pd.DataFrame(rows)

def combined_750(curves):
    '''The three highest peaks near 750 s in sectors 14-60 combined, the data of Romero et al. (2025).'''
    t, y = series(curves[curves.sector.isin(range(14, 61))])
    f = np.arange(*BAND_750, 1e6 / np.ptp(t) / OVERSAMPLE)
    a = amp_spectrum(t, y, f)
    top = significant_peaks(f, a, 0.0)[:3]
    return pd.DataFrame({'sectors': '14-60', 'part': 'combined', 'freq_uHz': f[top], 'period_s': 1e6 / f[top], 'amp_ppt': a[top]})

def pixel_check(sector_rows):
    '''Band power of each signal in each pixel of the 5x5 box around the target, in each target pixel file.'''
    rows = []
    paths = sorted(glob.glob(os.path.join(CACHE, '*-s_tp.fits')))
    for i, path in enumerate(paths, 1):
        with fits.open(path) as hdu:
            data = hdu[1].data
            good = (data['QUALITY'] == 0) & np.isfinite(data['FLUX']).all(axis=(1, 2))
            t = np.asarray(data['TIME'][good], float)
            cube = np.asarray(data['FLUX'][good], float)
            col, row = WCS(hdu[2].header).all_world2pix(hdu[0].header['RA_OBJ'], hdu[0].header['DEC_OBJ'], 0)
            sector = hdu[0].header['SECTOR']

        # Cut the 5x5 box; (row, col) is then the target position in the box
        r0 = int(round(float(row))) - 2
        c0 = int(round(float(col))) - 2
        cube = cube[:, r0:r0 + 5, c0:c0 + 5]
        row = float(row) - r0
        col = float(col) - c0

        here = sector_rows[sector_rows.sector_or_block == str(sector)]
        for period, center in zip(here.ref_period_s, here.track_freq_uHz):
            power = np.zeros((5, 5))
            snr = np.zeros((5, 5))
            for r in range(5):
                for c in range(5):
                    y = cube[:, r, c] - running_median(t, cube[:, r, c])
                    y = np.nan_to_num(y - np.nanmean(y))
                    power[r, c], err, _ = band_power(t * 86400, y, center)
                    snr[r, c] = power[r, c] / err

            detected = snr >= 3
            best = np.unravel_index(np.argmax(snr), snr.shape)
            if detected.any():
                w = np.where(detected, power, 0.0)
                rr, cc = np.indices(snr.shape)
                centroid = (np.sum(w * rr) / w.sum(), np.sum(w * cc) / w.sum())
            else:
                centroid = (np.nan, np.nan)
            rows.append({
                'sector': sector,
                'ref_period_s': period,
                'n_pixels_3sigma': int(detected.sum()),
                'max_snr': snr.max(),
                'max_snr_pixel_offset_arcsec': PIXEL_SIZE * np.hypot(best[0] - row, best[1] - col),
                'power_centroid_offset_arcsec': PIXEL_SIZE * np.hypot(centroid[0] - row, centroid[1] - col),
            })
        print(f'Target pixel file, sector {sector} ({i}/{len(paths)})')
    return pd.DataFrame(rows)

def table_path(name):
    '''Path of one of the five tables in output/.'''
    return os.path.join(OUT, f'{NAME}_{name}.csv')

def set_sector_axis(ax, order, ticks, labels):
    '''x-axis with one position per sector.'''
    ax.set_xlim(-0.6, len(order) - 0.4)
    tick_labels = [str(order[i]) for i in ticks] if labels else []
    ax.set_xticks(ticks, tick_labels)

def label_panel(ax, letter, text, right=None):
    '''Panel letter and title on the left, and an optional note on the right.'''
    ax.set_title(f'$\\mathbf{{({letter})}}$  {text}', loc='left')
    if right:
        ax.set_title(right, loc='right', fontsize=9, color='0.35')

def to_period(f):
    '''uHz to s, and s to uHz (the same formula), for the period axis of panel (g).'''
    return 1e6 / np.maximum(f, 1e-9)

def make_figure(curves):
    '''Figure 1, drawn from the tables in output/. Also writes the spectra of panels (a-f) to spectra.csv.'''
    signals = pd.read_csv(table_path('signals'))
    sectors = signals[signals.sector_or_block.str.isdigit()].copy()
    sectors['sector'] = sectors.sector_or_block.astype(int)
    peaks = pd.read_csv(table_path('750s_peaks'))
    orbits = peaks[peaks.part.isin(['orbit1', 'orbit2'])].copy()
    orbits['sector'] = orbits.sectors.astype(int)

    # The 39 sectors are plotted side by side, with a line between the blocks
    order = sorted(sectors.sector.unique())
    pos = {s: i for i, s in enumerate(order)}
    x = np.arange(len(order))
    block_of = {}
    for k, secs in enumerate(BLOCKS.values()):
        for s in secs:
            block_of[s] = k
    block = np.array([block_of[s] for s in order])
    edges = [i - 0.5 for i in range(1, len(order)) if block[i] != block[i - 1]]
    ticks = [pos[s] for s in TICK_SECTORS]

    times = {s: series(seg) for s, seg in curves.groupby('sector')}
    new = sectors[sectors.source == 'new'].ref_period_s.iloc[0]
    panel_periods = [new if p == 749.61 else p for p in ROMERO2022]   # 749.61 s is shown orbit by orbit in (g)
    offsets = np.linspace(-VIEW_HALF_WIDTH, VIEW_HALF_WIDTH, 701)
    point_style = dict(ms=3.8, mec='white', mew=0.4)
    limit_style = dict(ms=3.8, mfc='none', mec=LIMIT_COLOR, mew=0.6, ls='none')
    spectra = []

    with plt.rc_context(FIGURE_RC):
        fig = plt.figure(figsize=(FIG_WIDTH, 8.3), layout='constrained')
        fig.get_layout_engine().set(h_pad=0.025, w_pad=0.04, hspace=0.0, wspace=0.06)
        grid = fig.add_gridspec(6, 3, height_ratios=[0.1, 1, 0.09, 1, 0.09, 0.85])   # row 0: key, rows 2 and 4: space
        cells = [grid[1, 0], grid[1, 1], grid[1, 2], grid[3, 0], grid[3, 1], grid[3, 2]]

        # (a-f) Each signal: the spectrum of every sector in units of its mean noise, with the measured band (upper),
        # and the band amplitude (lower)
        for period, cell, letter in zip(panel_periods, cells, 'abcdef'):
            sub = cell.subgridspec(2, 1, height_ratios=[1.2, 1])
            ax_s = fig.add_subplot(sub[0])
            ax_a = fig.add_subplot(sub[1])
            d = sectors[np.isclose(sectors.ref_period_s, period)].set_index('sector').loc[order]
            f_ref = 1e6 / period
            amps = np.array([amp_spectrum(*times[s], f_ref + offsets) for s in order])
            for s, a in zip(order, amps):
                spectra.append(pd.DataFrame({'panel': letter, 'ref_period_s': period, 'sector': s, 'freq_uHz': f_ref + offsets,
                                             'amp_ppt': a, 'amp_over_mean_noise': a / d.mean_noise_ppt[s]}))

            image = (amps / d.mean_noise_ppt.to_numpy()[:, None]).T
            im = ax_s.imshow(image, origin='lower', aspect='auto', cmap='Greys', vmin=0, vmax=8, interpolation='nearest',
                             extent=[-0.5, len(order) - 0.5, offsets[0], offsets[-1]])
            for edge in (-HALF_WIDTH, HALF_WIDTH):
                ax_s.step(x, d.track_offset_uHz + edge, where='mid', color=BAND_COLOR, lw=0.9)

            detected = d.band_detected.to_numpy()
            for k in range(len(BLOCKS)):
                m = (block == k) & detected
                ax_a.errorbar(x[m], d.band_amp_ppt[m], d.band_amp_err_ppt[m], fmt=BLOCK_MARKERS[k], color=BLOCK_COLORS[k],
                              elinewidth=0.7, capsize=0, zorder=3, **point_style)
            ax_a.plot(x[~detected], d.band_amp_3sigma_upper_ppt[~detected], 'v', zorder=2, **limit_style)

            for e in edges:
                ax_s.axvline(e, color=EDGE_COLOR, lw=0.7)
                ax_a.axvline(e, color=EDGE_COLOR, lw=0.6, alpha=0.5, zorder=0)
            set_sector_axis(ax_s, order, ticks, labels=False)
            set_sector_axis(ax_a, order, ticks, labels=True)
            ax_s.set_yticks([-25, 0, 25])
            top = np.where(detected, d.band_amp_ppt + d.band_amp_err_ppt, d.band_amp_3sigma_upper_ppt)
            ax_a.set_ylim(0, 1.2 * np.nanmax(top))
            ax_a.yaxis.set_major_locator(MaxNLocator(nbins=3, steps=[1, 2, 5, 10], prune='upper'))
            ax_a.grid(True, axis='y', color='0.9', lw=0.4, zorder=0)
            ax_a.set_axisbelow(True)
            name = f'{period:.2f} s, new' if period == new else f'{period:.2f} s'
            label_panel(ax_s, letter, name, f'{f_ref:.1f} $\\mu$Hz')
            if letter in 'ad':
                ax_s.set_ylabel('$f - f_{\\rm ref}$ ($\\mu$Hz)')
                ax_a.set_ylabel('band amp. (ppt)')

        # (g) Every significant peak near 750 s, each TESS orbit searched on its own
        ax = fig.add_subplot(grid[5, :])
        xo = orbits.sector.map(pos) + np.where(orbits.part == 'orbit1', -0.2, 0.2)
        for p in ROMERO2025:
            ax.axhline(1e6 / p, color=INK, lw=0.7, ls=(0, (4, 3)), alpha=0.6, zorder=1)
        for k in range(len(BLOCKS)):
            m = (orbits.sector.map(block_of) == k).to_numpy()
            ax.scatter(xo[m], orbits.freq_uHz[m], s=AREA_PER_PPT * orbits.amp_ppt[m], marker=BLOCK_MARKERS[k],
                       color=BLOCK_COLORS[k], edgecolor='white', linewidth=0.4, zorder=3)
        for e in edges:
            ax.axvline(e, color=EDGE_COLOR, lw=0.6, alpha=0.5, zorder=0)
        set_sector_axis(ax, order, ticks, labels=True)
        ax.set_ylim(*BAND_750)
        ax.set_xlabel('TESS sector (two points per sector, one per orbit)')
        ax.set_ylabel('frequency ($\\mu$Hz)')
        period_axis = ax.secondary_yaxis('right', functions=(to_period, to_period))
        period_axis.set_ylabel('period (s)')
        label_panel(ax, 'g', '749.61 s: every significant peak at 1290\u20131345 $\\mu$Hz, each TESS orbit searched on its own')
        sizes = [plt.Line2D([], [], ls='none', marker='o', ms=np.sqrt(AREA_PER_PPT * a), color='0.5', mec='white', mew=0.4)
                 for a in (4, 8, 16)]
        ax.legend(sizes, ['4 ppt', '8 ppt', '16 ppt'], loc='lower left', ncol=3, title='peak amplitude', alignment='left',
                  handletextpad=0.2, columnspacing=1.0, borderaxespad=0.5)
        ax.text(0.99, 1e6 / ROMERO2025[0] + 0.8, 'dashed: the three periods of Romero et al. (2025)', transform=ax.get_yaxis_transform(),
                ha='right', va='bottom', fontsize=9, zorder=4, bbox=dict(boxstyle='square,pad=0.15', fc='white', ec='none', alpha=0.85))

        # One key for all panels
        head = fig.add_subplot(grid[0, :])
        head.axis('off')
        handles = [plt.Line2D([], [], ls='none', marker=BLOCK_MARKERS[k], color=BLOCK_COLORS[k], **point_style) for k in range(len(BLOCKS))]
        handles.append(plt.Line2D([], [], marker='v', **limit_style))
        labels = ['2019\u201320', '2021\u201322', '2023\u201324', 'not detected: 3$\\sigma$ upper limit']
        head.legend(handles, labels, loc='center left', bbox_to_anchor=(0, 0.45), ncol=4, handletextpad=0.2, columnspacing=1.2,
                    markerscale=1.3, borderaxespad=0)
        cax = head.inset_axes([0.81, 0.1, 0.19, 0.28])
        bar = fig.colorbar(im, cax=cax, orientation='horizontal', ticks=[0, 4, 8])
        bar.outline.set_linewidth(0.5)
        cax.tick_params(which='both', top=False, length=2)
        cax.set_title('amplitude / mean noise', fontsize=9, pad=2)

        # Run the layout once, then keep it fixed while saving with a tight bounding box
        fig.canvas.draw()
        fig.set_layout_engine('none')
        fig.savefig(os.path.join(OUT, 'Figure_1.pdf'), metadata={'CreationDate': None})
        plt.close(fig)

    spectra = pd.concat(spectra).round({'freq_uHz': 3, 'amp_ppt': 4, 'amp_over_mean_noise': 3})
    spectra.to_csv(table_path('spectra'), index=False)

def print_summary():
    '''Key results, from the tables as written.'''
    signals = pd.read_csv(table_path('signals'))
    peaks = pd.read_csv(table_path('750s_peaks'))
    is_sector = signals.sector_or_block.str.isdigit()
    sectors = signals[is_sector].copy()
    sectors['sector'] = sectors.sector_or_block.astype(int)
    blocks = signals[~is_sector]

    # Strongest peak near 750 s in each orbit of sectors 47-53, in time order
    orbits = peaks[peaks.part.isin(['orbit1', 'orbit2'])].copy()
    orbits['sector'] = orbits.sectors.astype(int)
    strongest = orbits.sort_values('amp_ppt').groupby(['sector', 'part']).tail(1).sort_values('btjd_mid')
    drift = strongest[strongest.sector.between(47, 53)]

    # Count the frequencies more than 1 uHz apart, going up from the lowest
    n_apart = 0
    previous = -np.inf
    for f in np.sort(drift.freq_uHz):
        if f > previous + 1.0:
            n_apart += 1
            previous = f

    first, last = drift.iloc[0], drift.iloc[-1]
    print('750 s signal, sectors 47-53: {p0:.2f} -> {p1:.2f} s ({rate:.1f} s/yr), {n} frequencies more than 1 uHz apart'.format(
        p0=first.period_s, p1=last.period_s, rate=(last.period_s - first.period_s) / ((last.btjd_mid - first.btjd_mid) / 365.25), n=n_apart))

    print('Sectors 14-60 combined: ' + ', '.join('{p:g} s ({a:.2f} ppt)'.format(p=r.period_s, a=r.amp_ppt) for r in peaks[peaks.part == 'combined'].itertuples()))
    print('457.17 s signal: changed by at most {:.2f} s'.format(np.ptp(blocks[np.isclose(blocks.ref_period_s, 457.17)].peak_period_s)))

    detected = sectors[sectors.band_detected]
    for p in (1633.57, 1284.58, 707.92):
        print('{p:g} s signal: last detected in sector {s}'.format(p=p, s=detected[np.isclose(detected.ref_period_s, p)].sector.max()))
    print('New signal: first detected in sector {s}'.format(s=detected[detected.source == 'new'].sector.min()))

def main():
    '''Measures everything, then writes the tables, the figure and a summary.'''
    os.makedirs(OUT, exist_ok=True)
    curves, info = load_data()
    print(f'{len(info)} sectors, {len(curves):,} measurements')

    # Significance threshold of each sector and each block, from flux-shuffled copies
    rng = np.random.default_rng(SEED)
    thresholds = {}
    for i, (s, seg) in enumerate(curves.groupby('sector'), 1):
        t, y = series(seg)
        thresholds[s] = significance_threshold(t, y, rng, N_SHUFFLE)
        print(f'Sector {s} ({i}/{len(info)}): threshold {thresholds[s]:.2f} ppt')
    for name, secs in BLOCKS.items():
        t, y = series(curves[curves.sector.isin(secs)])
        thresholds[name] = significance_threshold(t, y, rng, N_SHUFFLE_BLOCK)
        print(f'Block {name}: threshold {thresholds[name]:.2f} ppt')

    # The six signals in each sector, following each one's frequency from sector to sector
    centers = {p: 1e6 / p for p in ROMERO2022}
    sector_rows = []
    for s, seg in curves.groupby('sector'):
        t, y = series(seg)
        for p in ROMERO2022:
            centers[p] = track(t, y, p, centers[p], thresholds[s])
            sector_rows.append(measure(t, y, p, centers[p], thresholds[s], str(s)))

    # The six signals in each block, at the median of their centers in its sectors (at the published
    # frequencies in 2019-20, the data of Romero et al. 2022), and any other significant peak
    tracked = pd.DataFrame(sector_rows)
    block_rows, others = [], []
    for name, secs in BLOCKS.items():
        t, y = series(curves[curves.sector.isin(secs)])
        if name == '2019-20':
            centers = {p: 1e6 / p for p in ROMERO2022}
        else:
            in_block = tracked[tracked.sector_or_block.astype(int).isin(secs)]
            centers = in_block.groupby('ref_period_s').track_freq_uHz.median().to_dict()
        for p in ROMERO2022:
            block_rows.append(measure(t, y, p, centers[p], thresholds[name], name))
        others += other_peaks(t, y, name, thresholds[name], centers)

    # The strongest other peak is a new signal, measured like the six
    new = max(others, key=lambda peak: peak['amp_ppt'])
    print('New signal: {period_s:.2f} s in the {block} block'.format(**new))
    new_period, new_freq = new['period_s'], new['freq_uHz']
    center = new_freq
    for s, seg in curves.groupby('sector'):
        t, y = series(seg)
        center = track(t, y, new_period, center, thresholds[s])
        sector_rows.append(measure(t, y, new_period, center, thresholds[s], str(s)))
    for name, secs in BLOCKS.items():
        t, y = series(curves[curves.sector.isin(secs)])
        block_rows.append(measure(t, y, new_period, new_freq, thresholds[name], name))

    signals = pd.DataFrame(sector_rows + block_rows)
    print('Finding the peaks near 750 s in each sector and orbit')
    peaks = pd.concat([peaks_750(curves, thresholds), combined_750(curves)])
    pixels = pixel_check(pd.DataFrame(sector_rows))
    info['threshold_ppt'] = info.sector.map(thresholds)
    info['year'] = Time(0.5 * (info.btjd_start + info.btjd_end) + 2457000, format='jd').decimalyear.round(3)

    print('Writing the tables and Figure 1')
    info.to_csv(table_path('sectors'), index=False, float_format='%.4f')
    signals.to_csv(table_path('signals'), index=False, float_format='%.5f')
    peaks.to_csv(table_path('750s_peaks'), index=False, float_format='%.4f')
    pixels.to_csv(table_path('pixels'), index=False, float_format='%.3f')
    make_figure(curves)
    print_summary()
    print(f'Wrote {OUT}/')

if __name__ == '__main__':
    main()