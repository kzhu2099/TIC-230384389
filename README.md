# TIC 230384389 pulsation periods

Code for *Temporal Changes in Pulsation Periods in the ZZ Ceti Variable TIC 230384389*, to be submitted to the Research Notes of the AAS.

The results (five tables and the figure) are on Zenodo: [doi:10.5281/zenodo.23439428](https://doi.org/10.5281/zenodo.23439428).

`main.py` downloads the TESS SPOC 120-s light curves of the ZZ Ceti star TIC 230384389 (39 sectors, 2019-2024) and the target pixel files of 11 of those sectors. It measures the six pulsation periods of Romero et al. (2022) and a new signal in each sector and in three observing blocks, finds every significant peak near 750 s in each sector and TESS orbit, and checks in the pixel data where each signal's power comes from.

## Running

```
pip install -r requirements.txt
python main.py
```

The first run downloads about 580 MB from STScI's copy of the MAST TESS archive on AWS into `tess_cache/`, so it needs a network connection; later runs use the cache. A run takes about 30 minutes, most of it spent on the 10,000 flux-shuffled copies of each sector that set the significance thresholds. Tested with Python 3.12 and the versions pinned in `requirements.txt`.

The script writes these files to `output/`, the same ones as in the Zenodo deposit:

- `Figure_1.pdf`: Figure 1 of the note
- `TIC230384389_signals.csv`: each signal measured in each sector and in each block
- `TIC230384389_750s_peaks.csv`: every significant peak near 750 s in each sector and orbit, and the three highest peaks there in sectors 14-60 combined
- `TIC230384389_spectra.csv`: the amplitude spectra shown in Figure 1
- `TIC230384389_sectors.csv`: the sectors used, with the input file, dates, noise and significance threshold of each
- `TIC230384389_pixels.csv`: where each signal's power lies in the target pixel files

It prints its progress as it runs, and then the key results:

```
750 s signal, sectors 47-53: 752.29 -> 760.88 s (17.8 s/yr), 10 frequencies more than 1 uHz apart
Sectors 14-60 combined: 753.07 s (0.83 ppt), 749.623 s (0.73 ppt), 756.81 s (0.72 ppt)
457.17 s signal: changed by at most 0.14 s
1633.57 s signal: last detected in sector 25
1284.58 s signal: last detected in sector 41
707.92 s signal: last detected in sector 47
New signal: first detected in sector 56
```

## License

The code is under the MIT license. The data products are on Zenodo under CC BY 4.0.
