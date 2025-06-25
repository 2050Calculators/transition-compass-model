from statsmodels.robust.scale import mad
import numpy as np

# fix jumps
def fix_jumps(ts, mad_multiple=3, consec_do_nothing=False, consec_fill_with_nan=False):
    # MAD is median absolute deviation
    # mad_multiple sets the number of deviations beyond which a jump is detected. 3 is default.
    # If there is a nan in the series, the function will not be able to do the correction
    # and it will return the series as it is.
    # consec_do_nothing = True: if there are consecutive jumps, do nothing
    # consec_do_nothing = True and consec_fill_with_nan = True: if there are consecutive jumps, do nothing and susbtitute with nan

    # force float
    ts = ts.astype(float)

    if (not all(np.isnan(ts))) and (not all(ts == 0)):

        # drop zeroes before first non zero
        first_nonzero = np.where(ts != 0)[0][0]
        zeroes = np.where(ts == 0)[0]
        drop = list(zeroes[zeroes < first_nonzero])
        index = list(range(0, len(ts)))
        ts_zero = ts[[i in drop for i in index]]
        ts_nonzero = ts[[i not in drop for i in index]]

        # drop zeroes after last non zero
        last_nonzero = np.where(ts_nonzero != 0)[0]
        last_nonzero = last_nonzero[len(last_nonzero) - 1]
        index = list(range(0, len(ts_nonzero)))
        ts_zeroend = ts_nonzero[index > last_nonzero]
        ts_nonzero = ts_nonzero[index <= last_nonzero]

        # get deviation from median
        deviation_from_median = np.abs(ts_nonzero - np.nanmedian(ts_nonzero))
        mad_threshold = mad_multiple * mad(ts_nonzero)  # Use a threshold multiple of MAD
        jumps = np.where(deviation_from_median > mad_threshold)[0]

        # correct jumps with mean between before and after
        if len(jumps) > 0:

            # correct
            corrected_ts = ts_nonzero.copy()
            for i in range(0, len(jumps)):

                # if it's the first position
                if jumps[i] == 0:
                    if len(jumps) > 1:
                        # if there is a consecutive jump after
                        if jumps[i + 1] == jumps[i] + 1 and consec_do_nothing == True:
                            # either do nothing
                            if consec_do_nothing == True and consec_fill_with_nan == False:
                                corrected_ts[jumps[i]] = corrected_ts[jumps[i]]
                            # or fill with nan
                            if consec_do_nothing == True and consec_fill_with_nan == True:
                                corrected_ts[jumps[i]] = np.nan
                        # otherwise just fill with the following number
                        else:
                            corrected_ts[jumps[i]] = corrected_ts[jumps[i] + 1]
                    else:
                        corrected_ts[jumps[i]] = corrected_ts[jumps[i] + 1]

                # if it's the last position
                elif jumps[i] + 1 == len(corrected_ts):
                    if len(jumps) > 1:
                        # if there is a consecutive jump before
                        if jumps[i] - 1 == jumps[i - 1] and consec_do_nothing == True:
                            # either do nothing
                            if consec_do_nothing == True and consec_fill_with_nan == False:
                                corrected_ts[jumps[i]] = corrected_ts[jumps[i]]
                            # or fill with nan
                            if consec_do_nothing == True and consec_fill_with_nan == True:
                                corrected_ts[jumps[i]] = np.nan
                        # otherwise just fill with the previous number
                        else:
                            corrected_ts[jumps[i]] = corrected_ts[jumps[i] - 1]
                    else:
                        corrected_ts[jumps[i]] = corrected_ts[jumps[i] - 1]

                # if it's in the middle of the series
                else:
                    if len(jumps) > 1:
                        # if there is a consecutive jump before or after
                        if i + 1 == len(jumps):
                            condition = (jumps[i] - 1 == jumps[i - 1]) and consec_do_nothing == True
                        else:
                            condition = (jumps[i] - 1 == jumps[i - 1] or jumps[i] + 1 == jumps[
                                i + 1]) and consec_do_nothing == True
                        if condition:
                            # either do nothing:
                            if consec_do_nothing == True and consec_fill_with_nan == False:
                                corrected_ts[jumps[i]] = corrected_ts[jumps[i]]
                            # or fill with nan
                            if consec_do_nothing == True and consec_fill_with_nan == True:
                                corrected_ts[jumps[i]] = np.nan
                        # otherwise just fill with mean
                        else:
                            corrected_ts[jumps[i]] = (corrected_ts[jumps[i] - 1] + corrected_ts[jumps[i] + 1]) / 2
                    else:
                        corrected_ts[jumps[i]] = (corrected_ts[jumps[i] - 1] + corrected_ts[jumps[i] + 1]) / 2
        else:
            corrected_ts = ts_nonzero

        # remake ts
        ts_new = np.append(ts_zero, corrected_ts)
        ts_new = np.append(ts_new, ts_zeroend)

    else:
        ts_new = ts

    return ts_new


def fix_jumps_in_dm(dm, mad_multiple=3, consec_do_nothing=False, consec_fill_with_nan=False):
    # flatten
    if len(dm.dim_labels) == 3:
        dm_temp = dm.copy()
    if len(dm.dim_labels) == 4:
        dm_temp = dm.flatten()
    if len(dm.dim_labels) == 5:
        dm_temp = dm.flatten().flatten()
    if len(dm.dim_labels) == 6:
        dm_temp = dm.flatten().flatten().flatten()

    # fix jumps
    idx = dm_temp.idx
    countries = dm_temp.col_labels["Country"]
    variabs = dm_temp.col_labels["Variables"]
    for c in countries:
        for v in variabs:
            ts = dm_temp.array[idx[c], :, idx[v]]
            dm_temp.array[idx[c], :, idx[v]] = fix_jumps(ts, mad_multiple=mad_multiple,
                                                         consec_do_nothing=consec_do_nothing,
                                                         consec_fill_with_nan=consec_fill_with_nan)

    # deepen
    if len(dm.dim_labels) == 4:
        dm_temp.deepen()
    if len(dm.dim_labels) == 5:
        dm_temp.deepen()
        dm_temp.deepen()
    if len(dm.dim_labels) == 6:
        dm_temp.deepen()
        dm_temp.deepen()
        dm_temp.deepen()

    return dm_temp
