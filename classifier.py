from __future__ import division

from collections import defaultdict
import numpy as np
from sklearn.metrics import roc_curve, auc
import joblib
import random
import scipy.weave
import pickle

import device_metadata
from device_metadata import device_ids
from similar_devices import devices_clusters

# Earth mover's distance between two numpy arrays.
# Requires the python-emd module.
def my_emd(a, b):
    import emd
    pos = range(len(a))
    return emd.emd((pos, list(a)), (pos, list(b)), lambda x,y: abs(x-y)+0.0)

# Kullback-Leibler divergence
def kl_div(p, q):
    # Kludge to prevent div-by-zero messages.  Doesn't affect the output.
    pp = np.where(p > 0, p, 1)
    qq = np.where(p > 0, q, 1)
    return np.sum(np.where(p > 0, pp * np.log(pp / qq), 0))

# Kullback-Leibler divergence (fast version)
def kl_div_fast(p, q):
    p = p.flatten()
    q = q.flatten()
    N = len(p)
    N # silence pyflakes
    return scipy.weave.inline('''
    double accum = 0;
    for(size_t i=0; i<N; i++) {
        if(p[i] > 0) accum += p[i] * logf(p[i] / q[i]);
    }
    return_val = accum;
    ''', ['p','q','N'])

def normalize_dist(p):
    assert np.min(p) >= 0
    assert np.sum(p) > 0 # FIXME
    return p / np.sum(p)

def approx_gcd(a,b):
    assert a > 0 and b > 0
    while True:
        if a < b:
            (a,b) = (b,a)
        a = a % b
        if a < 1e-4:
            return b

def get_resolution(x):
    #xm = np.median(x)
    #xg = x[(x > xm-2) & (x < xm+2)]
    xg = x.copy()
    xg -= np.min(xg)
    xg = xg[xg > 0]
    v = np.array([ approx_gcd(random.choice(xg), random.choice(xg)) for i in range(1000) ])
    v = v[np.isfinite(v)]
    m = np.median(v)
    #return np.mean(v[(v > m*0.8) & (v < m*1.2)])
    return m

def parse_timeseries(timeseries, long_return=False):
    (t, x, y, z) = timeseries[:,:4].T

    out = dict()

    eps = 1e-10 # prevent log(0)

    out['min_time'] = t[0]
    out['max_time'] = t[-1]

    # resolution of accelerometer sensor
    out['x_res'] = get_resolution(x)

    mag = np.sqrt(x*x+y*y+z*z)

    mean_mag = np.mean(mag)
    out['mean_mag'] = mean_mag

    median_mag = np.median(mag)
    out['median_mag'] = median_mag

    # Histogram of acceleration magnitude
    out['hg_mag'] = normalize_dist(np.histogram(np.clip(mag, 0, 30), bins=300, range=(0,30))[0])

    # Histogram of log deviation of acceleration magnitude from its median
    mag_dev = np.abs(mag-median_mag)
    out['hg_log_mag_dev'] = normalize_dist(
        np.histogram(np.clip(np.log(mag_dev+eps), -5, 5), bins=50, range=(-5,5))[0])

    # Histogram of orientation
    lon = np.arctan2(y, x)
    lat = np.arctan2(z, np.sqrt(x*x+y*y))
    out['hg_orientation'] = normalize_dist(np.histogram2d(
        lon, lat, bins=20, range=[[-np.pi, np.pi], [-np.pi/2, np.pi/2]])[0])

    # 3D histogram of acceleration
    out['hg_xyz'] = normalize_dist(np.histogramdd(
        np.clip(np.array([x,y,z]).T, -15, 15), bins=10, range=[[-15,15],[-15,15],[-15,15]])[0])

    # Calculate jerk.
    dt = t[1:] - t[:-1]
    dx = x[1:] - x[:-1]
    dy = y[1:] - y[:-1]
    dz = z[1:] - z[:-1]
    jerk = np.sqrt(dx*dx + dy*dy + dz*dz)
    dt_good = (dt > 0) & (dt < 1000) & (jerk > 0)
    # take only vals where dt is good, and divide by dt
    gdt = dt[dt_good]
    gdx = dx[dt_good] / gdt
    gdy = dy[dt_good] / gdt
    gdz = dz[dt_good] / gdt
    jerk = jerk[dt_good] / gdt
    # also take x,y,z for good dt
    gx = x[1:][dt_good]
    gy = y[1:][dt_good]
    gz = z[1:][dt_good]

    #print np.min(jerk), np.max(jerk)

    # Histogram of log of jerk magnitude
    out['hg_jerk'] = normalize_dist(np.histogram(
        np.clip(np.log(jerk+eps), -15, 4), bins=50, range=(-15,4))[0])

    # 3D histogram of jerk magnitude
    out['hg_jerk_xyz'] = normalize_dist(np.histogramdd(
        np.clip(np.array([gdx,gdy,gdz]).T, -15, 15), bins=10, range=[[-15,15],[-15,15],[-15,15]])[0])

    # Histogram of log of jerk magnitude in direction of acceleration
    jerk_dot_acc = gdx*gx + gdy*gy + gdz*gz
    #print np.min(jerk_dot_acc), np.max(jerk_dot_acc)
    out['hg_jerk_dot_acc'] = normalize_dist(np.histogram(
        np.clip(np.log(np.abs(jerk_dot_acc)+eps), -6, 6), bins=20, range=(-6,6))[0])

    out['hg_dt1'] = normalize_dist(np.histogram(
        np.clip(np.log(dt[dt > 0]), 0, 6), bins=100, range=(0,6))[0])

    out['hg_dt2'] = normalize_dist(np.histogram(
        np.clip(dt, 0, 2000), bins=range(2001))[0])

    time_of_day_bins = 24
    ms_per_day = 8.64e7
    time_of_day = np.array(t / ms_per_day * time_of_day_bins, dtype=int) % time_of_day_bins
    out['hg_time_of_day'] = normalize_dist(np.histogram(
        time_of_day, bins=time_of_day_bins, range=(0,time_of_day_bins))[0])

    if long_return:
        out['extras'] = {}
        for key in (
                't', 'x', 'y', 'z', 'mag', 'mag_dev',
                'lon', 'lat', 'dt', 'gdt', 'gdx', 'gdy', 'gdz', 'jerk_dot_acc'):
            out['extras'][key] = locals()[key]
    return out

def parse_from_file(fn, key, take_slice=None):
    if take_slice is None:
        take_slice = slice(None, None)
    return parse_timeseries(data_files[fn][key][take_slice,:])

# Compare a histogram (by Kullback-Leibler divergence) to the histograms of the training data.
def compare_histograms(hg_key, train_wiz, test):
    hw = [ w[hg_key] for w in train_wiz ]
    t = test[hg_key]

    n = 1e-6 # prevent infinities
    d = np.array([ kl_div_fast(t, w*(1-n)+t*n) for w in hw ])
    d /= np.mean(d)
    return d

def evaluate_question(test_wiz, proposed_dev, long_return=False):
    # Get list of devices to compare against
    if compare_only_similar:
        ids_to_check = [proposed_dev] + [ x for x in devices_clusters[proposed_dev] if x != proposed_dev ]
        pdev_idx = 0
    else:
        ids_to_check = device_ids
        pdev_idx = idx_for_dev[proposed_dev]

    train_wiz = [ train_wisdom[x] for x in ids_to_check ]

    def percentile(arr):
        return np.sum(arr > arr[pdev_idx]) / len(arr)

    dm = compare_histograms('hg_mag', train_wiz, test_wiz)
    dmdev = compare_histograms('hg_log_mag_dev', train_wiz, test_wiz)
    dj = compare_histograms('hg_jerk', train_wiz, test_wiz)
    dx = compare_histograms('hg_xyz', train_wiz, test_wiz)
    dh = compare_histograms('hg_time_of_day', train_wiz, test_wiz)
    djx = compare_histograms('hg_jerk_xyz', train_wiz, test_wiz)
    ddt1 = compare_histograms('hg_dt1', train_wiz, test_wiz)
    ddt2 = compare_histograms('hg_dt2', train_wiz, test_wiz)

    gdiff = np.array([ np.abs(w['median_mag']-test_wiz['median_mag']) for w in train_wiz ])

    resdiff = np.array([ np.abs(w['x_res']-test_wiz['x_res']) for w in train_wiz ])

    ans = percentile(dm) + percentile(dj) + percentile(dx) + percentile(djx)

    # Worse?
    #ans += percentile(dmdev)

    if use_hours:
        ans += percentile(dh)
    if use_dt:
        ans += percentile(ddt1)
        ans += percentile(ddt2)
    if use_G:
        ans += percentile(gdiff)
    if use_resdiff:
        ans -= resdiff[pdev_idx] * 100

    # On Kaggle it was pointed out that the "true" proposals will typically correspond to
    # devices having more samples.  So this should be taken into consideration.
    nsamps_vec = np.array([ nsamps_for_dev[x] for x in ids_to_check ], dtype=float)
    nsamps_vec /= np.max(nsamps_vec)

    assert not np.isnan(ans)
    #if np.isnan(ans):
    #    ans = 0.5

    if cheat_time_compare:
        if test_wiz['min_time'] < train_wisdom[proposed_dev]['max_time']:
            ans = -1e9

    if long_return:
        ret = {
            'ans': ans,
            'proposed': { },
            'others_mean': { },
            'others_median': { },
            'percentiles': { },
        }
        for key in ('dm', 'dj', 'dx', 'dh', 'djx', 'ddt1', 'ddt2', 'nsamps_vec', 'gdiff', 'resdiff'):
            if not key in locals():
                continue
            arr = locals()[key]
            ret['proposed'][key] = arr[pdev_idx]
            others = np.concatenate([ arr[:pdev_idx], arr[pdev_idx+1:] ])
            ret['others_mean'][key] = np.mean(others)
            ret['others_median'][key] = np.median(others)
            ret['percentiles'][key] = percentile(arr)

        return ret
    else:
        return ans

# Is the proposed device likely to be the real device?
def answer_contest_question(sid, proposed_dev):
    test_wiz = parse_from_file('test', sid)
    if int(sid) % 1000 == 0:
        print sid
    return evaluate_question(test_wiz, proposed_dev)

def do_simulated_question(true_dev):
    #print true_dev
    guesses_match = []
    guesses_nomatch = []

    for test_slice in sim_test_sets[true_dev]:
        test_wiz = parse_from_file('train', true_dev, test_slice)

        # Get a false proposed device from among the list of similar devices
        possible_lies = devices_clusters[true_dev]
        assert len(possible_lies) > 1
        while True:
            lie = random.choice(possible_lies)
            if lie != true_dev:
                break

        data = evaluate_question(test_wiz, true_dev, long_return=True)
        data['true_dev'] = true_dev
        data['test_slice'] = test_slice
        guesses_match.append(data)

        data = evaluate_question(test_wiz, lie, long_return=True)
        data['true_dev'] = true_dev
        data['test_slice'] = test_slice
        guesses_nomatch.append(data)

    return (guesses_match, guesses_nomatch)

######################################################################

random.seed(1)

# If 1, then simulate the classification using a portion of the training data.
# If 0, then answer the contest questions.
simulate = 0

use_hours = True
use_dt = True
use_G = True

# Lowers the Kaggle score for some reason.
use_resdiff = False

compare_only_similar = False

# This is disabled since it only catches 4 things on Kaggle, and makes the simulation too easy.
cheat_time_compare = False

# Number of CPUs to use (0 to not use joblib).
num_cpus = 0

# If set to 0 then only train if not already trained
retrain = 0

if num_cpus > 0:
    par = joblib.Parallel(n_jobs=num_cpus, verbose=50)
    delayed = joblib.delayed
else:
    # If only one CPU, then just run in the parent process to facilitate debugging.
    #par = lambda seq: [ f(*args, **kwargs) for (f, args, kwargs) in seq ]
    par = list
    delayed = lambda x: x

idx_for_dev = { dev_id: idx for (idx, dev_id) in enumerate(device_ids) }
nsamps_for_dev = { dev_id: nsamps for (dev_id, nsamps) in zip(device_ids, device_metadata.train_lengths) }

# Read data files (if not already loaded)
if not 'data_files' in locals():
    data_files = {}
if not 'train' in data_files:
    print 'Reading train.pkl...'
    data_files['train'] = pickle.load(open('train.pkl', 'rb'))
    print '... done'
if not simulate and not 'test' in data_files:
    print 'Reading test.pkl...'
    data_files['test'] = pickle.load(open('test.pkl', 'rb'))
    print '... done'

if retrain or not 'train_wisdom' in locals():
    print 'Training.'

    sim_test_sets = defaultdict(list)
    num_sim_test_sets = 0

    train_slices = []

    if simulate:
        # Number of samples for a single test.
        test_set_size = 300

        # For simulation, the last several samples are reserved for testing.  Training happens
        # on everything except for these.
        for dev_id in device_ids:
            nsamps = nsamps_for_dev[dev_id]
            num_test_sets = int(nsamps * 0.01 / test_set_size)
            num_test_sets = max(num_test_sets, 1)
            #num_test_sets = min(num_test_sets, 5)

            # Put a gap between the last training data and the first test data, to make it
            # harder.
            gap = test_set_size*10 if num_test_sets else 0

            samps_for_train = nsamps - gap - num_test_sets * test_set_size
            train_slices.append((dev_id, slice(None, samps_for_train)))

            to_idx = samps_for_train + gap

            for test_idx in range(num_test_sets):
                from_idx = to_idx
                to_idx = from_idx + test_set_size
                sim_test_sets[dev_id].append(slice(from_idx, to_idx))
                num_sim_test_sets += 1

            assert to_idx == nsamps

        print 'num_sim_test_sets:', num_sim_test_sets
    else:
        # Not a simulation: train on all input data
        for dev_id in device_ids:
            train_slices.append((dev_id, slice(None, None)))

    train_wiz_list = par(
        delayed(parse_from_file)('train', dev_id, train_slice)
        for (dev_id, train_slice) in train_slices)
    assert len(train_wiz_list) == len(device_ids)
    train_wisdom = {}
    for (dev_id, train_wiz) in zip(device_ids, train_wiz_list):
        train_wisdom[dev_id] = train_wiz

# FIXME
#par = list; delayed = lambda x: x

if simulate:
    # Try one outside of parfor just to catch errors early.
    do_simulated_question(sim_test_sets.keys()[0])

    print 'Running simulated classification.'
    sim_results = par(delayed(do_simulated_question)(dev_id) for dev_id in sim_test_sets.keys())

    print 'Analyzing results.'
    data_match = []
    data_nomatch = []
    for (g_match, g_nomatch) in sim_results:
        data_match += g_match
        data_nomatch += g_nomatch

    ans_match = np.array([ x['ans'] for x in data_match ])
    ans_nomatch = np.array([ x['ans'] for x in data_nomatch ])

    num_m = len(data_match)
    num_nom = len(data_nomatch)
    (fpr, tpr, thresholds) = roc_curve([1]*num_m + [0]*num_nom, np.concatenate([ans_match, ans_nomatch]))
    roc_auc = auc(fpr, tpr)

    print '-- Simulation results:'
    print 'mean(ans_match) =', np.mean(ans_match)
    print 'mean(ans_nomatch) =', np.mean(ans_nomatch)
    print 'AUC =', roc_auc
    print 'Worst false negatives:', np.argsort(ans_match)[:10]
else:
    # Only read the data if it's not already loaded.
    if not 'questions' in locals():
        print 'Reading questions.'
        questions = np.loadtxt(open('questions.csv', 'r'), delimiter=',', skiprows=1, dtype=int)

    print 'Evaluating contest questions.'

    answers = par(
        delayed(answer_contest_question)(sid, proposed_dev)
        for (_foo, sid, proposed_dev) in questions)

    print 'Writing answers.csv'

    with open('answers_ds2.csv', 'w') as fh:
        fh.write('QuestionId,IsTrue\n')
        for (answer, (qid, sid, proposed_dev)) in zip(answers, questions):
            fh.write('%d,%f\n' % (qid, answer))

##############################################
# Helper functions for interactive experiments
##############################################

import pylab

def plot_me_vs_median(key):
    def get_feat(key, data):
        a = np.array([ x['proposed'][key] for x in data ])
        b = np.array([ x['others_median'][key] for x in data ])
        return (a, b)
        #return np.array([ x['others_mean'][key] / (x['others_mean'][key] + x['proposed'][key]) for x in data ])

    pylab.clf()
    (a,b) = get_feat(key, data_match)
    pylab.plot(a, b, 'b+')
    (a,b) = get_feat(key, data_nomatch)
    pylab.plot(a, b, 'r+')

def get_percentiles(key, data):
    return np.array([ x['percentiles'][key] for x in data ])

def plot_percentiles(key):
    pylab.clf()
    pylab.hist(
        [get_percentiles(key, x) for x in (data_match, data_nomatch)],
        100, histtype='step', normed=1, cumulative=1, color=('b','r'))

def plot_percentiles_2d(k1, k2):
    pylab.clf()
    for (data, sym) in ((data_match, 'g.'), (data_nomatch, 'r.')):
        v1 = get_percentiles(k1, data)
        v2 = get_percentiles(k2, data)
        pylab.plot(v1, v2, sym)

def train_vs_test(testcase):
    dev_id = testcase['true_dev']
    test_slice = testcase['test_slice']
    info_train = parse_timeseries(data_files['train'][dev_id][:test_slice.start,:], True)
    info_test = parse_timeseries(data_files['train'][dev_id][test_slice,:], True)
    return (info_train, info_test)

def compare_training_1d(a, b, kx):
    pylab.clf()
    pylab.plot(a['extras']['t'], a['extras'][kx], 'b-')
    pylab.plot(a['extras']['t'], a['extras'][kx], 'b+')
    pylab.plot(b['extras']['t'], b['extras'][kx], 'r-')
    pylab.plot(b['extras']['t'], b['extras'][kx], 'r+')

def compare_training2d(a, b, kx, ky):
    pylab.clf()
    pylab.plot(a['extras'][kx], a['extras'][ky], 'b,')
    pylab.plot(b['extras'][kx], b['extras'][ky], 'r,')

def plot_xyz(dev_id, sym='-'):
    data = data_files['train'][dev_id]
    pylab.clf()
    pylab.plot(data[:,0], data[:,1], sym)
    pylab.plot(data[:,0], data[:,2], sym)
    pylab.plot(data[:,0], data[:,3], sym)

#(a, b) = train_vs_test(data_match[833])

#q = np.floor(x/a + 0.5); q = q[(q > -20) & (q < 20)]; np.unique(q)
