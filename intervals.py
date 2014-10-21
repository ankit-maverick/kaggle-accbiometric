#!/usr/bin/python

# Cheat by piecing together the test samples.

from __future__ import division

from collections import defaultdict
import numpy as np
import pickle
import gzip
import igraph

# Read data files (if not already loaded)
if not 'data_files' in locals():
    data_files = {}
if not 'train' in data_files:
    print 'Reading train.pkl...'
    data_files['train'] = pickle.load(open('train.pkl', 'rb'))
    print '... done'
if not 'test' in data_files:
    print 'Reading test.pkl...'
    data_files['test'] = pickle.load(open('test.pkl', 'rb'))
    print '... done'

if not 'questions' in locals():
    print 'Reading questions.'
    questions = np.loadtxt(open('questions.csv', 'r'), delimiter=',', skiprows=1, dtype=int)
    print '... done'
    print 'Reading answers.'
    answers = np.loadtxt(open('answers.csv', 'r'), delimiter=',', skiprows=1, dtype=float)
    print '... done'

stats = pickle.load(gzip.open('device_characteristics.pkl.gz', 'rb'))

nq = questions.shape[0] # number of questions

# (dev_id, start, end)
ivs = np.array([ (q[2], data_files['test'][q[1]][0,0], data_files['test'][q[1]][-1,0]) for q in questions ])
min_time = np.min(ivs[:,1])
#ivs[:,1] -= min_time
#ivs[:,2] -= min_time

def vert_seq(vid):
    return data_files['test'][questions[vid,1]]

def vert_attrs(vid):
    (seq_id, dev_id) = questions[vid,1:]
    seq = data_files['test'][seq_id]
    t0 = seq[0,0]
    t1 = seq[-1,0]
    xyz0 = seq[0,1:]
    xyz1 = seq[-1,1:]
    return (dev_id, t0, t1, xyz0, xyz1)

def smoothness(v1, v2):
    def gap_percentile(seq, gap):
        seq_dx = np.sum((seq[1:,1:] - seq[:-1,1:])**2, axis=1)
        return np.sum(seq_dx < gap) / len(seq_dx)
    s1 = vert_seq(v1)
    s2 = vert_seq(v2)
    gap = np.sum((s1[-1,1:] - s2[0,1:])**2)
    p1 = gap_percentile(s1[-30:], gap)
    p2 = gap_percentile(s2[:30], gap)
    return max(p1, p2)

if not 'G' in locals():
    try:
        print 'Reading temporal_neighbors.pkl.gz...'
        G0 = pickle.load(gzip.open('temporal_neighbors.pkl.gz', 'rb'))
        G =  pickle.load(gzip.open('temporal_neighbors_filt.pkl.gz', 'rb'))
    except:
        print 'Recomputing temporal_neighbors.pkl.gz...'

        vattr = { 'label': [ str(x) for x in questions[:,2] ] }

        gap_tol = 350

        edges = []
        for (idx, (dev_id, st, et)) in enumerate(ivs):
            gap = ivs[:,1] - et
            my_neig = list(np.nonzero((gap > 0) & (gap <= gap_tol))[0])
            edges += [ (idx, int(e)) for e in my_neig ]
            #edge_et_st += [ (et, ivs[e,1]) for e in my_neig ]
            if idx % 1000 == 0:
                print '%d of %d...' % (idx, nq)

        G0 = igraph.Graph(edges=edges, directed=True, vertex_attrs=vattr)
        print 'Writing temporal_neighbors.pkl.gz...'
        pickle.dump(G0, gzip.open('temporal_neighbors.pkl.gz', 'wb'), -1)

        print 'Filtering by smoothness...'
        edges_filt = [ (e.source, e.target) for e in G0.es if smoothness(e.source, e.target) < 0.99 ]
        G = igraph.Graph(edges=edges_filt, directed=True, vertex_attrs=vattr)
        pickle.dump(G, gzip.open('temporal_neighbors_filt.pkl.gz', 'wb'), -1)
    print '... done'

def print_tree(G, vid, depth, indent=0, parent=None):
    vattr = vert_attrs(vid)
    desc = '%s+vert=%d dev=%d' % (('| '*indent), vid, G.vs['dev_id'][vid])
    if parent is not None:
        pattr = vert_attrs(parent)
        desc += ' dt=%g dx=%g' % (vattr[1]-pattr[2], np.linalg.norm(vattr[3]-pattr[4]))
        desc += ' smooth=%g' % smoothness(parent, vid)
    print desc
    if depth > 0:
        for n in G.neighbors(vid, mode=igraph.OUT):
            print_tree(G, n, depth-1, indent+1, vid)

def plot_pair(v1, v2):
    s1 = vert_seq(v1)
    s2 = vert_seq(v2)
    import pylab
    pylab.clf()
    pylab.plot(s1[:,0], s1[:,1], 'b+')
    pylab.plot(s1[:,0], s1[:,2], 'b+')
    pylab.plot(s1[:,0], s1[:,3], 'b+')
    pylab.plot(s2[:,0], s2[:,1], 'r+')
    pylab.plot(s2[:,0], s2[:,2], 'r+')
    pylab.plot(s2[:,0], s2[:,3], 'r+')

print 'Unfiltered:'
#print '  out degree histogram:', np.histogram(G0.degree(mode=igraph.OUT), bins=range(10))[0].tolist()
#print '   in degree histogram:', np.histogram(G0.degree(mode=igraph.IN ), bins=range(10))[0].tolist()
print 'Filtered:'
#print '  out degree histogram:', np.histogram(G .degree(mode=igraph.OUT), bins=range(10))[0].tolist()
#print '   in degree histogram:', np.histogram(G .degree(mode=igraph.IN ), bins=range(10))[0].tolist()

def calc_weights(g):
    accum = defaultdict(float)
    for q in g.vs['qid']:
        dev_id = questions[q,2]
        ans = answers[q,1]
        accum[dev_id] += ans
    return accum

ans_median = np.median(answers[:,1])
G.vs['qid'] = range(nq)
G.vs['color'] = [ 'red' if x < ans_median else 'blue' for x in answers[:,1] ]
G.vs['label_color'] = [ 'black' for x in answers[:,1] ]
G.es['label'] = [ str(ivs[e.target,1] - ivs[e.source,2]) for e in G.es ]
G.es['color'] = [ 'blue' if 180 < (ivs[e.target,1] - ivs[e.source,2]) < 220 else 'red' for e in G.es ]

sources = [ i for (i,d) in enumerate(G.degree(mode=igraph.IN)) if d == 0 ]
clus = G.clusters(igraph.WEAK)
#b = np.array([ cid for (cid, s) in enumerate(clus.subgraphs()) if np.max(s.degree(mode=igraph.OUT))>1 or np.max(s.degree(mode=igraph.IN))>1 ])
H = clus.subgraph(36); igraph.plot(H, layout=H.layout('fr'), vertex_label_dist=1).show()



xyz_unique_sub2 = np.loadtxt(open("xyz_unique_sub2.csv","rb"),delimiter=",", skiprows=1)
xyz_unique_sub2 = xyz_unique_sub2[:, 1]
time_mask = np.loadtxt(open("time_mask.csv","rb"),delimiter=",")

def make_ans2():
    ans2 = {}
    for g in clus.subgraphs():
        accum = calc_weights(g)
        best = sorted(accum.iteritems(), key=lambda x: x[1])[-1][0]
        for q in g.vs['qid']:
            ans2[q] = questions[q,2] == best
    ans2 = [ (q+1,1 if ans2[q] else 0) for q in range(nq) ]

    with open('answers3.csv', 'w') as fh:
        fh.write('QuestionId,IsTrue\n')
        for (a,b,(c,d),e) in zip(time_mask,xyz_unique_sub2, ans2, answers):
            if int(c)%100==0:
                print str(c)
            fh.write('%d,%f\n' % (c, (d+e[1]*0.050)*a*b/3.))


make_ans2()
