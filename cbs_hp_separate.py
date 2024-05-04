import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import logging
from scipy.stats import ttest_ind
import argparse  


log = logging.getLogger()
logging.basicConfig(level=logging.WARN)


def cbs_stat(x):
    '''Given x, Compute the subinterval x[i0:i1] with the maximal segmentation statistic t. 
    Returns t, i0, i1'''
    
    x0 = x - np.mean(x)
    n = len(x0)
    y = np.cumsum(x0)
    e0, e1 = np.argmin(y), np.argmax(y)
    i0, i1 = min(e0, e1), max(e0, e1)
    s0, s1 = y[i0], y[i1]
    print("n: ", n, ", i: ", i0, ", i1: ", i1)
    # Perform independent samples t-test
    t_statistic, p_value = ttest_ind(x[i0:(i1+1)], np.concatenate([x[0:i0],x[i1+1:n]]))
    return t_statistic, p_value, i0, i1+1
    return (s1-s0)**2*n/(i1-i0+1)/(n+1-i1+i0), i0, i1+1

def tstat(x, i):
    '''Return the segmentation statistic t testing if i is a (one-sided) breakpoint in x'''
    n = len(x)
    s0 = np.mean(x[:i])
    s1 = np.mean(x[i:])
    return (n-i)*i/n*(s0-s1)**2

def cbs(x, shuffles=1000, p=.05):
    '''Given x, find the interval x[i0:i1] with maximal segmentation statistic t. Test that statistic against
    given (shuffles) number of random permutations with significance p.  Return True/False, t, i0, i1; True if
    interval is significant, false otherwise.'''

    max_t, p_value, max_start, max_end = cbs_stat(x)
    print("p value: ", p_value)
    if max_end-max_start == len(x):
        return False, max_t, max_start, max_end
    if max_start < 5:
        max_start = 0
    if len(x)-max_end < 5:
        max_end = len(x)
    if p_value > p:
        return False, max_t, max_start, max_end
    return True, max_t, max_start, max_end


def rsegment(x, start, end, L=[], shuffles=1000, p=.05):
    '''Recursively segment the interval x[start:end] returning a list L of pairs (i,j) where each (i,j) is a significant segment.
    '''
    threshold, t, s, e = cbs(x[start:end], shuffles=shuffles, p=p)
    log.info('Proposed partition of {} to {} from {} to {} with t value {} is {}'.format(start, end, start+s, start+e, t, threshold))
    if (not threshold) | (e-s < 5) | (e-s == end-start):
        L.append((start, end))
    else:
        if s > 0:
            rsegment(x, start, start+s, L, shuffles=shuffles, p=p)
        if e-s > 0:
            rsegment(x, start+s, start+e, L, shuffles=shuffles, p=p)
        if start+e < end:
            rsegment(x, start+e, end, L, shuffles=shuffles, p=p)
    return L


def segment(x, shuffles=1000, p=.05):
    '''Segment the array x, using significance test based on shuffles rearrangements and significance level p
    '''
    start = 0
    end = len(x)
    L = []
    rsegment(x, start, end, L, shuffles=shuffles, p=p)
    return L


def validate(x, L, shuffles=1000, p=.01):
    S = [i[0] for i in L]+[len(x)]
    SV = [0]
    left = 0
    for test, s in enumerate(S[1:-1]):
        t = tstat(x[S[left]:S[test+2]], S[test+1]-S[left])
        log.info('Testing validity of {} in interval from {} to {} yields statistic {}'.format(S[test+1], S[left], S[test+2], t))
        threshold = 0
        thresh_count = 0
        site = S[test+1]-S[left]
        xt = x[S[left]:S[test+2]].copy()
        flag = True
        for k in range(shuffles):
            np.random.shuffle(xt)
            threshold = tstat(xt, site)
            if threshold > t:
                thresh_count += 1
            if thresh_count >= p*shuffles:
                flag = False
                log.info('Breakpoint {} rejected'.format(S[test+1]))
                break
        if flag:
            log.info('Breakpoint {} accepted'.format(S[test+1]))
            SV.append(S[test+1])
            left += 1
    SV.append(S[-1])
    return SV

def read_array_of_methylation_perc(filepath):
    '''Read input from sorted bedmethyl file and create an array with sequence of %methylation '''
    print("Reading CpG array for haplotype-1")
    df = pd.read_csv(filepath, sep="\t", header=None)
    chrom_str = df[0][0]
    data_positions = np.array(df.iloc[:, 1], dtype=np.int64)
    df = df.iloc[:, 10]
    data = np.array(df, dtype=np.float64)
    print(data[0:5], data_positions[0:5])
    return data, data_positions, chrom_str

def draw_segmented_data(data, S, title=None):
    '''Draw a scatterplot of the data with vertical lines at segment boundaries and horizontal lines at means of 
    the segments. S is a list of segment boundaries.'''
    # Define custom colors based on condition
    colors = ['red' if (60 <= y <= 100) else 'blue' if (0 <= y <= 30) else 'grey' for y in data]
    print("colors size: ", len(colors))
    j=sns.scatterplot(x=range(len(data)),y=data,color=colors,size=.1,legend=None)
    print("data size: ", len(data))
    print("S: ", S)
    for x in S:
        j.axvline(x)
    for i in range(1,len(S)):
        j.hlines(np.mean(data[S[i-1]:S[i]]),S[i-1],S[i],color='green')
    j.set_title(title)
    j.set_ylim(-10,110)
    j.get_figure().set_size_inches(16,4)
    return j

if __name__ == '__main__':
    parser = argparse.ArgumentParser()  
  
    # creating two variables using the add_argument method  
    parser.add_argument("-file", metavar='', required=True, help="modkit filtered input file")
    parser.add_argument("-o", metavar='', required=True, help="output file", type=str)
    parser.add_argument("-p", metavar='', required=False, default=0.5, help="p-value, DEFAULT: 0.5", type=float)
    parser.add_argument("-s", metavar='', required=True, help="sample name", type=str)
    parser.add_argument("-t", metavar='', required=True, help="target name", type=str)
    parser.add_argument("-plot", metavar='', required=False, default="no", help="make segmentation plots?='yes/no', DEFAULT: no", type=str)

    args = parser.parse_args()

    log.setLevel(logging.INFO)
    # array with delta methylation % from combined
    sample, sample_positions, chrom_str = read_array_of_methylation_perc(filepath=args.file)
    
    L = segment(sample, shuffles=1000, p=args.p)
    S = validate(sample, L, p=args.p)
    csv_arr = []
    seg_off_file_obj = open(args.o, "a")
    for i in range(len(S)-1):
        segment_mean = np.mean(sample[S[i]:S[i+1]])
        if segment_mean >= 60:
            # methylated
            segment_igv_color = [255, 0, 0]
        elif segment_mean <= 30:
            # unmethylated
            segment_igv_color = [0, 0, 255]
        else:
           segment_igv_color = [192, 192, 192]
        seg_off_file_obj.write(chrom_str + "\t" + str(sample_positions[S[i]]) + "\t" + str(sample_positions[S[i+1]-1]+1) + "\t" + args.t + "_" + str(i+1) + "\t" + "0\t.\t" + str(sample_positions[S[i]]) + "\t" + str(sample_positions[S[i+1]-1]+1) + "\t" + str(segment_igv_color[0]) + "," + str(segment_igv_color[1]) + "," + str(segment_igv_color[2]) + "\t" + str(segment_mean) + "\t" + str(S[i+1]-S[i]) + "\n")
    
    seg_off_file_obj.close()
    if args.plot == "yes":
        ax = draw_segmented_data(sample,  S, title='Circular Binary Segmentation of ' + args.t + ' target region in ' + args.s)
        ax.get_figure().savefig(args.s + "_" + args.t + "_" + chrom_str + "_" + str(sample_positions[S[0]]) + "_" + str(sample_positions[S[-1]-1]+1) + ".png")

    # M11AO_hp1_FMR_vntr1_chrom_startoff_endoff.png