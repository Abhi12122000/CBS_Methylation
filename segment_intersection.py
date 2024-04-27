import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import logging
from scipy.stats import ttest_ind
import argparse  

def read_segments_and_expand(filenames_ls):
    segments_across_samples_over_target_map = {}
    max_end_offset_per_chrom = {}
    target_to_chrom_mapper = {}
    target_order = []
    for fn in filenames_ls:
        df_i = pd.read_csv(fn, sep="\t", header=None)
        print(df_i.head())
        for j in range(len(df_i)):
            target_name = df_i[3][j].split('_')[0]
            if target_name not in segments_across_samples_over_target_map:
                target_order.append(target_name)
                target_to_chrom_mapper[target_name] = df_i[0][j]
                segments_across_samples_over_target_map[target_name] = []
                max_end_offset_per_chrom[target_name] = 0
            # print(target_name, df_i[1][j])
            segments_across_samples_over_target_map[target_name].append(int(df_i[1][j]))
            max_end_offset_per_chrom[target_name] = max(max_end_offset_per_chrom[target_name], int(df_i[2][j]))
    
    for chrom, segments in segments_across_samples_over_target_map.items():
        segments_across_samples_over_target_map[chrom] = sorted(list(set(segments)))
        
    return segments_across_samples_over_target_map, max_end_offset_per_chrom, target_to_chrom_mapper, target_order

def write(outfile, segments_across_samples_over_target_map, max_end_offset_per_chrom, target_to_chrom_mapper, target_order):
    intersection_seg_off_file_obj = open(outfile, "w")
    for target_name in target_order:
        segments = segments_across_samples_over_target_map[target_name]
        for seg_idx in range(len(segments)):
            seg_beg_offset = segments[seg_idx]
            seg_end_offset = (segments[seg_idx + 1]) if (seg_idx < (len(segments) - 1)) else max_end_offset_per_chrom[target_name]
            intersection_seg_off_file_obj.write(target_to_chrom_mapper[target_name] + "\t" + str(seg_beg_offset) + "\t" + str(seg_end_offset) + "\t" + target_name + "_" + str(seg_idx+1) + "\n")
    intersection_seg_off_file_obj.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()  
  
    infiles = ["inputs/input_samples/segment_boundaries_imprinted_M11AO.bed", "inputs/input_samples/segment_boundaries_imprinted_M11AR.bed", "inputs/input_samples/segment_boundaries_imprinted_M11AU.bed"]
    outfile = "intersected_segment_boundaries.bed"
    # # creating two variables using the add_argument method  
    # parser.add_argument("-infile", help = "haplotype 1 input file")  
    # parser.add_argument("-outfile", help = "haplotype 2 input file")  
    # args = parser.parse_args()

    segments_across_samples_over_target_map, max_end_offset_per_chrom, target_to_chrom_mapper, target_order = read_segments_and_expand(infiles)
    write(outfile, segments_across_samples_over_target_map, max_end_offset_per_chrom, target_to_chrom_mapper, target_order)



