#!/usr/bin/env python3
# isitabarrel.py IsItABarrel prediction method for bacterial TMBBs developed by the Slusky Lab, University of Kansas.
#
# Copyright (C) 2022 University of Kansas
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Affero General Public License as published
#    by the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU Affero General Public License for more details.
#
#    You should have received a copy of the GNU Affero General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
#   @authors Rebecca Bernstein, Daniel Montezano

# Details:
# Iterates through every contact map pkl file in a folder and calculates 6 different scores.
# Explanation of the Different Scores:
	# CC2 score: the average of the diagonal line of the closing contact; 0 for nonTMBB, > 0 for TMBB, always < 1
	# BSS score: the number of hairpins found; >= bss_threshold for TMBB, < bss_threshold for nonTMBB, always a whole number
	# H4 score: the number of hairpins found that are h4_low_threshold-h4_high_threshold in length; >= h4_threshold for TMBB, < h4_threshold for nonTMBB, always a whole number
	# CC2 -> BSS score:
		# 1. checks CC2 score
		# 2. if CC2 score > 0 then CC2 -> BSS score = CC2 score (TMBB)
		# 3. if CC2 score = 0, checks BSS score -> if BSS score >= bss_threshold then CC2 -> BSS score = BSS score (TMBB), if BSS score < bss_threshold then CC2 -> BSS score = 0 (nonTMBB)
	# CC2 -> H4 score:
		# 1. checks CC2 score
		# 2. if CC2 score > 0 then CC2 -> H4 score = CC2 score (TMBB)
		# 3. if CC2 score = 0, checks H4 score -> if H4 score >= h4_threshold then CC2 -> H4 score = H4 score (TMBB), if H4 score < h4_threshold then CC2 -> H4 score = 0 (nonTMBB)
	# BSS -> CC2 -> BSS score:
		# 1. checks BSS score - if there are fewer than (bss2_threshold-2) hairpins, BSS -> CC2 -> BSS score = 0 (nonTMBB)
		# 2. if there are more than (bss2_threshold-2) hairpins, checks CC2 score -> if CC2 score > 0 then BSS -> CC2 -> BSS score = CC2 score (TMBB)
		# 3. if there are more than (bss2_threshold-2) hairpins and CC2 score = 0, checks BSS score again -> if there are (bss2_threshold+2) or more hairpins then BSS -> CC2 -> BSS score = BSS score (TMBB); if there are fewer than (bss2_threshold+2) hairpins then BSS -> CC2 -> BSS score = 0 (nonTMBB)

# Note 1: RaptorX was used to create contact maps but contact maps created with other tools may be used.
# Note 2: For reproducing results in the publication, use the column CC2_TO_H4 in the output file.

# Note 3: Parameters for predicting bacterial sequences after optimization by grid search:
# SCAN_BOX_SIZE=10
# N_BOX=52
# Z_THRESHOLD=18
# H4_THRESHOLD=4
# BSS_THRESHOLD=5
# BSS2_THRESHOLD=3
# GAP_DIAGONAL_STRAND=27
# SEARCH_FOR_HAIRPIN=20
# ROW_RANGE=69
# COLUMN_RANGE=7
# LENGTH_SEARCH=40
# H4_LOW_THRESHOLD=16
# H4_HIGH_THRESHOLD=25

import pickle
import numpy as np
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sklearn
import sklearn.metrics
from sklearn.metrics import balanced_accuracy_score

# Arguments
parser = argparse.ArgumentParser("ISITABARREL")
parser.add_argument("protid_list",help="file with the list of protein IDs one per line.")
parser.add_argument("data_folder",help="folder with the maps")
parser.add_argument("--scan_box_size", type=int, default=10, help="size of square box used for scanning for closing contacts.")
parser.add_argument("--n_box", type=int, default=52, help="size of box around the main diagonal when looking for hairpins.")
parser.add_argument("--z_threshold", type=float, default=18, help="threshold for magnitude of hairpin peaks.")
parser.add_argument("--h4_threshold", type=int, default=4, help="threshold for number of beta strands h4_low_threshold-h4_high_threshold in length.")
parser.add_argument("--bss_threshold", type=int, default=5, help="threshold for number of beta strands for CC2 to BSS score.")
parser.add_argument("--bss2_threshold", type=int, default=3, help="threshold for number of beta strands for BSS to CC2 to BSS score.")
parser.add_argument("--gap_diag_strand", type=int, default=27, help="number of spaces off the diagonal to look for start of a hairpin.")
parser.add_argument("--search_for_hairpin", type=int, default=20, help="number of spaces to search around the suggested start of a hairpin.")
parser.add_argument("--row_range", type=int, default=69, help="how large of a box to search for the closing contact in the row direction.")
parser.add_argument("--column_range", type=int, default=7, help="how large of a box to search for the closing contact in the column direction.")
parser.add_argument("--length_search",type=int, default=40, help="maximum number of residues to search for length of hairpins.")
parser.add_argument("--h4_low_threshold",type=int, default=16, help="minimum length of hairpins that are counted for H4 score.")
parser.add_argument("--h4_high_threshold",type=int, default=25, help="maximum length of hairpins that are counted for H4 score.")
args = parser.parse_args()

print("Using the following arguments:")
print("SCAN_BOX_SIZE:",args.scan_box_size)
print("N_BOX:",args.n_box)
print("Z_THRESHOLD:",args.z_threshold)
print("H4_THRESHOLD:",args.h4_threshold)
print("BSS_THRESHOLD:",args.bss_threshold)
print("BSS2_THRESHOLD:",args.bss2_threshold)
print("GAP_DIAG_STRAND:",args.gap_diag_strand)
print("SEARCH_FOR_HAIRPIN:",args.search_for_hairpin)
print("ROW_RANGE:",args.row_range)
print("COLUMN_RANGE:",args.column_range)
print("LENGTH_SEARCH:",args.length_search)
print("H4_LOW_THRESHOLD:",args.h4_low_threshold)
print("H4_HIGH_THRESHOLD:",args.h4_high_threshold)
scan_box_size = args.scan_box_size
n_box = args.n_box
pw = int(n_box/2)					#pw = pad width: amount of padding to apply when calculating beta strand score
z_threshold = args.z_threshold
h4_threshold = args.h4_threshold
bss_threshold = args.bss_threshold
bss2_threshold = args.bss2_threshold
gap_diag_strand = args.gap_diag_strand
search_for_hairpin = args.search_for_hairpin + 1
row_range = args.row_range
column_range = args.column_range
length_search = args.length_search
h4_low_threshold = args.h4_low_threshold
h4_high_threshold = args.h4_high_threshold

map_name = []						# saves the names of the PKL files containing the contact maps (the protein IDs)
map_name = []						# creates list with protein ids used to name the PKL files containing the contact maps (the protein IDs)
fh = open(args.protid_list)
for line in fh:
	map_name.append(line.rstrip().split('\t')[0])
fh.close()

beta_hairpin_tuple = []	# saves the beta hairpin tuple for each protein
cc2 = []			# saves the CC2 score for each protein
h4 = []                        # saves the H4 score for each protein
cc2_to_h4 = []			# saves the CC2 -> H4 score for each protein
cc2_to_bss = []		# saves the CC2 -> BSS score for each protein
bss_to_cc2_to_bss = []		# saves the BSS -> CC2 -> BSS score for each protein


#######################
# FUNCTION DEFINITIONS
#######################

def diagonal_line(cmap, lowest_residue_i, highest_residue_i, lowest_residue_j, highest_residue_j, pw=0, box_width=scan_box_size):
	''' Parameter cmap contains the contact map (Numpy array).
	Use the parameters lowest_residue_i, highest_residue_i, lowest_residue_j, highest_residue_j
	to define a rectangular region from the contact map.
	Use the parameter pw to define the padding width (default 0).
	Use the parameter box_width to define the size of the rectangular region (default scan_box_size).
	RETURNS: A 2-tuple. The first element is the average of the flipped diagonal of the rectangular region
	(if the flipped diagonal contains all elements greater than zero) OR None. The second element is
	a Numpy array of the rectangular region offset by the padding width. '''
	i_low =  lowest_residue_i  + pw
	i_high = highest_residue_i + pw
	j_low =  lowest_residue_j  + pw
	j_high = highest_residue_j + pw
	a = cmap[i_low:i_high, j_low:j_high]
	dline = np.diag(np.flipud(a))
	count = 0
	for x in dline:
		if x == 0:
			break
		else:
			count = count + 1
	value = None
	if count == box_width:
		assert a.shape == (box_width,box_width)
		average1 = np.mean(dline)
		average2 = (np.sum(np.tril(np.flipud(a), k = -3)))/np.sum(range(box_width-2))
		average3 = (np.sum(np.tril(np.fliplr(a), k = -3)))/np.sum(range(box_width-2))
		value = average1 - (average2 + average3)/2
	return (value,a)


def find_hairpin(cmap, num1, num2, pw, box_width=scan_box_size):
    ''' Parameter cmap contains the contact map (Numpy array).
	Use the parameters num1 and num2 to define bottomleft corner index of rectangular region to start
	searching for hairpin.
	NOTE that num1 and num2 define a point on the main diagonal so they always start equal.
	Use the parameter pw to define the padding width.
	Use the parameter box_width to define the size of the rectangular region (default scan_box_size).
	Loops over positions looking for a hairpin line. This is needed because sometimes there is
	a gap between the main diagonal and the hairpin line.
	RETURNS: A 2-tuple. The first element is the column index of the bottomleft corner of the box
    if a full diagonal was found OR None. The second element is the row index of the bottomleft
    corner of the box if a full diagonal was found OR None. '''
    assert num1 == num2
    # it moves the box up and right to look for a strand that does not touch the diagonal
    for _ in range(gap_diag_strand):
        (a,_) = diagonal_line(cmap, num1-(box_width-1), num1+1, num2, num2+box_width, pw)
        if a is not None:
            return (num1,num2)
        num1 = num1 - 1
        num2 = num2 + 1

def calculate_beta_strand_score(cmap, L, n, z, pw):
    ''' Parameter cmap contains the contact map (Numpy array).
    Use the parameter L to define the length of the protein.
    Use the parameter n to define a square box to scan the map down the middle diagonal line, starting at the top left corner of the map.
    Use the parameter z to define the threshold to count the number of peaks above.
    Use the parameter pw to define the padding width.
    RETURNS: A 2-tuple. The first element is the number of hairpins >= z (BSS Score).
    The second element is a list of the residue indices where a peak was found (sum is above the z threshold). '''

    vectorized_map = cmap.copy().reshape((L*L,1)) # copy, otherwise the original map will be acted upon
    vectorized_map.sort(axis=0)
    cutoff = vectorized_map[-int(np.log2(L*L)*L)] # threshold uses log2()
    clean_cmap = cmap.copy()
    clean_cmap[clean_cmap < cutoff] = 0
    # pad both types of maps
    clean_cmap = np.pad(clean_cmap, pad_width=pw, mode='constant', constant_values=0)
    full_cmap = np.pad(cmap, pad_width=pw, mode='constant', constant_values=0)
    assert clean_cmap.shape[0] == clean_cmap.shape[1]
    assert clean_cmap.shape[0] == (L + pw + pw)
    assert full_cmap.shape[0] == full_cmap.shape[1]
    assert full_cmap.shape[0] == (L + pw + pw)

    # computes the sum of the flipped diagonal for every rectangle on the diagonal line up the middle of the contact map
    sum_list = list()
    for offset in range(pw,pw+L):
        the_box = clean_cmap[offset-int(n/2) : offset+int(n/2), offset-int(n/2) : offset+int(n/2)]
        a = np.sum(np.diag(np.flipud(the_box)))
        sum_list.append(a)
    sum_list = np.array(sum_list)

    # determine the residue indices where the sum of the flipped diagonal is greater than or equal to z
    # count the number of hairpins by counting the number of residue indices where the sum of the flipped diagonal is greater than or equal to z
    peaks = []                         # the residue indices where the sum of the flipped diagonal is greater than or equal to z
    num_hairpins = 0                   # the number of hairpins (if we don't find anything this is left at 0)
    assert len(sum_list) == L
    for i in range(len(sum_list)):
        if sum_list[i] >= z:
            num_hairpins = num_hairpins + 1
            peaks.append(i)
            # after finding a putative hairpin, we test if the new one is too close to the previous one
            if len(peaks) >= 2:
                # we want to guarantee that any two hairpins are at least 5 indices apart
                if i <= peaks[num_hairpins-2] + 5:
                    # if it is too close we delete it from the list and decrement the hairpin count back by one
                    peaks.pop(num_hairpins-1)
                    num_hairpins = num_hairpins - 1

    return (num_hairpins,peaks)

def calculate_cc2(cmap, L, pw, num_hairpins, peaks):
    ''' This function uses predicted beta hairpins as anchors to identify closing contacts line.
	Parameter cmap contains the contact map (Numpy array).
    Use the parameter L to define the length of the protein.
    Use the parameter pw to define the padding width.
    Use the parameter num_hairpins to define the number of hairpins.
    Use the parameter peaks to define a list of the residue indices where there is a putative hairpin.
    RETURNS: CC2 Score. '''

    vectorized_map = cmap.copy().reshape((L*L,1)) # copy, otherwise the original map will be acted upon.
    vectorized_map.sort(axis=0)
    cutoff = vectorized_map[-int(np.log2(L*L)*L)]
    clean_cmap = cmap.copy()
    clean_cmap[clean_cmap < cutoff] = 0
    clean_cmap = np.pad(clean_cmap, pad_width=pw, mode='constant', constant_values=0)
    full_cmap = np.pad(cmap, pad_width=pw, mode='constant', constant_values=0)
    this_cmap = clean_cmap

    # y needs to be initialized here, otherwise it may not be reached
    y = None
    # start with the last rightmost hairpin detected
    try:
        lh = peaks.pop(len(peaks)-1)
    except:
        return 0                         # CC2 score = 0

    # determine residue column index for the starting point of the last rightmost hairpin
    for r in range(search_for_hairpin):
        last_i = lh + r
        last_j = lh + r
        last_hairpin = find_hairpin(this_cmap, last_i, last_j, pw)
        if last_hairpin is None:
            last_i = lh - r
            last_j = lh - r
            last_hairpin = find_hairpin(this_cmap, last_i, last_j, pw)
            if last_hairpin is not None:
                last_hairpin = last_hairpin[1]
                break
        else:
            last_hairpin = last_hairpin[1]
            break
    if last_hairpin is None:
        last_hairpin = lh

    # determine residue row index for the starting point of all of the hairpins (excluding the last rightmost hairpin)
    residue_list = []
    for x in peaks:
        for r in range(search_for_hairpin):
            i = x + r
            j = x + r
            a = find_hairpin(this_cmap, i, j, pw)
            if a is None:
                i = x - r
                j = x - r
                a = find_hairpin(this_cmap, i, j, pw)
                if a is not None:
                    a = a[0]
                    break
            else:
                a = a[0]
                break
        residue_list.append(a)

	# after these new indices are found, loop from the first hairpin to the before-before-last hairpin
    for x in residue_list[0:(len(residue_list)-3)+1]:
        if x is None:
            continue
        if L <= last_hairpin + ((row_range/2) + (scan_box_size-1)):
            # define anchor for a box that is on edge of map if last rightmost hairpin is found close to end of map
            lrj = int(L - scan_box_size)
            hrj = int(L - 1)
        else:
            # define anchor for a box if there is enough room away from end of map
            lrj = int(last_hairpin + (row_range/2))
            hrj = int(last_hairpin + ((row_range/2) + (scan_box_size-1)))
        # use column index of the last rightmost hairpin and row index from residue_list as an anchor
        # build rectangle that has length and width defined by row_range and column_range around this anchor (anchor is center of rectangle)
        done = False
        for i in range(row_range+1):
            lri = int(x - (column_range/2))
            hri = int(x - ((column_range/2)-(scan_box_size-1)))
            for j in range(column_range+1):
                # search for flipped diagonal line in every smaller box inside of rectangle
                # smaller box: squares in which length & width = scan_box_size
                (y,_) = diagonal_line(clean_cmap, lri, hri+1, lrj, hrj+1, pw)
                if y is not None:
                    return y             # y = CC2 score = average of flipped diagonal
                    done = True
                    break
                lri = lri + 1
                hri = hri + 1
            if done:
                break
            lrj = lrj - 1
            hrj = hrj - 1
        if done:
            break
    if y is None:
        return 0                         # CC2 score = 0

def find_length_of_hairpins(cmap,low_i,high_j,pw):
	''' Parameter cmap contains the contact map (Numpy array).
    Use the parameter low_i to define row index for start of hairpin.
	Use the parameter low_j to define column index for start of hairpin.
	Use the parameter pw to define the padding width.
	RETURNS: length of a hairpin. '''
	length = 0
	for y in range(length_search):
		(a,_) = diagonal_line(cmap, low_i, low_i+scan_box_size, high_j-(scan_box_size-1), high_j+1)
		if a is None:
			# there is not a full diagonal line, so count the number of indices along the diagonal line that are >0
			b = cmap[low_i:(low_i+scan_box_size), high_j-(scan_box_size-1):high_j+1]
			dline = np.diag(np.flipud(b))
			for x in dline:
				if x != 0:
					length = length + 1
			break
		else:
			# there is a full diagonal line, so increase length by 1
			length = length + 1
		high_j = high_j - 1
		low_i = low_i + 1
	return length

def find_beginning_of_hairpins(cmap, L, z, pw):
	''' Parameter cmap contains the contact map (Numpy array).
    Use the parameter L to define the length of the protein.
    Use the parameter z to define the threshold to count the number of peaks above.
    Use the parameter pw to define the padding width.
    RETURNS: A 2-tuple. The first element is a list of the row indices for the beginning of all of the hairpins.
    The second element is a list of the column indices for the beginning of all of the hairpins. '''

	# computes the sum of the flipped diagonal for every rectangle on the diagonal line up the middle of the contact map
	sum_list = list()
	residue_list = [x-0.5 for x in range(L+pw)]
	for offset in range(pw,pw+L):
		a = np.sum(np.diag(np.flipud(cmap[offset-pw:(offset+pw)+1, offset-pw:(offset+pw)+1])))
		sum_list.append(a)
	sum_list = np.array(sum_list)

	# determine the residue indices where the sum of the flipped diagonal is greater than or equal to z
	peaks = []                         # the residue indices where the sum of the flipped diagonal is greater than or equal to z
	num_hairpins = 0
	for i in range(len(sum_list)):
		if sum_list[i] >= z:
			num_hairpins = num_hairpins + 1
			peaks.append(residue_list[i])
			if len(peaks) >= 2:
				if residue_list[i] <= peaks[num_hairpins-2] + 5:
					peaks.pop(num_hairpins-1)
					num_hairpins = num_hairpins - 1

	residue_list_j = []                # list of the column indices for the beginning of all of the hairpins
	residue_list_i = []                # list of the row indices for the beginning of all of the hairpins

	# create residue_list_j and residue_list_i
	for x in peaks:
		for y in range(search_for_hairpin):
			flag = False
			x1 = int(x) + y
			x2 = int(x) + y
			a = find_hairpin(cmap, x1, x2, pw)
			if a is None:
				x1 = int(x) - y
				x2 = int(x) - y
				a = find_hairpin(cmap, x1, x2, pw)
				if a is not None:
					flag = True
					b = a[0]
					c = a[1]
					break
			else:
				flag = True
				b = a[0]
				c = a[1]
				break
		if flag is False:
			residue_list_j.append(a)
			residue_list_i.append(a)
		else:
			residue_list_j.append(b)
			residue_list_i.append(c)

	return (residue_list_i,residue_list_j)

def length_of_hairpins_test(cmap):
	''' Parameter cmap contains the contact map (Numpy array).
	RETURNS: H4 Score. '''

	contact_map1 = cmap.copy()
	vectorized_map = contact_map1.reshape((L*L,1))
	vectorized_map.sort(axis=0)
	cutoff = vectorized_map[-int(15*L)]
	contact_map1 = cmap.copy()
	contact_map1[contact_map1 < cutoff] = 0
	padded_contact_map = np.pad(contact_map1, pad_width=pw, mode='constant', constant_values=0)

	list1 = []
	list2 = []
	# get lists of the row and column indices for the suggested beginning of all of the hairpins
	a = find_beginning_of_hairpins(padded_contact_map, L, z_threshold, pw)
	for x in a:
		count = 0
		for y in range(len(x)):
			if x[count] is None:
				# remove None from lists
				a[0].pop(a[0].index(a[0][count]))
				a[1].pop(a[1].index(a[1][count]))
			else:
				count = count + 1
	res_list_i = a[0]                  # list of the row indices for the suggested beginning of all of the hairpins
	res_list_j = a[1]                  # list of the column indices for the suggested beginning of all of the hairpins
	b = np.array([res_list_i,res_list_j])
	count = 0
	# loop through all of the hairpins
	for x in range(b.shape[1]):
		for y in range(search_for_hairpin):
			# create a temporary list of different possibilities of the length of the hairpin using different starting points to count the length
			list1.append(find_length_of_hairpins(contact_map1,b[0,count],b[1,count],pw))
			list1.append(find_length_of_hairpins(contact_map1,b[0,count],b[1,count]+y,pw))
			list1.append(find_length_of_hairpins(contact_map1,b[0,count],b[1,count]-y,pw))
			list1.append(find_length_of_hairpins(contact_map1,b[0,count]+y,b[1,count],pw))
			list1.append(find_length_of_hairpins(contact_map1,b[0,count]-y,b[1,count],pw))
		# length = the maximum length value in the temporary list
		# create a list of the lengths of all of the hairpins
		list2.append(max(list1))
		list1 = []
		count = count + 1

	# H4 score = number of hairpins with a length greater than or equal to h4_low_threshold and less than or equal to h4_high_threshold
	list2 = np.array(list2)
	h4 = list2[(list2 >= h4_low_threshold) & (list2 <= h4_high_threshold)].size

	return h4


##############################
# END OF FUNCTION DEFINITIONS
##############################

for k in range(len(map_name)):
	# load and threshold the map for the current protein
	pickle_data = open(args.data_folder + "/" + map_name[k] + ".pkl","rb")
	contact_map = pickle.load(pickle_data)
	pickle_data.close()
	L = contact_map.shape[0]           # the lengh of the unpadded map

	# the tuple returned has (num_hairpins,peaks)
	beta_hairpin_tuple.append(calculate_beta_strand_score(contact_map, L, n_box, z_threshold, pw))

	# the list returned has CC2 score
	cc2.append(calculate_cc2(contact_map, L, pw, num_hairpins=beta_hairpin_tuple[-1][0], peaks=beta_hairpin_tuple[-1][1]))

	# the list returned has H4 score
	h4.append(length_of_hairpins_test(contact_map))

# one of the lists returned has CC2 -> H4 score
# the other list returned has CC2 -> BSS score
for k in range(len(cc2)):
	if cc2[k] == 0:
		if h4[k] >= h4_threshold:
			cc2_to_h4.append(h4[k])
		if h4[k] < h4_threshold:
			cc2_to_h4.append(0)
		if (beta_hairpin_tuple[k])[0] >= bss_threshold:
			cc2_to_bss.append((beta_hairpin_tuple[k])[0])
		if (beta_hairpin_tuple[k])[0] < bss_threshold:
			cc2_to_bss.append(0)
	else:
		cc2_to_h4.append(cc2[k])
		cc2_to_bss.append(cc2[k])

# the list returned has BSS -> CC2 -> BSS score
for k in range(len(beta_hairpin_tuple)):
	if (beta_hairpin_tuple[k])[0] <= bss2_threshold-2:
		bss_to_cc2_to_bss.append(0)
	else:
		if cc2[k] == 0:
			if (beta_hairpin_tuple[k])[0] >= bss2_threshold+2:
				bss_to_cc2_to_bss.append((beta_hairpin_tuple[k])[0])
			if (beta_hairpin_tuple[k])[0] < bss2_threshold+2:
				bss_to_cc2_to_bss.append(0)
		else:
			bss_to_cc2_to_bss.append(cc2[k])

# create text file with results
stringg = 'MAP_NAME' + '\t' + 'CC2' + '\t' + 'BSS' + '\t' + 'H4' + '\t' + 'CC2_TO_H4' + '\t' + 'CC2_TO_BSS' + '\t' + 'BSS_TO_CC2_TO_BSS' + '\n'
for z in range(len(map_name)):
	stringg = stringg + map_name[z] + '\t' + str(cc2[z]) + '\t' + str((beta_hairpin_tuple[z])[0]) + '\t' + str(h4[z]) + '\t' + str(cc2_to_h4[z]) + '\t' + str(cc2_to_bss[z]) + '\t' + str(bss_to_cc2_to_bss[z]) + '\n'
file = open("results.tsv","w")
file.write(stringg)
file.close()
