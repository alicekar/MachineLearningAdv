import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from math import pi, sin, cos, e
import scipy.optimize as opt
import matplotlib.patches as mpatches
import random as rd


the_sum = 54
#the_seq = [2, 2, 0, 2, 6, 4, 0, 1, 0, 0] 
#the_seq = [1, 1, 1, 2, 0, 0, 2, 0, 0, 0]
the_seq = [0,0,0,1,0,0,0,0,0,0]
K = len(the_seq) 

print('total sum = ', the_sum)
print()
print('the sequence = ', the_seq)
print()


# WHICH PARTIAL SUMS ARE OK???

ok_partial_sums = []
for k in range(K):
	k_list = []
	for s in range(6*K):
		k_list.append(0)
	ok_partial_sums.append(k_list)

ok_partial_sums = np.array(ok_partial_sums)


# Find all possible based on the sum: 
ok_partial_sums[K-1,the_sum-1] = 1
start_index = the_sum-1
stop_index = the_sum-1

indices_ok = np.zeros([K,2])
indices_ok[K-1,0] = start_index
indices_ok[K-1,1] = stop_index

for k in range(K-1,0,-1):
	#print('step', k)
	sigma = the_seq[k] 	
	if sigma == 0:
		#print('unknown')
		if (start_index-6)>= k-1: 
			start_index = start_index-6
		else: 
			start_index = k-1
		if (stop_index-1)<= 6*k-1: 
			stop_index = stop_index-1
		else: 
			stop_index = 6*k-1

		for j in range(start_index, stop_index+1):
			ok_partial_sums[k-1,j] = 1
	else: 
		if (start_index-sigma)>= k-1: 
			start_index = start_index-sigma
		else: 
			start_index = k-1
		if (stop_index-sigma)<= 6*k-1: 
			stop_index = stop_index-sigma
		else: 
			stop_index = 6*k-1
		for j in range(start_index, stop_index+1):
			ok_partial_sums[k-1,j] = 1
	indices_ok[k-1,0] = start_index
	indices_ok[k-1,1] = stop_index

		

#print(indices_ok)
#print()
print(ok_partial_sums)




#PROBABILITES

# x = output 1,2,3,4,5,6
# z = state t^k_1, t^k_2
# s = partial sum, different for each k, given by indices_ok


pi = np.array([[0.5, 0.5]])
A = np.array([[0.25, 0.75],[0.75, 0.25]])
dist_1 = np.array([1/6, 1/6, 1/6, 1/6, 1/6, 1/6])
dist_2 = np.array([1/2, 1/10, 1/10, 1/10, 1/10, 1/10])

# Joint probabilities: output, state and partial sum
joint_probs_x_z_s_k = []

start_x_z_1 = pi[0,0]*dist_1
start_x_z_2 = pi[0,1]*dist_2
start_x_z = np.array([start_x_z_1, start_x_z_2])
start_x_z_s = []
start_norm = 0
if the_seq[0] == 0: 
	for part_sum in range(int(indices_ok[0,0]), int(indices_ok[0,1])+1):
		prob_mat = np.zeros([2,6])
		prob_mat[0,part_sum] = start_x_z[0,part_sum]
		prob_mat[1,part_sum] = start_x_z[1,part_sum]
		start_norm += (start_x_z[0,part_sum] + start_x_z[1,part_sum])
		start_x_z_s.append(prob_mat)
else:
	for part_sum in range(int(indices_ok[0,0]), int(indices_ok[0,1])+1):
		prob_mat = np.zeros([2,6])
		if part_sum == the_seq[0]-1:
			prob_mat[0,part_sum] = start_x_z[0,part_sum]
			prob_mat[1,part_sum] = start_x_z[1,part_sum]
			start_norm += (start_x_z[0,part_sum] + start_x_z[1,part_sum])
		start_x_z_s.append(prob_mat)
start_x_z_s = np.array(start_x_z_s)
#print('start norm ', start_norm)
if start_norm != 0:
	start_x_z_s /= start_norm
joint_probs_x_z_s_k.append(start_x_z_s)


state_probs_z_s_k = []

start_z_s = []
for mat in joint_probs_x_z_s_k[0]:
	start_z_s.append(np.array([np.sum(mat, axis=1)]))
start_z_s = np.array(start_z_s)
state_probs_z_s_k.append(start_z_s)


for k in range(1,K):
	this_x_z_s = []
	this_norm = 0
	for part_sum in range(int(indices_ok[k,0]), int(indices_ok[k,1])+1):
		prob_mat = np.zeros([2,6])
		this_x_z_s.append(prob_mat)
	curr_index = 0
	for last_state_prob in state_probs_z_s_k[k-1]:
		this_state_prob = np.dot(last_state_prob,A)
		#print()
		#print('k ', k)
		#print('state ', this_state_prob)
		
		this_joint_prob_1 = this_state_prob[0,0]*dist_1
		this_joint_prob_2 = this_state_prob[0,1]*dist_2
		this_joint_prob = np.array([this_joint_prob_1, this_joint_prob_2])
		#print('joint ', this_joint_prob)

		if the_seq[k] == 0:
			for sigma in range(1,7):
				res_part_sum = indices_ok[k-1,0] + curr_index + sigma		
				if res_part_sum >= indices_ok[k,0] and res_part_sum <= indices_ok[k,1]:
					index = int(res_part_sum-indices_ok[k,0])
					this_x_z_s[index][0,sigma-1] = this_joint_prob[0,sigma-1]
					this_x_z_s[index][1,sigma-1] = this_joint_prob[1,sigma-1]
					# print('curr_index ', curr_index)					
					# print(this_x_z_s)
					this_norm += (this_joint_prob[0,sigma-1] + this_joint_prob[1,sigma-1])
		else: 
			set_sigma = the_seq[k]
			res_part_sum = indices_ok[k-1,0] + curr_index + set_sigma		
			if res_part_sum >= indices_ok[k,0] and res_part_sum <= indices_ok[k,1]:
				index = int(res_part_sum-indices_ok[k,0])
				this_x_z_s[index][0,set_sigma-1] = this_joint_prob[0,set_sigma-1]
				this_x_z_s[index][1,set_sigma-1] = this_joint_prob[1,set_sigma-1]
				# print()
				# print(this_x_z_s)				
				this_norm += (this_joint_prob[0,set_sigma-1] + this_joint_prob[1,set_sigma-1])

		curr_index += 1
	this_x_z_s = np.array(this_x_z_s)
	#print('this norm ', this_norm )
	if this_norm != 0:
		this_x_z_s /= this_norm
	joint_probs_x_z_s_k.append(this_x_z_s)

	this_z_s = []
	for mat in joint_probs_x_z_s_k[k]:
		this_z_s.append(np.array([np.sum(mat, axis=1)]))
	this_z_s = np.array(this_z_s)
	state_probs_z_s_k.append(this_z_s)


joint_probs_x_z_s_k = np.array(joint_probs_x_z_s_k)
#print(joint_probs_x_z_s_k) 
state_probs_z_s_k = np.array(state_probs_z_s_k)
#print(state_probs_z_s_k)


final_probs_x_z_k = []
for k in range(K):
	#print('k', k)
	this_final_prob = np.zeros([2,6])
	for i in range(int(indices_ok[k,1] - indices_ok[k,0] +1)):
		#print(joint_probs_x_z_s_k[k][i])
		this_final_prob += joint_probs_x_z_s_k[k][i]
	#print(this_final_prob)
	final_probs_x_z_k.append(this_final_prob)

final_probs_x_z_k = np.array(final_probs_x_z_k)
print(final_probs_x_z_k) 
		 
	
				
	
	


