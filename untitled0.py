# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 10:29:16 2018

@author: ICO0XXX
"""

import csv
import numpy as np
import pandas as pd
import datetime
import time
import os

start_time = time.time()
work_dir = os.getcwd()
#cus_list = list(range(n_y+1))
#stat_list = [0]+list(range(1001,1101))

def add_mins_to_time(timeval, mins_to_add):
    dummy_date = datetime.date(1, 1, 1)
    full_datetime = datetime.datetime.combine(dummy_date, timeval)
    added_datetime = full_datetime + datetime.timedelta(minutes=int(mins_to_add))
    if added_datetime.time() < datetime.time(8,0):
        return datetime.time(8,0).isoformat('minutes')
    return added_datetime.time().isoformat('minutes')

def time_diff_in_mins (start_tm,end_tm):
    dummy_date = datetime.date(1,1,1)
    full_start_tm = datetime.datetime.combine(dummy_date, start_tm)
    full_end_tm = datetime.datetime.combine(dummy_date, end_tm)
    time_diff = full_end_tm - full_start_tm
    return time_diff.seconds/60

class Source:
    vehs = None
    per_charge_cost = 50 # charge cost (given)
    wait_cost_perh = 24 # idle cost (given)
    T = 960 # available time (given)
    
    
    def __init__(self,dist_time,input_node,n):
        self.dist_arr = np.empty([n,n],dtype = int)
        self.time_arr = np.empty([n,n],dtype = int)

        with open(os.path.join(work_dir,dist_time)) as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                self.dist_arr[int(row['from_node'][-4:]),int(row['to_node'][-4:])] = row['distance']
                self.time_arr[int(row['from_node'][-4:]),int(row['to_node'][-4:])] = row['spend_tm']
        
        df = pd.read_excel(os.path.join(work_dir,input_node))
        self.weights = []
        self.volumes = []
        self.e_i = []
        self.l_i = []
        self.loc = []
        self.n = n
        self.n_2 = 1
        self.n_3 = 1
        
        for  row in df.itertuples():
            if row[2] == 2:
                self.n_2 += 1
                self.n_3 += 1
            elif row[2] == 3:
                self.n_3 += 1
            self.loc.append([row[3],row[4]])
            if row[5] != '-':
                self.weights.append(row[5])
            if row[6] != '-':
                self.volumes.append(row[6])
            if row[7] != '-' and row[8] != '-' and type(row[7]) is datetime.time and type(row[8]) is datetime.time:
                #time_ranges.append([time_diff_in_mins(row[7],datetime.time(8,0)),time_diff_in_mins(row[8],datetime.time(8,0))])
                self.e_i.append(time_diff_in_mins(datetime.time(8,0),row[7]))
                self.l_i.append(time_diff_in_mins(datetime.time(8,0),row[8]))
        if self.vehs is None:
            self.set_vehicle_type()
        
    def set_vehicle_type(self):
        df = pd.read_excel(os.path.join(work_dir,'input_vehicle_type.xlsx'),index_col = 0)
        self.vehs = df.rename(str.strip,axis = 'columns')

#src1 = Source('inputdistancetime_1_1601.txt','inputnode_1_1601.xlsx',1601)
#src2 = Source('inputdistancetime_2_1501.txt','inputnode_2_1501.xlsx',1501)
#src3 = Source('inputdistancetime_3_1401.txt','inputnode_3_1401.xlsx',1401)
#src4 = Source('inputdistancetime_4_1301.txt','inputnode_4_1301.xlsx',1301)
src5 = Source('inputdistancetime_5_1201.txt','inputnode_5_1201.xlsx',1201)



#%%

def save_matrix(src):
    save = np.zeros([src.n_3,src.n_3])
    for i in range(src.n_3):
        for j in range(src.n_3):
            if i != j:
                save[i,j] = src.dist_arr[0,i] + src.dist_arr[j,0] - src.dist_arr[i,j]
    src.save = save

def gen_csv(ans,out_filename):
    with open(os.path.join(work_dir,out_filename),'w', newline = '') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',',
                            quotechar='"', quoting=csv.QUOTE_MINIMAL)
        header = ['trans_code','vehicle_type','dist_seq','distribute_lea_tm','distribute_arr_tm','distance','trans_cost','charge_cost','wait_cost','fixed_use_cost','total_cost','charge_cnt']
        csvwriter.writerow(header)
        i = 0
        for item in ans:
            i += 1
            inline = []
            inline.append('DP'+str(i).zfill(4)) # Trans_code
            inline += item
            # Formating
            csvwriter.writerow(inline)
            
class Cluster:   
    def __init__(self,src,init_nodes):
        self.n = len(init_nodes)
        self.save_list = []
        self.paths = []
        self.weis_M = list(np.zeros(self.n))
        self.weis_S = list(np.zeros(self.n))
        self.weis_E = list(np.zeros(self.n))
        self.vols_M = list(np.zeros(self.n))
        self.vols_S = list(np.zeros(self.n))
        self.vols_E = list(np.zeros(self.n))
        self.src = src
        part_save = src.save[np.ix_(init_nodes,init_nodes)]
        for i in range(self.n):
            for j in range(self.n):
                if part_save[i,j] > 0:
                    self.save_list.append((part_save[i,j],init_nodes[i],init_nodes[j]))
        self.save_list.sort(key = lambda tup:tup[0])
        idx = 0
        for node in init_nodes:
            self.paths.append([node])
            self.weis_M[idx] = src.weights[node - 1]
            self.vols_M[idx] = src.volumes[node - 1]
            if node < src.n_2:
                self.weis_S[idx] = src.weights[node - 1]
                self.vols_S[idx] = src.volumes[node - 1]
            elif node < src.n_3:
                self.weis_E[idx] = src.weights[node - 1]
                self.vols_E[idx] = src.volumes[node - 1]
            idx += 1
            
    def closest_station(self,i,j,y):
        stat_lst = []
        for node in [0]+list(range(self.src.n_3,self.src.n)):
            if self.src.dist_arr[i,node] + y <= self.src.vehs['driving_range'][1]:
                stat_lst.append(node)
        if len(stat_lst) == 0:
            return False
        min_dist = 999999
        for node in stat_lst:
            dist = self.src.dist_arr[i,node] + self.src.dist_arr[node,j]
            if dist < min_dist:
                min_dist = dist
                idx = node
        return idx

    def total_dist(self,path):
        ttl_dist = 0
        for i in range(len(path) - 1):
            ttl_dist += self.src.dist_arr[int(path[i]),int(path[i+1])]
        return ttl_dist
    
    def check_VW(self,start,end):
        max_vols = max(self.vols_M[start]+self.vols_S[end],self.vols_E[start]+self.vols_S[end],self.vols_E[start]+self.vols_M[end])
        if max_vols > self.src.vehs['max_volume'][1]:
            return False
        max_weis = max(self.weis_M[start]+self.weis_S[end],self.weis_E[start]+self.weis_S[end],self.weis_E[start]+self.weis_M[end])
        if max_weis > self.src.vehs['max_weight'][1]:
            return False
        return True

    def check_dist(self,start,end):
        if self.total_dist([0]+self.paths[start]+self.paths[end]+[0]) > self.src.vehs['driving_range'][1]:
            l = self.closest_station(start,end,self.total_dist([0]+self.paths[start]))
            return l
        else:
            return "OK"
        
    def check_time(self,dist_seq):
        path = [0] + dist_seq + [0]
        curr_tm = 0
        for i in range(len(path) - 1):
            try:
                curr_tm += self.src.time_arr[int(path[i]),int(path[i+1])]
                if int(path[i + 1]) < self.src.n_3:
                    if self.src.l_i[int(path[i+1])] < curr_tm:
                        return False
                    if self.src.e_i[int(path[i+1])] > curr_tm:
                        curr_tm = self.src.e_i[int(path[i+1])]    
                if path[i+1] != 0:
                        curr_tm += 30
            except IndexError:
                print(dist_seq,i)
                raise
        return True

    def combine(self,start,end,l):
        self.vols_M[start] = max(self.vols_M[start]+self.vols_S[end],self.vols_E[start]+self.vols_S[end],self.vols_E[start]+self.vols_M[end])
        self.vols_M.pop(end)
        self.weis_M[start] = max(self.weis_M[start]+self.weis_S[end],self.weis_E[start]+self.weis_S[end],self.weis_E[start]+self.weis_M[end])
        self.weis_M.pop(end)
        self.vols_S[start] = self.vols_S[start] + self.vols_S[end]
        self.vols_S.pop(end)
        self.vols_E[start] = self.vols_E[start] + self.vols_E[end]
        self.vols_E.pop(end)
        self.weis_S[start] = self.weis_S[start] + self.weis_S[end]
        self.weis_S.pop(end)
        self.weis_E[start] = self.weis_E[start] + self.weis_E[end]
        self.weis_E.pop(end)
        if l != "OK":
            self.paths[start] = self.paths[start] + [l] + self.paths[end]
        else:
            self.paths[start] = self.paths[start] + self.paths[end]
        self.paths.pop(end)
    
    def get_next_path(self):
        try:
            max_path = self.save_list.pop()
            self.saved_dist = max_path[0]
            self.start_node = max_path[1]
            self.end_node = max_path[2]
            return True
        except IndexError:
            return False
    
    def transport_cost(self,veh_type , dist_seq):
        unit_cost = self.src.vehs['unit_trans_cost'][veh_type]
        trav_dist = 0
        path = dist_seq.split(';')
        for idx, val in enumerate(path):
            if idx + 1 < len(path):
                trav_dist += self.src.dist_arr[int(val),int(path[idx+1])]
        tran_cost = round(trav_dist * unit_cost / 1000,2)
        return [trav_dist,tran_cost]
        
    def charge_cost(self,dist_seq):
        path = dist_seq.split(';')
        charge_cnt = 0
        for x in path:
            if int(x) >= self.src.n_3:
                charge_cnt += 1
        charge_cost = charge_cnt * self.src.per_charge_cost 
        return [charge_cost, charge_cnt]

    def fixed_use_cost(self,veh_type):
        return self.src.vehs['vehicle_cost'][veh_type]

    def distribute_arr_tm(self,dist_seq, lea_tm):
        path = dist_seq.split(';')
        curr_tm = lea_tm
        wait_tm = 0
        for idx,val in enumerate(path):
            if idx + 1 < len(path):
                curr_tm += self.src.time_arr[int(val),int(path[idx+1])]
                if idx + 1 != len(path) - 1:
                    if int(path[idx + 1]) < self.src.n_3 and self.src.l_i[int(path[idx+1])] < curr_tm:
                        print(val,path[idx+1],curr_tm)
                        return False
                    if int(path[idx + 1]) < self.src.n_3 and self.src.e_i[int(path[idx+1])] > curr_tm:
                        wait_tm += (self.src.e_i[int(path[idx+1])] - curr_tm)
                        curr_tm = self.src.e_i[int(path[idx+1])]
                    curr_tm += 30
        wait_cost = wait_tm * self.src.wait_cost_perh / 60
        return [add_mins_to_time(datetime.time(8,0),curr_tm),wait_cost]

    def get_ans(self):
        ans = []
        for path in self.paths:
            veh_type = 1
            dist_seq = '0;' + ';'.join(str(node) for node in path) + ';0'
            try:
                lea_tm = max(self.src.e_i[int(path[0])] - self.src.time_arr[0][int(path[0])],0)
            except IndexError:
                print(path[0])
                lea_tm = 0
            tmp = [veh_type,dist_seq,add_mins_to_time(datetime.time(8,0),lea_tm)]
            tmp += self.distribute_arr_tm(dist_seq,lea_tm) # Distribute_arr_tm, Wait_cost
            tmp += self.transport_cost(veh_type,dist_seq) # Distance, Trans_cost
            tmp += self.charge_cost(dist_seq) # Charge_cost, Charge_cnt
            tmp.append(self.fixed_use_cost(veh_type)) # Fixed_cost
            tmp.append(round(tmp[4]+tmp[6]+tmp[7]+tmp[9],2))
            myorder = [0,1,2,3,5,6,7,4,9,10,8]
            tmp = [tmp[i] for i in myorder]
            ans.append(tmp)
        return ans
    
    def main(self):
        while self.get_next_path():
            i = 0
            start = -1
            end = -1
            for item in self.paths:
                if item[0] == self.end_node:
                    end = i
                if item[-1] == self.start_node:
                    start = i
                i += 1
            if start != -1 and end != -1:
                if self.check_VW(start,end):
                    l = self.check_dist(start,end) 
                    if l == False:
                        continue
                    elif l != "OK":
                        new_path = [0] + self.paths[start] + [l] + self.paths[end] + [0]
                    else:
                        new_path = [0] + self.paths[start] + self.paths[end] + [0]
                    if self.check_time(new_path):
                        self.combine(start,end,l)
#%%
save_matrix(src5)
clus5 = Cluster(src5,range(1,300))
clus5.main()
gen_csv(clus5.get_ans(),'result_tmp.csv')
            
