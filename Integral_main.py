from threading import Thread
import numpy as np
import math
import os
import traceback

import rs_method
import matplotlib.pyplot as plt
import control as ctrl

#
# G(s) =  a1 / (s+a0)
#

class ModelIdentification(object):

    def __init__(self):
        # data
        self.clear_data_list()
        self.T = 0.01
        self.search_style = 1
        self.start_ind = 30
        self.data_num = 400
        self.end_ind = -1
        self.pedal_input = 0
        
        self.dict_a0 = {}
        return

    def clear_data_list(self):
        self.time_list = []
        self.speed_list = []
        self.acc_list = []
        self.pedal_cmd_list = []
        self.pedal_fb_list = []
        return

    def extract_file(self,file):
        if os.path.exists(file):
            with open(file,'r') as f:
                count = 0
                time_start = 0
                for line in f:
                    if count < 1:
                        count = count + 1
                        continue

                    line_sp = line.split(',')
                    if count == 1:
                        time_start = float(line_sp[0])
                        count = count + 1
                    timestamp = float(line_sp[0]) - time_start
                    speed = float(line_sp[1])
                    acc = float(line_sp[2]) + float(line_sp[3])
                    pedal_cmd = float(line_sp[4])
                    pedal_fb = float(line_sp[5])
                    self.time_list.append(timestamp)
                    self.speed_list.append(speed)
                    self.acc_list.append(acc)
                    self.pedal_cmd_list.append(pedal_cmd)
                    self.pedal_fb_list.append(pedal_fb)
        return

    def get_psi_y(self):
        psi_y = []
        for acc in self.acc_list[self.start_ind:self.end_ind]:
            if not psi_y:
                acc_integ = 0
            else:
                acc_integ = psi_y[-1] + acc * self.T
            psi_y.append(acc_integ)
        
        psi_y = [-e for e in psi_y]
        return psi_y
    
    def get_psi_u(self):
        psi_u = []
        for pedal_cmd,i in zip(self.pedal_cmd_list[self.start_ind:self.end_ind],\
                            np.arange(self.start_ind,self.end_ind)):

            if not psi_u:
                pedal_integ = 0
                psi_u.append(pedal_integ)
                continue

            if i <= self.start_ind:
                pedal_integ = psi_u[-1]
            else:
                pedal_integ = pedal_cmd * self.T*(i-self.start_ind)
            psi_u.append(pedal_integ)
        return psi_u
    
    def update_psi(self):
        y = self.acc_list[self.start_ind:self.end_ind]
        y = np.array([y])
        psi_y = self.get_psi_y()
        psi_u = self.get_psi_u()
        psi = np.array([psi_y,psi_u])

        print('y=',np.shape(y))
        print('psi_y=',np.shape(psi_y))
        print('psi_u=',np.shape(psi_u))
        print('psi=',np.shape(psi))

        data = np.concatenate((y.T,psi.T),axis=1)

        return data,y,psi

    def ls_method(self,data):
        featureNum = 2
        initialTheta = 0.5 * np.ones((featureNum, 1))
        Theta,theta_array = rs_method.RLS_Fun(data, initialTheta, featureNum)
        return Theta,theta_array

    def show_theta(self,theta_array):
        plt.figure()
        lengend_list = []
        for i in np.arange(len(theta_array)):
            plt.plot(theta_array[i])
            lengend_list.append("theta_"+str(i))
        plt.legend(lengend_list)
        plt.grid()
        return

    def show_origin(self):
        plt.figure()
        ax1 = plt.subplot(3,1,1)
        plt.title('speed')
        plt.plot(self.time_list,self.speed_list,label='speed')
        plt.legend()
        plt.grid()

        ax2 = plt.subplot(3,1,2,sharex=ax1)
        plt.title('acc')
        plt.plot(self.time_list,self.acc_list,label='acc')
        plt.legend()
        plt.grid()

        ax3 = plt.subplot(3,1,3,sharex=ax2)
        plt.title('pedal')
        plt.plot(self.time_list,self.pedal_cmd_list,label='pedal_cmd')
        plt.plot(self.time_list,self.pedal_fb_list,label='pedal_fb')
        plt.legend()
        plt.grid()

    def show_respond(self):
        a0 = self.Theta[0][0]
        a1 = self.Theta[1][0]
        pedal = 'pedal:'+str(self.pedal_input)
        a0_str = 'a0:'+str(round(self.Theta[0][0],2))
        a1_str = 'a1:'+str(round(self.Theta[1][0],2))
        title = pedal+' '+a0_str+' '+a1_str

        s = ctrl.tf('s')
        sys = a1 / (s + a0)
        iosys = ctrl.tf2io(sys)
        T = []
        for i in np.arange(self.end_ind-self.start_ind):
            T.append(i*0.01)
        U = self.pedal_cmd_list[self.start_ind:self.end_ind]
        T = np.array(T)

        X0 = self.acc_list[self.start_ind]
        t, response = ctrl.input_output_response(iosys,T,U,X0)

        plt.figure()
        plt.subplot(211)
        plt.title(title)
        plt.plot(t, response,label='fit_data')
        t = self.time_list[self.start_ind:self.end_ind]
        t = [ti-t[0] for ti in t]
        acc = self.acc_list[self.start_ind:self.end_ind]
        plt.plot(t,acc,label='train_data')
        plt.grid()
        plt.legend()
        
        plt.subplot(212)
        plt.plot(t,U,label='pedal_cmd')
        plt.grid()
        plt.legend()

        return

    def data_show(self,theta_array):
        self.show_theta(theta_array)
        self.show_origin()
        self.show_respond()
        return

    # search_style: 0-manu，1-acc<0
    def get_start_ind(self,search_style=0):

        if search_style == 1:
            for i in np.arange(len(self.acc_list)):
                if self.acc_list[i] < 0.2:
                    self.start_ind = i
                    break
        elif search_style == 0:
            self.start_ind = self.start_ind
        else:
            print('search style error!')
        self.end_ind = self.start_ind + self.data_num
        size = len(self.time_list)
        if self.end_ind >= size:
            self.end_ind = size-1

        return

    def run(self,file):
        file_split = file.split('_')
        self.pedal_input = float(file_split[1])
        self.extract_file('./data/'+file)
        # get start id with acc < 0
        self.get_start_ind(search_style=self.search_style)
        # update PSI
        data,y,psi = self.update_psi()
        # LS
        self.Theta,self.theta_array = self.ls_method(data)
        print(self.Theta)

        if self.pedal_input not in self.dict_a0:
            self.dict_a0[self.pedal_input] = round(self.Theta[0][0],2)

        # print(theta_array)
        # Show
        self.data_show(self.theta_array)
        return

def auto_save_fig():
    figures_root = './figures_int'
    if not os.path.exists(figures_root):
        os.makedirs(figures_root)
    identifier = ModelIdentification()
    file_list = os.listdir("./data")
    for file in file_list:
        identifier.clear_data_list()
        if '.csv' not in file or 'pedal' not in file or 'speed' not in file:
            continue
        pedal_str = file.split('.')[0].split('_')[1]
        print('\n\npedal_str=',pedal_str)
        identifier.run(file)
        plt.savefig(figures_root+'/'+pedal_str+'.png')
    print('dict_a0=')
    print(sorted(identifier.dict_a0))
    a0_l = []
    tau_l = []
    for key in sorted(identifier.dict_a0):
        print(round(1.0/identifier.dict_a0[key],3))
        a0_l.append(identifier.dict_a0[key])
        tau_l.append(1/identifier.dict_a0[key])
    plt.figure()
    plt.title('a0')
    plt.plot(sorted(identifier.dict_a0),a0_l)
    plt.grid()
    plt.savefig(figures_root+'/'+'a0.png')

    return

def main():
    identifier = ModelIdentification()
    identifier.run('pedal_-20.0_speed_4.0.csv')
    
    plt.show()

if __name__ == '__main__':
    auto_save_fig()

    

    