"""  
made by Kataoka @2023/12/XX.  

"""  

import sys
import decimal

import numpy as np


class ResidualSampler():

    def __init__(self, dim, dt, seq_dt, time_range):
        
        self.dim = dim
        
        self.dt = dt
        self.seq_dt = seq_dt
        self.start_time = time_range[0]
        self.end_time = time_range[1]


    def sample_residual_points(self, sampler, params={}, method="sequence", same_sampling=False):

        if method=="sequence":
            output_dict = self.sequence_generator(sampler, params, same_sampling)
        else:
            pass

        return output_dict


    def sequence_generator(self, sampler, params, same_sampling):

        et = decimal.Decimal(str(self.end_time))
        st = decimal.Decimal(str(self.start_time))
        sdt = decimal.Decimal(str(self.seq_dt))

        if (et-st)/sdt%1!=0:
            print("time range is not indivisible by sequence time.")
            sys.exit()

        sdt = decimal.Decimal(str(self.seq_dt))
        dt = decimal.Decimal(str(self.dt))

        if (sdt/dt)%1!=0:
            print("sequence time is not indivisible by delta-t.")
            sys.exit()

        num_sequence = int((et-st)/sdt)
        step_per_sequence = int(sdt/dt)

        points_dict = {}

        for sequence in range(num_sequence):

            points_dict_temp = {}

            for step in range(step_per_sequence + 1):

                current_step = sequence*step_per_sequence + step
                current_time = current_step * self.dt

                if same_sampling==False:
                    points_dict_sampled = sampler(**params)
                else:
                    if points_dict_sampled:
                        pass
                    else:
                        points_dict_sampled = sampler(**params)

                if step==0:

                    coordinate_array = points_dict_sampled["residual"]
                    time_array = np.full((len(coordinate_array),1), current_time)
                    points_dict_temp["initial"] = np.concatenate([coordinate_array, time_array], axis=1)

                    # points_dict_temp["initial"] = np.empty(shape=(0,3))

                    # for key in list(points_dict_sampled.keys()):

                    #     coordinate_array = points_dict_sampled[key]
                    #     time_array = np.full((len(coordinate_array),1), current_time)
                    #     temp_array = np.concatenate([coordinate_array, time_array], axis=1)
                    #     points_dict_temp["initial"] = np.concatenate([points_dict_temp["initial"], temp_array], axis=0)

                elif step==1:

                    for key in list(points_dict_sampled.keys()):

                        coordinate_array = points_dict_sampled[key]
                        time_array = np.full((len(coordinate_array),1), current_time)
                        points_dict_temp[key] = np.concatenate([coordinate_array, time_array], axis=1)
                
                else:

                    for key in list(points_dict_sampled.keys()):

                        coordinate_array = points_dict_sampled[key]
                        time_array = np.full((len(coordinate_array),1), current_time)
                        append_array = np.concatenate([coordinate_array, time_array], axis=1)
                        points_dict_temp[key] = np.concatenate([points_dict_temp[key], append_array], axis=0)

            if sequence==0:

                for key in list(points_dict_temp.keys()):
                    points_dict[key] = points_dict_temp[key].reshape(1, -1, self.dim+1)

            else:

                for key in list(points_dict.keys()):
                    append_array = points_dict_temp[key].reshape(1, -1, self.dim+1)
                    points_dict[key] = np.concatenate([points_dict[key], append_array], axis=0)

        points_dict["residual"] = np.concatenate([points_dict["initial"], points_dict["residual"]], axis=1)
                    
        return points_dict