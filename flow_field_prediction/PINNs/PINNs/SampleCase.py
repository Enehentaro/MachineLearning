"""  
made by Kataoka @2023/12/XX.  

"""  

import sys

import numpy as np


class CavitySampler():
    
    def __init__(
        self,
        num_sampling_residual = 10000,
        num_sampling_left_edge = 2500,
        num_sampling_right_edge = 2500,
        num_sampling_bottom_edge = 2500,
        num_sampling_top_edge = 2500,
        x_range = [-3, 14],
        y_range = [-3, 3],
        non_dimensionalization = True, 
        seed = None
    ):
        
        self.num_sampling_residual = num_sampling_residual
        self.num_sampling_left_edge = num_sampling_left_edge
        self.num_sampling_right_edge = num_sampling_right_edge
        self.num_sampling_bottom_edge = num_sampling_bottom_edge
        self.num_sampling_top_edge = num_sampling_top_edge
        
        self.x_range = x_range
        self.y_range = y_range

        self.non_dimensionalization = non_dimensionalization
        self.seed = seed


    def __call__(self):
        if self.seed==None:
            pass
        else:
            np.random.seed(self.seed)

        temp_x = np.full((self.num_sampling_left_edge, 1), self.x_range[0])
        temp_y = self.sample_from_line(self.num_sampling_left_edge, self.y_range)
        left = np.concatenate([temp_x, temp_y], axis=1)

        temp_x = np.full((self.num_sampling_right_edge, 1), self.x_range[1])
        temp_y = self.sample_from_line(self.num_sampling_right_edge, self.y_range)
        right = np.concatenate([temp_x, temp_y], axis=1)

        temp_x = self.sample_from_line(self.num_sampling_bottom_edge, self.y_range)
        temp_y = np.full((self.num_sampling_bottom_edge, 1), self.y_range[0])
        bottom = np.concatenate([temp_x, temp_y], axis=1)

        temp_x = self.sample_from_line(self.num_sampling_top_edge, self.y_range)
        temp_y = np.full((self.num_sampling_top_edge, 1), self.y_range[1])
        top = np.concatenate([temp_x, temp_y], axis=1)

        temp = self.sample_from_square(self.num_sampling_residual, self.x_range, self.y_range)
        residual = np.concatenate([temp, left, right, bottom, top], axis=0)

        keys = ["residual", "left_edge", "right_edge", "bottom_edge", "top_edge"]
        values = [residual, left, right, bottom, top]

        point_dict = dict(zip(keys, values))

        if self.seed==None:
            pass
        else:
            self.seed += 1

        return point_dict


    @classmethod
    def sample_from_line(
        cls,
        num_sampling, 
        x_range
    ):
        points_array = np.random.rand(num_sampling, 1)
        
        size_array = np.concatenate([np.full((num_sampling, 1), x_range[1]-x_range[0])], axis=1)
        min_array = np.concatenate([np.full((num_sampling, 1), x_range[0])], axis=1)

        return points_array * size_array - min_array

        
    @classmethod
    def sample_from_square(
        cls, 
        num_sampling, 
        x_range, 
        y_range
    ):

        points_array = np.random.rand(num_sampling, 2)
        
        size_array = np.concatenate([np.full((num_sampling, 1), x_range[1]-x_range[0]), np.full((num_sampling, 1), y_range[1]-y_range[0])], axis=1)
        min_array = np.concatenate([np.full((num_sampling, 1), x_range[0]), np.full((num_sampling, 1), y_range[0])], axis=1)

        return points_array * size_array - min_array


class CylinderSampler():
    
    def __init__(
        self,
        num_sampling_residual = 10000,
        num_sampling_left_edge = 2500,
        num_sampling_right_edge = 2500,
        num_sampling_bottom_edge = 2500,
        num_sampling_top_edge = 2500,
        num_sampling_cylinder_surface = 5000,
        x_range = [-3, 14],
        y_range = [-3, 3],
        cylinder_radius = 0.5,
        non_dimensionalization = True, 
        seed = None
    ):
        
        self.num_sampling_residual = num_sampling_residual
        self.num_sampling_left_edge = num_sampling_left_edge
        self.num_sampling_right_edge = num_sampling_right_edge
        self.num_sampling_bottom_edge = num_sampling_bottom_edge
        self.num_sampling_top_edge = num_sampling_top_edge
        self.num_sampling_cylinder_surface = num_sampling_cylinder_surface
        
        self.x_range = x_range
        self.y_range = y_range

        self.cylinder_radius = cylinder_radius
        self.non_dimensionalization = non_dimensionalization
        self.seed = seed


    def __call__(self):
        if self.seed==None:
            pass
        else:
            np.random.seed(self.seed)

        temp_x = np.full((self.num_sampling_left_edge, 1), self.x_range[0])
        temp_y = self.sample_from_line(self.num_sampling_left_edge, self.y_range)
        left = np.concatenate([temp_x, temp_y], axis=1)

        temp_x = np.full((self.num_sampling_right_edge, 1), self.x_range[1])
        temp_y = self.sample_from_line(self.num_sampling_right_edge, self.y_range)
        right = np.concatenate([temp_x, temp_y], axis=1)

        temp_x = self.sample_from_line(self.num_sampling_bottom_edge, self.y_range)
        temp_y = np.full((self.num_sampling_bottom_edge, 1), self.y_range[0])
        bottom = np.concatenate([temp_x, temp_y], axis=1)

        temp_x = self.sample_from_line(self.num_sampling_top_edge, self.y_range)
        temp_y = np.full((self.num_sampling_top_edge, 1), self.y_range[1])
        top = np.concatenate([temp_x, temp_y], axis=1)

        surface = self.sample_from_circle_edge(self.num_sampling_cylinder_surface, self.cylinder_radius)

        temp = self.sample_from_square_except_circle(self.num_sampling_residual, self.x_range, self.y_range, self.cylinder_radius)
        residual = np.concatenate([temp, left, right, bottom, top, surface], axis=0)

        keys = ["residual", "left_edge", "right_edge", "bottom_edge", "top_edge", "surface"]
        values = [residual, left, right, bottom, top, surface]

        point_dict = dict(zip(keys, values))

        if self.seed==None:
            pass
        else:
            self.seed += 1

        return point_dict


    @classmethod
    def sample_from_line(
        cls,
        num_sampling, 
        x_range
    ):
        points_array = np.random.rand(num_sampling, 1)
        
        size_array = np.concatenate([np.full((num_sampling, 1), x_range[1]-x_range[0])], axis=1)
        min_array = np.concatenate([np.full((num_sampling, 1), x_range[0])], axis=1)

        return points_array * size_array - min_array
    

    @classmethod
    def sample_from_circle_edge(
        cls,
        num_sampling,
        cylinder_radius
    ):
        points_array = np.random.rand(num_sampling, 1)

        x_array = cylinder_radius * np.cos(np.pi * points_array)
        y_array = cylinder_radius * np.sin(np.pi * points_array)

        output_array = np.concatenate([x_array.reshape(-1,1), y_array.reshape(-1,1)], axis=1)

        return output_array
        
    
    @classmethod
    def sample_from_square_except_circle(
        cls,
        num_sampling, 
        x_range,
        y_range, 
        circle_radius
    ):

        points_array = np.empty(shape=(0,2))
        
        for i in range(2*num_sampling):

            candidate = np.random.rand(1,2)
            candidate[0][0] = (x_range[1]-x_range[0]) * candidate[0][0] - x_range[0]
            candidate[0][1] = (y_range[1]-y_range[0]) * candidate[0][1] - y_range[0]
        
            if np.abs(candidate[0][0])<circle_radius:

                boundary_value = np.sqrt(circle_radius**2 - candidate[0][0]**2)
                print(boundary_value)

                if np.abs(candidate[0][1])<boundary_value:
                    continue
                else:
                    points_array = np.concatenate([points_array, candidate], axis=0)

            else:
                points_array = np.concatenate([points_array, candidate], axis=0)

            if len(points_array)==num_sampling:
                break

        if len(points_array)>num_sampling or len(points_array)<num_sampling:
            print("failed to sample points.")
            sys.exit()

        return points_array


