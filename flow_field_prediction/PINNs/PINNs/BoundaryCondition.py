"""  
made by Kataoka @2023/12/XX.  

"""  


class BoundaryCondition():
    
    def __init__(self, point_type, condition=None, value=None):

        self.point_type = point_type
        self.condition = condition
        self.value = value

    
    def values(self):

        return self.value