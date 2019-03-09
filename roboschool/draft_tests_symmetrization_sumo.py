'''
        #test pG in 0:
        p_GinG = self.cpp_robots[0].root_part.pose()
        p_GinG.set_xyz(1,0,0)
        p_GinG.set_rpy(0,0,0)
        p_Gin0 = self._transform_position_of_fromGto_(p_GinG, 0)
        print(self.init_matrices[0])
        print('--'*20)
        print(p_Gin0)
        
        print('--'*20)
        print('--'*20)
        
        #test pG in 1:
        p_GinG = self.cpp_robots[0].root_part.pose()
        p_GinG.set_xyz(1,0,0)
        p_GinG.set_rpy(0,0,0)
        p_Gin1 = self._transform_position_of_fromGto_(p_GinG, 1)
        print(self.init_matrices[1])
        print('--'*20)
        print(p_Gin1)
        
        '''

        '''
        #test q0 in 1:
        p_0inG = self.cpp_robots[0].root_part.pose()
        p_0inG.set_xyz(-1,0,0)
        p_0inG.rotate_z(np.pi)
        q_0in1 = self._transform_orientation_of_fromGto_(p_0inG, 1)
        print(q_0in1)
        
        p_1inG = self.cpp_robots[1].root_part.pose()
        p_1inG.set_xyz(1,0,0)
        p_1inG.rotate_z(-np.pi)
        q_1in0 = self._transform_orientation_of_fromGto_(p_1inG, 0)
        print(q_1in0)
        '''