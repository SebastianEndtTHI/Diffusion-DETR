import numpy as np

class DTIGenerator():
    def __init__(self, args):
        super().__init__()

        # load diffusion protocol
        self.direction_table = np.loadtxt(args.diff_file, delimiter=",")

        # compartment parameters
        self.dist = args.dist

        # noise level
        self.noise_level = getattr(args, 'noise_level', 0.01)



    def get_md_fa_values(self, sampling_method='global_uniform'):

        if sampling_method == 'global_uniform':
            # sample over the whole MD and FA range, ignoring tissue-specific distributions
            md = np.random.uniform(0, 4)  # overall range of MD values in brain tissue
            fa = np.random.uniform(0, 1)       # FA values are always between 0 and 1
        
        else:
            raise ValueError("Invalid sampling method. Choose 'uniform' or 'normal'.")

        # ensure MD is positive
        md = max(md, 0.01)  # minimum MD to avoid non-physical values

        return md, fa
    


    def compute_eigenvalues_from_md_fa(self, md, fa):
        """
        Compute eigenvalues from MD and FA assuming lmd2 = lmd3.
        
        Given:
        - MD = (lmd1 + lmd2 + lmd3) / 3 = (lmd1 + 2*lmd2) / 3
        - FA² = (lmd1 - lmd2)² / (lmd1² + 2*lmd2²)
        
        Solve for lmd1 and lmd2.
        """
        
        # Use quadratic formula to solve for lmd2/MD ratio
        # FA² = 9(1 - x)² / (9 - 12x + 6x²) where x = lmd2/MD
        # Rearranged: (6FA² - 9)x² + (-12FA² + 18)x + (9FA² - 9) = 0
        
        fa_sq = fa ** 2
        
        a = 6 * fa_sq - 9
        b = -12 * fa_sq + 18
        c = 9 * fa_sq - 9
        
        # Handle edge case: if FA is very close to 0, lmd1 ≈ lmd2 ≈ lmd3 ≈ MD
        if abs(fa) < 1e-6:
            lmd1 = md
            lmd2 = md
            lmd3 = md
            return lmd1, lmd2, lmd3
        
        # Handle edge case: if FA is very close to 1
        if abs(fa - 1.0) < 1e-6:
            # In this case, lmd2 ≈ lmd3 ≈ 0, and lmd1 ≈ 3*MD
            lmd1 = 3 * md
            lmd2 = 0.01  # small positive value
            lmd3 = 0.01
            return lmd1, lmd2, lmd3
        
        # Solve quadratic equation
        discriminant = b ** 2 - 4 * a * c
        
        if discriminant < 0:
            # If no real solution, approximate with closest valid values
            # This shouldn't happen with valid FA values, but handle it just in case
            lmd2 = md
            lmd1 = md
            lmd3 = md
        else:
            # Two solutions, choose the one that gives lmd1 > lmd2
            x1 = (-b + np.sqrt(discriminant)) / (2 * a)
            x2 = (-b - np.sqrt(discriminant)) / (2 * a)
            
            # Calculate corresponding lmd2 values
            lmd2_opt1 = x1 * md
            lmd2_opt2 = x2 * md
            
            # Calculate corresponding lmd1 values
            lmd1_opt1 = 3 * md - 2 * lmd2_opt1
            lmd1_opt2 = 3 * md - 2 * lmd2_opt2
            
            # Choose solution where lmd1 >= lmd2 >= 0
            if lmd1_opt1 >= lmd2_opt1 >= 0:
                lmd1 = lmd1_opt1
                lmd2 = lmd2_opt1
            elif lmd1_opt2 >= lmd2_opt2 >= 0:
                lmd1 = lmd1_opt2
                lmd2 = lmd2_opt2
            else:
                # Fallback: use the solution with larger lmd1
                if lmd1_opt1 > lmd1_opt2:
                    lmd1 = max(lmd1_opt1, 0)
                    lmd2 = max(lmd2_opt1, 0)
                else:
                    lmd1 = max(lmd1_opt2, 0)
                    lmd2 = max(lmd2_opt2, 0)
        
        lmd3 = lmd2  # assumption: lmd2 = lmd3
        
        return lmd1, lmd2, lmd3



    def get_eigenvalue_matrix(self, ev1, ev2, ev3):

        # sort eigenvalues
        lmds = np.sort([ev1,ev2,ev3])

        # matrix with diagonal eigenvalues, normed to realistic value range
        lmd_mtrx = np.diag(np.array([lmds[2], lmds[1], lmds[0]])) * 1e-3

        return lmd_mtrx



    def get_rotation_matrix(self, r):

        # generate random vector, non-parallel to r
        rand_vec = np.random.randn(3)
        while np.allclose(np.cross(r, rand_vec), 0):
            rand_vec = np.random.randn(3)

        
        # orthogonal vectors returned by cross product, normed on unit sphere
        v2 = np.cross(r, rand_vec)
        v2 /= np.linalg.norm(v2)

        v3 = np.cross(r, v2)
        v3 /= np.linalg.norm(v3)

        
        # orthogonal vectors build orientation matrix Q
        Q = np.column_stack((r, v2, v3))

        return Q
    


    def get_diffusionTensor(self, eigenmatrix, Q):

        # computing diffusion tensor with eigenmatrix and rotation matrix for direction
        D = Q @ eigenmatrix @ Q.T

        return D



    def compute_signal_and_spectrogram(self, D, b_value, r):

        # simulating diffusion signals with generated diffusion tensor and directions form a diffusion protocoll
        rDr = r.T @ D @ r 

        signals = np.exp(-b_value * rDr)

        return signals
    


    def get_multiKompSignal(self, comp_signals):

        comp_count = np.size(comp_signals, axis=0)

        min_weight = 0.05  # 5% minimum limit
        if comp_count * min_weight > 1.0:
            raise ValueError("The minimum weights add up to more than 1.0!")

        # minimum weight for each compartment subtracted, then the remaining "free weight" is distributed randomly
        free_weight_total = 1.0 - (comp_count * min_weight)

        # random distribution of free weight with sorted random borders to ensure weights sum up to 1
        weight_borders = np.random.uniform(0, 1, comp_count-1)
        weight_areas = np.sort(np.append(weight_borders, [0, 1]))
        

        # random weighting for every compartment signal
        weights = []
        weighted_signals = []

        for t_idx in range(comp_count):
            # area/weight caculated as two adjacent border values in sorted random borders
            random_piece = weight_areas[t_idx+1] - weight_areas[t_idx]
            
            # scaled to the "free weight" size, then the 5% base added
            weight = min_weight + (random_piece * free_weight_total)
            weights.append(weight)


            # weighted signal for every single compartment
            weighted_comp = np.multiply(weight, comp_signals[t_idx])
            weighted_signals.append(weighted_comp)


        # summed up weighted compartment signals build a voxel signal
        voxel_signals = np.sum(weighted_signals, axis=0)

        return voxel_signals, weights
    



    def generate_Voxel(self, n_comp:int):

        # generate voxel signal and compartment metrics 
        dir_list = []
        D_list = []
        signal_list = []
        md_list = []
        fa_list = []

        
        # for each compartment in one voxel
        for c in range(n_comp):

            # generate MD and FA from defined distribution
            md, fa = self.get_md_fa_values(self.dist)

            # calculate three eigenvalues from MD and FA (with lmd2 = lmd3)
            lmd1, lmd2, lmd3 = self.compute_eigenvalues_from_md_fa(md, fa)

            # define diagonal eigenvalue matrix
            eig_matrix = self.get_eigenvalue_matrix(lmd1, lmd2, lmd3)


            # generate continuous directions
            # random vector with x-coord >= 0
            direction = np.random.randn(3)

            while direction[0]<0:
                direction = np.random.randn(3)

            # norm direction on unit sphere
            direction /= np.linalg.norm(direction) 

            # define orientation matrix Q
            Q = self.get_rotation_matrix(direction)

            # save normed directions for ground truth
            dir_list.append(direction)


            # compute diffusion tensor 
            D_comp = self.get_diffusionTensor(eig_matrix, Q)
            D_list.append(D_comp)


            # signal simulation
            comp_signals = []

            for b in self.direction_table:

                # simulate signal for every diffusion protocol entry 
                b_signal = self.compute_signal_and_spectrogram(D_comp, b[0], b[1:])
                comp_signals.append(b_signal)

            # save compartment signals
            signal_list.append(comp_signals)


            # save MD and FA values (already sampled, no need to recalculate)
            md_list.append(md)
            fa_list.append(fa)


        # sum of random weighted compartment signals build the complete voxel signal
        signals, weight_list = self.get_multiKompSignal(signal_list)  

        # add gaussian noise to voxel signal
        noise = np.random.normal(0, self.noise_level, signals.shape)
        signals += noise  
        

        # return voxel signal and metrics of each compartment
        return signals, md_list, fa_list, dir_list, weight_list
