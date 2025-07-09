import torch

class ManifoldSculpting():
    ''' 
    Dependencies: import torch
    '''
    def __init__(self, k=5, n_dim=2, niter=100, sigma=0.98, patience=20): # rotate = True
        # Implementation of the Manifold Sculpting algorithm in PyTorch

        # Hyperparameters of the algorithm
        # Only torch.tensor()
        self.k = k                    # -- number of neighbors considered
        self.n_dim = n_dim            # -- dimension of the searched manifold
        self.niter = niter            # -- 1st stopping criterion for the iterative cicle 
        self.sigma = sigma            # -- scaling factor at each iteration (for extra dimensions)
        self.scale_factor = 1         # -- cumulative scale factor
        self.patience = patience      # -- 2nd stopping criterion

    def transform(self, data):
        ####### MAIN #######

        ## ---- Import
        # In case of images or weirdly shaped input points
        if len(data.size()) > 2:
            flatten   = torch.nn.Flatten()
            self.data = flatten(data) 
            # self.original_p_size = data.size[1:] # superfluo
            #print("Flattened")
        else:
            self.data = data
            
        self.p_size = self.data.size()[1]   # single point flattenend dimension 
        self.n_datapoints = data.size()[0]  # int

        ## ---- Compute neighbors relations
        self.dist, self.neighb    = self.neighb_distance()
        self.colinear, self.theta = self.colinear_neighb()
        print('''
        INFO: Neighbor relations computed
        ''')
        
        self.avg_dist = torch.mean(self.dist)

        self.nudge = self.avg_dist

        ## ---- PCA transform
        self.data = self.pca_transform()
        
        # Distinguish dimensions to be scaled/preserved
        self.preserv_dim = torch.tensor(list(range(self.n_dim)))
        self.scaled_dim  = torch.tensor(list(range(self.n_dim, self.p_size)))

        ## ---- Iterative transformation
        epoch = 1
        print('''
        INFO: Starting preliminary adjustments (NO error comparison)
        ''')
        
        # Adjust a bunch of times without comparing errors
        while self.scale_factor > 0.01: # can be tuned
            mean_error = self.step()
            epoch += 1

        epochs_since_improvement = 0
        best_error = torch.Tensor(float('inf'))
        print('''
        INFO: First round of adjustments finished 
        INFO: Start comparing errors (stopping criteria: niter, patience)
        ''')

        # Continue adjusting, start comparing errors
        while (epoch < self.niter) and (epochs_since_improvement < self.patience):
            mean_error = self.step()

            if mean_error < best_error:
                best_error = mean_error
                self.best_data  = torch.clone(self.data)
                self.best_error = best_error
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += 1
                
            epochs += 1
            if epochs%20:
                print(f'''
                INFO: Elapsed epochs: {epochs}''')

        ## DEBUG and monitoring
        self.elapsed_epochs = epoch
        self.last_error = mean_error



    def neighb_distance(self):
        '''
        Returns:
        - tensor.size() = (n_datapoints, k)
          (i,j) hosts the DISTANCE of point i to its j-th neighbor 
          with distances decreasing in j
        - tensor.size() = (n_datapoints, k) 
          (i,j) hosts the INDEX (relative to self.data) of the j-th neighbor of point i
        '''
        x2 = self.data * self.data
        x2 = x2.sum(axis=1)
        data_t = torch.transpose(self.data, 0, 1) # pb in dim >i (es foto...), credo funzioni cmq
        xx = self.data@data_t
        
        all_distances = torch.sqrt( x2 - 2*xx + x2.unsqueeze(dim=1) ) # they have different dimensions, we leverage pytorch default for the operations
        
        _, indices = torch.sort(all_distances)

        kneighb = indices[:, 1:self.k+1]
        #all_distances   = torch.zeros_like(kneighb)

        # Cherry-pick neighbors distances (sorted)
        d = []  # For some reason it doesn't allow directly with tensors
        for i in range(self.n_datapoints):
            #print(kneighb[i,:])
            #print(all_distances.size())
            d.append( all_distances[kneighb[i,:], i] )
        
        kdist = torch.reshape( torch.cat( d, 0 ), (-1, self.k) )
        
        return kdist, kneighb

    
    def avg_neighb_distance(self):
        dist, _ = self.neighb_distance()
        avg_dist = torch.mean(dist)
        
        return avg_dist

    def colinear_neighb(self):
        '''
        For clarity:
        Variables with _idx  have values in (0, m-1) (selecting points from self.data)
        Variables with _kidx have values in (0, k-1)

        Returns:
        - tensor.size() = (n_datapoints, k)
        (i,j) hosts the ANGLE theta (i-j-l) of the most colinear point l to the couple i-j
        - tensor.size() = (n_datapoints, k)
        (i,j) hosts the NEIGHBOR INDEX of l with respect to the neighbourhood of j
        '''
        theta    = torch.ones((self.n_datapoints,self.k))
        colinear = torch.ones((self.n_datapoints,self.k))
        
        # Loop over data points
        for i_idx in range(self.n_datapoints):
            # Loop over neighbors of i
            for j_kidx, j_idx in enumerate(self.neighb[i_idx]):
                
                p2j = self.data[i_idx,:] - self.data[j_idx,:]
                #print(self.data[i_idx,:])
                #print(p2j.size())
                #print(p2j)
                p2j /= torch.norm(p2j)
        
                colinear_kidx =  torch.ones(self.k)
                cos_i2l = torch.ones(self.k)
                # Loop over neighbor points of j
                for l_kidx, l_idx in enumerate(self.neighb[j_idx]):
                    
                    p2l = self.data[l_idx] - self.data[j_idx]
                    p2l /= torch.norm(p2l)
                    cos_i2l[l_kidx] = p2l@p2j
                    
                # Extract the colinear angle and neighbor
                cos_max, colinear[i_idx, j_kidx] = torch.max(cos_i2l, dim=0)
                colinear = colinear.int()
                theta[i_idx, j_kidx] = torch.acos(cos_max)

        return colinear, theta
    
            
    def pca_transform(self):
        '''
        Returns
        - tensor.size() = (n_datapoints, p_size)
          the data linearly transformed to the principal components basis
        '''
        cov = torch.cov(self.data)
        eigenval, eigenvec = torch.linalg.eigh(-cov) # -cov because it outputs sorted descending eigenval
        eigenval = -eigenval
        pca_data = eigenvec@self.data
        
        return pca_data
        
    
    def compute_error(self, p, visited):
        '''
        Parameters:
        - p : (int)         index of the point
        - visited: (list)   list of already adjusted points
        
        Returs:
        - (float) error relative to the neighbourhood of p
        '''
        
        w = torch.ones(self.k)
        for j in range(self.k):
            w[j] = 10 if self.neighb[p,j] in visited else 1
    
        total_err = 0
        for i in range(self.k):
            # Extract indices
            n = self.neighb[p,i].item()
            c = self.colinear[p,i].item()

            # Compute theta_p2c, the angle in p-n-c
            p2n = self.data[p] - self.data[n]
            c2n = self.data[c] - self.data[n]
            p2n /= torch.norm(p2n)
            c2n /= torch.norm(c2n)
            theta_p2c = torch.acos(p2n@c2n)

            # Compute error
            err_dist = .5*(torch.norm(p2n) - self.dist[p,i]) / self.avg_dist
            err_theta = (theta_p2c - self.theta[p,i])/3.1415926535
            
            total_err += w[i] * (err_dist*err_dist + err_theta*err_theta)
            
        return total_err
    
    
    def adjust_point(self, p, visited):
        '''
        Parameters:
        - p: (int) index of the point to be adjusted
        - visited: (list) list of already adjusted points in current epoch

        Returns:
        - number of hill descent steps
        - error for the adjusted point
        '''
        # Slightly randomize the entity of the update
        nudge = self.nudge * ( .6 + .4*torch.rand(1).item() ) # float
        
        s = -1 
        improved = True
        err = self.compute_error(p, visited)
        
        while (s<30) and improved:
            s += 1
            improved = False

            ## --- Downhill update
            # Loop over dimensions (try the same nudge for all of them)
            for d in self.preserv_dim:
                
                self.data[p,d] += nudge  # Try one direction
                new_err = self.compute_error(p, visited)

                if new_err >= err:
                    self.data[p,d] -= 2*nudge  # Try in the opposite
                    new_err = self.compute_error(p, visited)
                    
                if new_err >= err:
                    self.data[p,d] += nudge # Stay put
                    
                else:
                    err = new_err
                    improved = True
                    
        return s, err


    def step(self):
        '''
        Returns:
        - (float) mean error after adjustment
        '''
        # ---- a)
        # (a,b refer to pseudo-code (Fig 2.2) in original paper)
        
        self.scale_factor *= self.sigma
        
        # Downscale component along scaled dimensions
        self.data[:, self.scaled_dim] *= self.sigma
    
        # Upscale the component along preserved dimensions
        while (self.avg_neighb_distance() < self.avg_dist): # mi sfugge li senso di qsto criterio
            self.data[:, self.preserv_dim] /= self.sigma
            
        # ---- b)
        pr_idx = torch.multinomial(torch.ones(self.n_datapoints), num_samples=1)
        # pr = data[rand_indx,:]

        queue_idx = []
        queue_idx.append(pr_idx.item())
        visited = []
    
        step = 0
        mean_error = 0
        counter = 0
        
        # while the queue is not empty
        while queue_idx:
            p = queue_idx.pop(0)
            if p in visited:
                continue
    
            # Add p's neighbors in the queue
            for n in self.neighb[p]:
                queue_idx.append(n.item())
                
            s, err = self.adjust_point(p,visited) 
    
            step += s
            mean_error += err
            counter += 1
            visited.append(p)
    
        mean_error /= counter
    
        # numbers from author's implementation (weight decay-like)
        if step < self.n_datapoints:
            self.nudge *= 0.87
        else:
            self.nudge /= 0.91
            
        return mean_error
