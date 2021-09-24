import scipy as sp
import numpy as np
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import spsolve
from scipy.linalg import det, eigh, qr
from scipy.special import comb
import pickle

from suftware.src import utils

# Import error handling
#from utils import ControlledError
from suftware.src.utils import ControlledError


# Class container for Laplacian operators. Constructor computes spectrum.
class Laplacian:
    """
    Class container for Laplacian operators. Constructor computes specturm.

    Methods:
        get_G(): 
            Returns the (total) number of gridpoints

        get_kernel_dim(): 
            Returns the dimension of the kernel

        get_dense_matrix(): 
            Returns a dense scipy matrix of the operator

        get_sparse_matrix(): 
            Returns a scipy.sparse csr matrix of the operator

        save_to_file(filename):
            Pickles instance of class and saves to disk.
    """

    def __init__(self, operator_type, operator_order, num_gridpoints, grid_spacing=1.0 ):
        """
        Constructor for Smoothness_operator class

        Args:
            operator_type (str): 
                The type of operator. Accepts one of the following values:
                    '1d_bilateral'
                    '1d_periodic'
                    '2d_bilateral'
                    '2d_periodic'

            operator_order (int): 
                The order of the operator.

            num_gridpoints: 
                The number of gridpoints in each dimension of the domain.
        """
        
        # Make sure grid_spacing is valid
        if not isinstance(grid_spacing, float):
            raise ControlledError('/Laplacian/ grid_spacing must be a float: grid_spacing = %s' % type(grid_spacing))
        if not (grid_spacing > 0):
            raise ControlledError('/Laplacian/ grid_spacing must be > 0: grid_spacing = %s' % grid_spacing)
        
        if '1d' in operator_type:
            self._coordinate_dim = 1

            # Make sure operator_type is valid
            if operator_type == '1d_bilateral':
                periodic = False
            elif operator_type == '1d_periodic':
                periodic = True
            else:
                raise ControlledError('/Laplacian/ Cannot identify operator_type: operator_type = %s' % operator_type)
                
            self._type = operator_type
            
            self._sparse_matrix, self._kernel_basis = \
                laplacian_1d(num_gridpoints, operator_order, grid_spacing, periodic)
            
            self._G = self._kernel_basis.shape[0]
            self._kernel_dim = self._kernel_basis.shape[1]
            self._alpha = operator_order

        elif '2d' in operator_type:
            self._coordinate_dim = 2

            assert( len(num_gridpoints)==2 )
            assert( all([isinstance(n,utils.NUMBER) for n in num_gridpoints]) )

            assert( len(grid_spacing)==2 )
            assert( all([isinstance(n,utils.NUMBER) for n in grid_spacing]) )

            if operator_type == '2d_bilateral':
                periodic = False
            elif operator_type == '2d_periodic':
                periodic = True
            else:
                raise ControlledError('ERROR: cannot identify operator_type.')

            
            self._type = operator_type
            
            self._sparse_matrix, self._kernel_basis = \
                laplacian_2d( num_gridpoints, 
                              operator_order, 
                              grid_spacing, 
                              periodic=periodic, 
                              sparse=True,
                              report_kernel=True)

            self._Gx = int(num_gridpoints[0])
            self._Gy = int(num_gridpoints[1])
            self._G = self._Gx * self._Gy
            self._alpha = operator_order
            assert( self._G == self._kernel_basis.shape[0] )
            self._kernel_dim = self._kernel_basis.shape[1]

        else:
            raise ControlledError('/Laplacian/ Cannot identify operator_type: operator_type = %s' % operator_type)

        # Compute spectrum, and set lowest rank eigenvectors as kernel
        self._dense_matrix = self._sparse_matrix.todense()
        eigenvalues, eigenvectors = eigh(self._dense_matrix)
        self._eigenvalues = eigenvalues
        self._eigenbasis = utils.normalize(eigenvectors)
        #self._kernel_basis = self._eigenbasis[:,:self._kernel_dim]

        # Set kernel eigenvalues and eigenvectors
        self._eigenvalues[:self._kernel_dim] = 0.0
        self._eigenbasis[:,:self._kernel_dim] = self._kernel_basis

    def get_G(self):
        """ Return the total number of gridpoints used by this operator. """
        return self._G
    
    def get_kernel_basis(self):
        """ Returns the kernel as a kernel_dim x G numpy array """
        return sp.copy(self._kernel_basis)

    def get_kernel_dim(self):
        """ Return the dimension of the kernel of this operator. """
        return self._kernel_dim
    
    def get_sparse_matrix(self):
        """ Return a sparse matrix version of this operator. """
        return self._sparse_matrix

    def get_sparse_Lambda(self):
        """ Return a sparse matrix version of Lambda. """
        return self._sparse_matrix

    def get_dense_matrix(self):
        """ Return a dense matrix version of this operator. """
        return self._sparse_matrix.todense()

    def get_dense_Lambda(self):
        """ Return a dense matrix version of Lambda. """
        return self._sparse_matrix.todense()

    # def save(self, filename):
    #     """ Saves the current Laplacian in a way that can be recovered """
    #     pickle.dump(self, file(filename, 'w'))


# # Function for loading Laplacian from file
# def load(filename):
#     """ Loads a picked Laplacian from a file, and returns instance. """
#     operator = pickle.load(file(filename))
#     return operator


def derivative_matrix_1d(G, grid_spacing):
    """ Returns a (G-1) x G sized 1d derivative matrix. """
    
    # Create matrix
    tmp_mat = sp.diag(sp.ones(G),0) + sp.diag(-1.0*sp.ones(G-1),-1)
    right_partial = tmp_mat[1:,:]/grid_spacing
    
    return sp.mat(right_partial)


def laplacian_1d(G, alpha, grid_spacing, periodic, sparse=True, report_kernel=True):
    """ Returns a G x G sized 1d bilateral laplacian matrix of order alpha """

    # Make sure sparse is valid
    if not isinstance(sparse, bool):
        raise ControlledError('/laplacian_1d/ sparse must be a boolean: sparse = %s' % type(sparse))
    # Make sure report_kernel is valid
    if not isinstance(report_kernel, bool):
        raise ControlledError('/laplacian_1d/ report_kernel must be a boolean: report_kernel = %s' % type(report_kernel))
    
    x_grid = (sp.arange(G) - (G-1)/2.)/(G/2.)

    # If periodic boundary conditions, construct regular laplacian
    if periodic:
        tmp_mat = 2*sp.diag(sp.ones(G),0) - sp.diag(sp.ones(G-1),-1) - sp.diag(sp.ones(G-1),+1)
        tmp_mat[G-1,0] = -1.0
        tmp_mat[0,G-1] = -1.0
        Delta = (sp.mat(tmp_mat)/(grid_spacing**2))**alpha
        
        # Get kernel, which is just the constant vector v = sp.ones([G,1])
        # kernel_basis = utils.normalize(v, grid_spacing)
        kernel_basis = utils.legendre_basis_1d(G, 1, grid_spacing)

    # Otherwise, construct bilateral laplacian
    else:
    
        # Initialize to G x G identity matrix
        right_side = sp.diag(sp.ones(G),0)
        
        # Multiply alpha derivative matrices of together. Reduce dimension going left
        for a in range(alpha):
            right_side = derivative_matrix_1d(G-a, grid_spacing)*right_side 
        
        # Construct final bilateral laplacian
        Delta = right_side.T*right_side

        # Make sure Delta is valid
        if not (Delta.shape[0] == Delta.shape[1] == G):
            raise ControlledError('/laplacian_1d/ Delta must have shape (%d, %d): Delta.shape = %s' % (G, G, Delta.shape))

        # Construct a basis for the kernel from legendre polynomials
        kernel_basis = utils.legendre_basis_1d(G, alpha, grid_spacing)

        # Make sure kernel_basis is valid
        if not ((kernel_basis.shape[0] == G) and (kernel_basis.shape[1] == alpha)):
            raise ControlledError('/laplacian_1d/ kernel_basis must have shape (%d, %d): kernel_basis.shape = %s' %
                                  (G,alpha,kernel_basis.shape))
        
    # Sparsify matrix if requested
    if sparse:  
        Delta = csr_matrix(Delta)

    # Report kernel if requested
    if report_kernel:
        return Delta, kernel_basis

    # Otherwise, just report matrix
    else:
        return Delta


def laplacian_2d( num_gridpoints, alpha, grid_spacing=[1.0,1.0], periodic=False, sparse=False, report_kernel=False):
    """ Returns a GxG (G=GxGy) sized 2d Laplacian """
    assert(len(num_gridpoints)==2)
    Gx = num_gridpoints[0]
    Gy = num_gridpoints[1]
    G = Gx*Gy
    assert(Gx == int(Gx))
    assert(Gy == int(Gy))
    assert(alpha == int(alpha))
    assert(alpha >= 1)
    assert(len(grid_spacing)==2)
    assert(type(grid_spacing[0]) == float)
    assert(type(grid_spacing[1]) == float)
    hx = grid_spacing[0]
    hy = grid_spacing[0]
    assert(hx > 0.0)
    assert(hy > 0.0)
    
    # Identity matrices, which will be used below
    I_x = sp.mat(sp.identity(Gx))
    I_y = sp.mat(sp.identity(Gy))

    # Compute x-coords and y-coords
    x_grid = (sp.arange(Gx) - (Gx-1)/2.)/(Gx/2.)
    y_grid = (sp.arange(Gy) - (Gy-1)/2.)/(Gy/2.)
    xs,ys = np.meshgrid(x_grid,y_grid)
    
    # If periodic boundary conditions,
    if periodic:
        Delta_x = laplacian_1d(Gx, alpha=1, grid_spacing=hx, periodic=True)    
        Delta_y = laplacian_1d(Gy, alpha=1, grid_spacing=hy, periodic=True)
        
        # Use the kroneker product to generate a first-order operator
        Delta_1 = sp.mat(sp.kron(Delta_x,I_y) + sp.kron(I_x,Delta_y))

        # Raise operator to alpha power 
        Delta = Delta_1**alpha
    
    # If bilateral, construct alpha-order bilateral laplacian algorithmically
    else:
        Delta_x_array = [I_x]
        Delta_y_array = [I_y]

        for a in range(1,alpha+1):
            Delta_x_array.append( laplacian_1d(Gx, alpha=a, grid_spacing=hx) )
            Delta_y_array.append( laplacian_1d(Gy, alpha=a, grid_spacing=hy) )

        for a in range(alpha+1):
            Dx = Delta_x_array[alpha-a]
            Dy = Delta_y_array[a]
            coeff = comb(alpha,a)
            if a == 0:
                Delta = coeff*sp.mat(sp.kron(Dx,Dy))
            else:
                Delta += coeff*sp.mat(sp.kron(Dx,Dy))
        
    # Build kernel from 2d legendre polynomials
    if periodic:
        kernel_basis = utils.legendre_basis_2d(Gx, Gy, 1, grid_spacing)
    else:
        kernel_basis = utils.legendre_basis_2d(Gx, Gy, alpha, grid_spacing)

    # Sparsify matrix if requested
    if sparse:  
        Delta = csr_matrix(Delta)

    # Report kernel if requested
    if report_kernel:
        return Delta, kernel_basis

    # Otherwise, just report matrix
    else:
        return Delta
