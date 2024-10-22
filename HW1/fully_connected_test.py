from fully_connected import FullyConnected
import torch

def J(y):
    return torch.mean(y)

def fully_connected_test():
    """
    Provides Unit tests for the FullyConnected autograd Function

    PROVIDED CONSTANTS
    ------------------
    TOL (float): The error tolerance for the backward mode. If the error >= TOL, then is_correct is false
    DELTA (float): The difference parameter for the finite difference computations
    X (Tensor): of size (48 x 2), the inputs
    W (Tensor): of size (2 x 72), the weights
    B (Tensor): of size (72), the biases

    Returns
    -------
    is_correct (boolean): True if and only iff FullyConnected passes all unit tests
    err (Dictionary): with the following keys
                    1. dzdx: the error between the analytical and numerical gradients w.r.t X
                    2. dzdw (float): ... w.r.t W
                    3. dzdb (float): ... w.r.t B

    Note
    ----
    The error between arbitrary tensors x and y is defined here as the maximum value of the absolute difference between
    x and y.
    """
    # %%% DO NOT EDIT BELOW %%%%
    dataset = torch.load("fully_connected_test.pt")
    X = dataset["X"]
    W = dataset["W"]
    B = dataset["B"]
    TOL = dataset["TOL"]
    DELTA = dataset["DELTA"]
    full_connected = FullyConnected.apply
    # %%% DO NOT EDIT ABOVE

    Y = full_connected(X,W,B)
    Z = J(Y)
    DZDY = (torch.autograd.grad(Z, Y))[0]
    
    Z.backward()
    
    dzdx_analytical = X.grad.clone()
    dzdw_analytical = W.grad.clone()
    dzdb_analytical = B.grad.clone()

    with torch.no_grad():
        dzdx_numerical = torch.zeros_like(X)
        dzdw_numerical = torch.zeros_like(W)
        dzdb_numerical = torch.zeros_like(B)
       
        for t in range(X.shape[0]):
            for i in range(X.shape[1]):
                X_plus = X.clone()
                X_minus = X.clone()
                X_plus[t, i] += DELTA
                X_minus[t, i] -= DELTA
               
                y_plus = full_connected(X_plus, W, B)
                y_minus = full_connected(X_minus, W, B)
                
               
                dzdx_numerical[t][i] =  torch.sum(DZDY*((y_plus) - (y_minus)) / (2 * DELTA))
               
        for i in range(W.shape[0]):
            for j in range(W.shape[1]):
                W_plus = W.clone()
                W_minus = W.clone()
                W_plus[i, j] += DELTA
                W_minus[i, j] -= DELTA
               
                y_plus = full_connected(X, W_plus, B)
                y_minus = full_connected(X, W_minus, B)
               
                dzdw_numerical[i][j] =  torch.sum(DZDY*((y_plus) - (y_minus)) / (2 * DELTA))
        
        for i in range(B.shape[0]):
            B_plus = B.clone()
            B_minus = B.clone()
            B_plus[i] += DELTA
            B_minus[i] -= DELTA
           
            y_plus = full_connected(X, W, B_plus)
            y_minus = full_connected(X, W, B_minus)
           
            dzdb_numerical[i] =  torch.sum(DZDY*((y_plus) - (y_minus)) / (2 * DELTA))
        
        dzdx_err = torch.max(torch.abs(dzdx_analytical - dzdx_numerical))
        dzdw_err = torch.max(torch.abs(dzdw_analytical - dzdw_numerical))
        dzdb_err = torch.max(torch.abs(dzdb_analytical - dzdb_numerical))
   
    is_correct = dzdx_err < TOL and dzdw_err < TOL and dzdb_err < TOL 
    err = {
        'dzdx': dzdx_err.item(),
        'dzdw': dzdw_err.item(),
        'dzdb': dzdb_err.item()
    }
    torch.save([is_correct , err], 'fully_connected_test_results.pt')
    

    return is_correct, err


if __name__ == '__main__':
    tests_passed, errors = fully_connected_test()
    assert tests_passed
    print(errors)
