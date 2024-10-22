from generalized_logistic import GeneralizedLogistic
import torch

def J(y):
    return torch.mean(y)

def generalized_logistic_test():
    """
    Provides Unit tests for the GeneralizedLogistic autograd Function

    PROVIDED CONSTANTS
    ------------------
    TOL1 (float): the  error tolerance for the forward mode. If the error >= TOL1, is_correct is false
    TOL2 (float): The error tolerance for the backward mode
    DELTA (float): The difference parameter for the finite differences computation
    X (Tensor): size (48 x 2) of inputs
    L, U, and G (floats): The parameter values necessary to compute the hyperbolic tangent (tanH) using
                        GeneralizedLogistic
    Returns:
    -------
    is_correct (boolean): True if and only if GeneralizedLogistic passes all unit tests
    err (Dictionary): with the following keys
                        1. y (float): The error between the forward direction and the results of pytorch's tanH
                        2. dzdx (float): the error between the analytical and numerical gradients w.r.t X
                        3. dzdl (float): ... w.r.t L
                        4. dzdu (float): ... w.r.t U
                        5. dzdg (float): .. w.r.t G
     Note
     -----
    The error between arbitrary tensors x and y is defined here as the maximum value of the absolute difference between
    x and y.
    """
    # %%%% DO NOT EDIT BELOW %%%
    dataset = torch.load("generalized_logistic_test.pt")
    X = dataset["X"]
    L = dataset["L"]
    U = dataset["U"]
    G = dataset["G"]
    TOL1 = dataset["TOL1"]
    TOL2 = dataset["TOL2"]
    DELTA = dataset["DELTA"]
    generalized_logistic = GeneralizedLogistic.apply
    # %%%  DO NOT EDIT ABOVE %%%
    y =  generalized_logistic(X, L, U, G)
    z = J(y)
    z.backward()



    dzdx_analytical = X.grad.clone()
    dzdl_analytical = L.grad.clone() if L.numel() > 1 else L.grad.clone().view(1)
    dzdu_analytical = U.grad.clone() if U.numel() > 1 else U.grad.clone().view(1)
    dzdg_analytical = G.grad.clone() if G.numel() > 1 else G.grad.clone().view(1)

    

    with torch.no_grad():
        DZDY = torch.autograd.grad(z, y)[0]
        dzdx_numerical = torch.zeros_like(X)
        dzdl_numerical = torch.zeros_like(L) if L.numel() > 1 else torch.tensor(0.)
        dzdu_numerical = torch.zeros_like(U) if U.numel() > 1 else torch.tensor(0.)
        dzdg_numerical = torch.zeros_like(G) if G.numel() > 1 else torch.tensor(0.)
        
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                X_plus = X.clone()
                X_minus = X.clone()
                X_plus[i, j] += DELTA
                X_minus[i, j] -= DELTA
                
                y_plus = generalized_logistic(X_plus, L, U, G)
                y_minus = generalized_logistic(X_minus, L, U, G)
                
                dzdx_numerical[i, j] = torch.sum(DZDY*((y_plus) - (y_minus)) / (2 * DELTA))
        if L.numel() > 1:
            for i in range(L.shape[0]):
                for j in range(L.shape[1]):
                    L_plus = L.clone()
                    L_minus = L.clone()
                    L_plus[i, j] += DELTA
                    L_minus[i, j] -= DELTA
                    
                    y_plus = generalized_logistic(X, L_plus, U, G)
                    y_minus = generalized_logistic(X, L_minus, U, G)
                    
                    dzdl_numerical[i, j] = torch.sum(DZDY*((y_plus) - (y_minus)) / (2 * DELTA))
        else:
            L_plus = L + DELTA
            L_minus = L - DELTA
            y_plus = generalized_logistic(X, L_plus, U, G)
            y_minus = generalized_logistic(X, L_minus, U, G)
            dzdl_numerical = torch.sum(DZDY*((y_plus) - (y_minus)) / (2 * DELTA))
        
        if U.numel() > 1:
            for i in range(U.shape[0]):
                for j in range(U.shape[1]):
                    U_plus = L.clone()
                    U_minus = L.clone()
                    U_plus[i, j] += DELTA
                    U_minus[i, j] -= DELTA
                    
                    y_plus = generalized_logistic(X, L, U_plus, G)
                    y_minus = generalized_logistic(X, L, U_minus, G)
                    
                    dzdu_numerical[i, j] = torch.sum(DZDY*((y_plus) - (y_minus)) / (2 * DELTA))
        else:
            U_plus = U + DELTA
            U_minus = U - DELTA
            y_plus = generalized_logistic(X, L, U_plus, G)
            y_minus = generalized_logistic(X, L, U_minus, G)
            dzdu_numerical = torch.sum(DZDY*((y_plus) - (y_minus)) / (2 * DELTA))

        if G.numel() > 1:
            for i in range(G.shape[0]):
                for j in range(G.shape[1]):
                    G_plus = G.clone()
                    G_minus = G.clone()
                    G_plus[i, j] += DELTA
                    G_minus[i, j] -= DELTA
                    
                    y_plus = generalized_logistic(X, L, U, G_plus)
                    y_minus = generalized_logistic(X, L, U, G_minus)
                    
                    dzdg_numerical[i, j] = torch.sum(DZDY*((y_plus) - (y_minus)) / (2 * DELTA))
        else:
            G_plus = G + DELTA
            G_minus = G - DELTA
            y_plus = generalized_logistic(X, L, U, G_plus)
            y_minus = generalized_logistic(X, L, U, G_minus)
            dzdg_numerical = torch.sum(DZDY*((y_plus) - (y_minus)) / (2 * DELTA))
    dzdx_err = torch.max(torch.abs(dzdx_analytical - dzdx_numerical))
    dzdl_err = torch.max(torch.abs(dzdl_analytical - dzdl_numerical))
    dzdu_err = torch.max(torch.abs(dzdu_analytical - dzdu_numerical))
    dzdg_err = torch.max(torch.abs(dzdg_analytical - dzdg_numerical))
    tanh_output = torch.tanh(X)
    y_err = torch.max(torch.abs(y - tanh_output))

    is_correct = dzdx_err < TOL2 and dzdl_err < TOL2 and dzdu_err < TOL2 and dzdg_err < TOL2 and y_err < TOL1
    gradcheck_result = torch.autograd.gradcheck(generalized_logistic, (X, L, U, G), eps=DELTA, atol=TOL2)
    is_correct = is_correct and gradcheck_result

    err = {
        'dzdx': dzdx_err.item(),
        'dzdl': dzdl_err.item(),
        'dzdu': dzdu_err.item(),
        'dzdg': dzdg_err.item(),
        'y': y_err.item()
    }
    #print(TOL1)
    #print(TOL2)
    
    
    torch.save([is_correct, err], 'generalized_logistic_test_results.pt')
    return is_correct, err


if __name__ == '__main__':
    test_passed, errors = generalized_logistic_test()
    #print(errors)
    assert test_passed
    #print(test_passed)
    print(errors)
