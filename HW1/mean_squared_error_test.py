from mean_squared_error import MeanSquaredError
import torch

def J(s,t):
    return MeanSquaredError.apply(s,t)
def mean_squared_error_test():
    """
     Unit tests for the MeanSquaredError autograd Function.

    PROVIDED CONSTANTS
    ------------------
    TOL (float): the absolute error tolerance for the backward mode. If any error is equal to or
                greater than TOL, is_correct is false
    DELTA (float): The difference parameter for the finite difference computation
    X1 (Tensor): size (48 x 2) denoting 72 example inputs each with 2 features
    X2 (Tensor): size (48 x 2) denoting the targets

    Returns
    -------
    is_correct (boolean): True if and only if MeanSquaredError passes all unit tests
    err (Dictionary): with the following keys
                    1. dzdx1 (float): the  error between the analytical and numerical gradients w.r.t X1
                    2. dzdx2 (float): The error between the analytical and numerical gradients w.r.t X2
    Note
    -----
    The error between arbitrary tensors x and y is defined here as the maximum value of the absolute difference between
    x and y.
    """
    # %%% DO NOT EDIT BELOW %%%
    dataset = torch.load("mean_squared_error_test.pt")
    X1 = dataset["X1"]
    X2 = dataset["X2"]
    TOL = dataset["TOL"]
    DELTA = dataset["DELTA"]
    mean_squared_error = MeanSquaredError.apply
    # %%% DO NOT EDIT ABOVE %%%
    y = mean_squared_error(X1, X2)
    z = y
    z.backward()
    dzdx1_analytical = X1.grad.clone()
    dzdx2_analytical = X2.grad.clone()
    with torch.no_grad():
        dzdx1_numerical = torch.zeros_like(X1)
        dzdx2_numerical = torch.zeros_like(X2)

        DZDY = torch.autograd.grad(z, y)[0]

        for i in range(X1.shape[0]):
            for j in range(X1.shape[1]):
                X1_plus = X1.clone()
                X1_minus = X1.clone()
                X1_plus[i, j] += DELTA
                X1_minus[i, j] -= DELTA

                y_plus = mean_squared_error(X1_plus, X2)
                y_minus = mean_squared_error(X1_minus, X2)

                dzdx1_numerical[i, j] = torch.sum(DZDY*((y_plus) - (y_minus)) / (2 * DELTA))

        for i in range(X2.shape[0]):
            for j in range(X2.shape[1]):
                X2_plus = X2.clone()
                X2_minus = X2.clone()
                X2_plus[i, j] += DELTA
                X2_minus[i, j] -= DELTA

                y_plus = mean_squared_error(X1, X2_plus)
                y_minus = mean_squared_error(X1, X2_minus)

                dzdx2_numerical[i, j] = torch.sum(DZDY*((y_plus) - (y_minus)) / (2 * DELTA))

        dzdx1_err = torch.max(torch.abs(dzdx1_analytical - dzdx1_numerical))
        dzdx2_err = torch.max(torch.abs(dzdx2_analytical - dzdx2_numerical))
    
    # Check if analytical gradients are correct
    is_correct = dzdx1_err < TOL and dzdx2_err < TOL
    gradcheck_result = torch.autograd.gradcheck(mean_squared_error, (X1, X2), eps=DELTA, atol=TOL)
    is_correct = is_correct and gradcheck_result
    
    err = {
        'dzdx1': dzdx1_err.item(),
        'dzdx2': dzdx2_err.item()
    }
    torch.save([is_correct, err], 'mean_squared_error_test_results.pt')


    return is_correct, err


if __name__ == '__main__':
    tests_passed, errors = mean_squared_error_test()
    assert tests_passed
    print(errors)
