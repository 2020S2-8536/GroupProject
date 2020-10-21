import numpy as np
import torch


def b_oh(b_o,b_h):
    '''
    b_o is the coordinate information of object
    b_h is the coordinate information of human
    The formation of b_o and b_h is [x,y,w,h]
    b_oh = [(x_o - x_h)/w_h,(y_o - y_h)/h_h,log(w_o/w_h),log(h_o/h_h)] 
    _o represents object
    _h represents human
    '''
    a = (b_o[0]-b_h[0])/b_h[2]
    b = (b_o[1]-b_h[1])/b_h[3]
    c = np.log(b_o[2]/b_h[2])
    d = np.log(b_o[3]/b_h[3])

    return torch.from_numpy(np.array([a,b,c,d]))

def Gaussian_fuc(b_oh,mu_ah,sigma=0.3):
    '''
    mu_ah is the predicts from the human_centric branch,
    which means the 4_d mean location of the target given the b_h and action
    sigma is a hyperparameter
    g_aho = exp(norm2(b_oh - mu_ah)/2*sigma**2)
    '''
    g_aho = np.exp(np.linalg.norm(b_oh - mu_ah)/2*np.power(sigma,2))
    return g_aho

# if __name__ == "__main__":
#     mu_ah = [1,2,3,4]
#     sigma = 1
#     b_o = [1,2,3,4]
#     b_h = [4,3,2,1]
#     b_o_h = b_oh(b_o,b_h)
#     print(b_o_h)
#     g = Gaussian_fuc(b_o_h,mu_ah,sigma)
#     print(g)

    



