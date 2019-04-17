# Author: Guo-qing Jiang (jianggq@mit.edu)
# Pytorch second oder gradient calculation for the diagonal of Hessian matrix 
# feel free to copy
import torch
import time


def get_second_order_grad(grads, xs,device):
    start = time.time()
    grads2 = []
    for j, (grad, x) in enumerate(zip(grads, xs)):
        print('2nd order on layer ', j)
        print(x.size())
        grad = torch.reshape(grad, [-1])
        grads2_tmp = []
        grad_wts = torch.ones(grad.shape).to(device)
        g2 = torch.autograd.grad(grad,x, grad_wts, retain_graph=True)[0]
        grads2.append(g2)
        # for count, g in enumerate(grad):
            # g2 = torch.autograd.grad(g, x, retain_graph=True)[0]
            # g2 = torch.reshape(g2, [-1])
            # grads2_tmp.append(g2[count].data.cpu().numpy())
        # grads2.append(torch.from_numpy(np.reshape(grads2_tmp, x.size())).to(DEVICE_IDS[0]))
        # grads2.append(grads2_tmp)
    # grad_mask = torch.eye(len(xs))
    # sec_order = []
    # for i in range(grad_mask.shape[0]):
    #     # mask = grad_mask[i,:]
    #     sec_order.append(torch.autograd.grad([grads[i]], xs, grad_mask, retain_graph=True)[0])

    print('Time used is ', time.time() - start)
    # print(torch.stack(sec_order))
    # for grad in grads2:  # check size
    #     print(grad.size())
    return grads2

# datainput/model/optimizer setup is ommited here
# optimizer.zero_grad()
# xs = optimizer.param_groups[0]['params']
# ys = loss  # put your own loss into ys

# xs= torch.ones(3,requires_grad=True)
# xs = torch.tensor([1,2,3],requires_grad=True,dtype=torch.float)
# ys=xs**3
# # ys2=ys*xs
# grad_mask = torch.ones(ys.shape)
# grads = torch.autograd.grad(ys, xs, grad_mask, create_graph=True)[0]  # first order gradient
# grad_vec = grads.clone().detach()
# grad_vec_prod = torch.mul(grads,grad_vec)
# # print("Grad out:",grads)
# # print("X.grad:",xs.grad)
# # print(grad_vec_prod.requires_grad)
# print(get_second_order_grad(grads, xs))  # second order gradient

