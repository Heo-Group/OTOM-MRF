import torch

def my_loss(output,target,device):
    # min_8=torch.tensor([3.0,30.0,500,0.1,1.0,100,0.2,100e-6],device=self.device,requires_grad=False)
    # max_8=torch.tensor([0.2,1.0,5,0.001,1.0e-2,5,0.001,1e-6],device=self.device,requires_grad=False)

    # output_norm=torch.div(output-min_4,max_4-min_4)
    # target_norm=torch.div(target-min_4,max_4-min_4)

    diff_norm = (output-target)**2
    
    Rm_diff=torch.sqrt(torch.mean(diff_norm[:,0]))
    Mm_diff=torch.sqrt(torch.mean(diff_norm[:,1]))
    T2m_diff=torch.sqrt(torch.mean(diff_norm[:,2]))
    T1w_diff=torch.sqrt(torch.mean(diff_norm[:,3]))

    diff_L1 = abs(output-target)

    error_total = torch.mean(diff_L1)

    return error_total,T1w_diff,Rm_diff,Mm_diff,T2m_diff