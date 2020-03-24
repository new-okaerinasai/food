import torch


class glass_reg(torch.nn.Module):
    def __init__(self):
        super(glass_reg, self).__init__()

    def forward(self, model: torch.nn.Module, V: torch.nn.Linear, target: torch.Tensor) -> torch.Tensor: #embedding - i-th col - phi(x_i), label_matrix - V, target - i-th col - i-th labels

        n_classes = model.size()[1]
        target_sparse = torch.sparse.torch.eye(n_classes).to(target.device)
        target_sparse = target_sparse.index_select(0, target)
        
        A = torch.tensordot(target_sparse.t,target_sparse)
        diag = torch.diagonal(A)
        diag[diag==0]=float('Inf')
        Z = torch.diag(1./diag)
        reg = torch.tensordot(V.weight,V.weight.T) - (torch.tensordot(A,Z)+torch.tensordot(Z,A))/2
        L=torch.norm(reg,ord='fro')/n_classes**2

        return L
    
class glass_simple_loss(torch.nn.Module):
    def __init__(self):
        super(glass_loss, self).__init__()
    
    def forward(self, target: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
        L = torch.Tensor([0])
        c = 0.1
        for i in range(target.shape[0]):
            l = max(predicted - predicted[target[i]] + c,0)
        L+= l.sum()-c,  # added pred[ind]-pred[ind]+c    
        return L
        
class glass_loss(torch.nn.Module):
    def __init__(self, alpha: float = 1.):
        super(glass_loss, self).__init__()
        self._alpha = alpha
        self._reg = glass_reg()
        self._loss = glass_simple_loss()    

    def forward(self, model: torch.nn.Module, V: torch.nn.Linear, target: torch.Tensor,
                prediction: torch.Tensor) -> torch.Tensor:
        loss = self._loss(target, prediction)
        reg = self._reg(self, model, V, target)
        
        return self._alpha * reg + loss    
