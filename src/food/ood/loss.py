import torch
import torch.nn.functional as F


class glass_reg(torch.nn.Module):
    def __init__(self):
        super(glass_reg, self).__init__()

    def forward(self, model: torch.nn.Module, V: torch.nn.Module, target: torch.Tensor) -> torch.Tensor: #embedding - i-th col - phi(x_i), label_matrix - V, target - i-th col - i-th labels

        n_classes = V.weight.shape[0]
        target_sparse = torch.sparse.torch.eye(n_classes).to(target.device)
        target_sparse = target_sparse.index_select(0, target)

        A = torch.matmul(target_sparse.T,target_sparse)
        diag = torch.diag(A)
        ind = diag.nonzero(as_tuple = True)
        diag[ind] = 1./ diag[ind]
        Z = torch.diag(diag)

        reg = torch.matmul(V.weight,V.weight.T) - (torch.matmul(A,Z)+torch.matmul(Z,A))/2
        L=torch.norm(reg,p='fro')/n_classes**2

        return L
    
class glass_simple_loss(torch.nn.Module):
    def __init__(self):
        super(glass_simple_loss, self).__init__()
    
    def forward(self, target: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
        L = torch.Tensor([0]).to(target.device)
        c = 0.1
        for i in range(target.shape[0]):
            l = torch.max((prediction[i] - prediction[i][target[i]] + c).T,torch.zeros_like(prediction[i]))
            l[target[i]] = 0
            L+= l.sum()  # added pred[ind]-pred[ind]+c 
        return L/prediction.shape[0]
        
class glass_loss(torch.nn.Module):
    def __init__(self, alpha: float = 10.):
        super(glass_loss, self).__init__()
        self._alpha = alpha
        self._reg = glass_reg()
        self._loss = torch.nn.CrossEntropyLoss()

    def forward(self, model: torch.nn.Module, V: torch.Tensor, target: torch.Tensor,
            prediction: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        mask = target<V.weight.shape[0]
        self._loss = self._loss.to(target.device)
        loss = self._loss(logits[mask,:],target[mask])
        reg = self._reg(model, V, target[mask])
        
        return loss +self._alpha * reg  
    

