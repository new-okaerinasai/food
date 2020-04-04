import torch
import torch.nn.functional as F


class GlassReg(torch.nn.Module):
    def __init__(self):
        super(GlassReg, self).__init__()

    # embedding - i-th col - phi(x_i), label_matrix - V, target - i-th col - i-th labels
    def forward(self, model: torch.nn.Module, V: torch.nn.Module, target: torch.Tensor) -> torch.Tensor:

        n_classes = V.weight.shape[0]
        target_sparse = torch.sparse.torch.eye(n_classes).to(target.device)
        target_sparse = target_sparse.index_select(0, target)

        A = torch.matmul(target_sparse.T, target_sparse)
        diag = torch.diag(A)
        ind = diag.nonzero(as_tuple=True)
        diag[ind] = 1. / diag[ind]
        Z = torch.diag(diag)

        reg = torch.matmul(V.weight, V.weight.T) - \
            (torch.matmul(A, Z) + torch.matmul(Z, A)) / 2
        L = (torch.norm(reg, p='fro') / n_classes) ** 2
        return L


class GlassSimpleLoss(torch.nn.Module):
    def __init__(self):
        super(GlassSimpleLoss, self).__init__()

    def forward(self, target: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
        L = torch.Tensor([0]).to(target.device)
        c = 0.1
        for i in range(target.shape[0]):
            l = torch.max(
                (prediction[i] - prediction[i][target[i]] + c).T, torch.zeros_like(prediction[i]))
            print(prediction[i])
            print(target[i])
            print(l)
            l[target[i]] = 0
            L += l.sum()  # added pred[ind]-pred[ind]+c
        return L / prediction.shape[0]


class GlassLoss(torch.nn.Module):
    def __init__(self, alpha: float = 10., loss_type='crossentropy'):
        super(GlassLoss, self).__init__()
        self._alpha = alpha
        self._reg = GlassReg()
        self._loss = torch.nn.CrossEntropyLoss()
        self._type = loss_type
        if loss_type == 'simple':
            self._loss = GlassSimpleLoss()

    def forward(self, model: torch.nn.Module, V: torch.Tensor, target: torch.Tensor,
                prediction: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        mask = target < V.weight.shape[0]
        self._loss = self._loss.to(target.device)
        if self._type == 'simple':
            loss = self._loss(target[mask], logits[mask, :])
        else:
            loss = self._loss(logits, target)
        reg = self._reg(model, V, target[mask])

        return loss + self._alpha * reg
