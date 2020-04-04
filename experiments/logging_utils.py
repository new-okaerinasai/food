import torch
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

def get_accuracy_with_logits(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """
    Calculate accuracy with logits
    :param y_true: torch.Tensor of ints containing true labels. shape = (N,)
    :param y_pred: torch.Tensor of predicted logits. shape = (N, N_classes)
    :return acc: float which equals to accuracy score
    """

    return (y_true == y_pred.argmax(dim=1)).float().mean().item()


def get_ood_predictions(y_true: torch.Tensor,
                        y_pred: torch.Tensor,
                        thr: float = None,
                        ood_label: int = 0):
    p_pred = F.softmax(y_pred, dim=1)
    ood_pred = (p_pred.max(dim=1) < thr).int()
    ood_true = (y_true == ood_label).int()
    return ood_true, ood_pred


def get_ood_precision_with_logits(y_true: torch.Tensor,
                                  y_pred: torch.Tensor,
                                  thr: float = None,
                                  ood_label: int = 0) -> float:
    ood_true, ood_pred = get_ood_predictions(y_true, y_pred, thr, ood_label)
    return precision_score(ood_true.cpu().detach().numpy(), ood_pred.cpu().detach().numpy())


def get_ood_recall_with_logits(y_true: torch.Tensor,
                               y_pred: torch.Tensor,
                               thr: float = None,
                               ood_label: int = 0) -> float:
    ood_true, ood_pred = get_ood_predictions(y_true, y_pred, thr, ood_label)
    return recall_score(ood_true.cpu().detach().numpy(), ood_pred.cpu().detach().numpy())


def get_ood_histograms(y_true, y_pred, ood_label=0):
    y_pred_ood, _ = F.softmax(y_pred[y_true == ood_label], dim=1).max(dim=1)
    y_pred_nor, _ = F.softmax(y_pred[y_true != ood_label], dim=1).max(dim=1)
    # print(y_pred_ood.shape, y_pred_nor.shape)
    return y_pred_ood, y_pred_nor


def get_metrics_dict(y_true, y_pred, thr=None, ood_label=0):
    metrics_dict = {}
    metrics_dict["scalar"] = {}
    metrics_dict["scalar"]["accuracy"] = get_accuracy_with_logits(y_true, y_pred)
    if thr is not None:
        metrics_dict["scalar"]["ood_recall"] = get_ood_recall_with_logits(y_true, y_pred, thr, ood_label)
        metrics_dict["scalar"]["ood_precision"] = get_ood_precision_with_logits(y_true, y_pred, thr, ood_label)
    metrics_dict["hist"] = {}
    metrics_dict["hist"]["ood_max_ood"], metrics_dict["hist"]["ood_max_known"] = get_ood_histograms(y_true, y_pred,
                                                                                                    ood_label)
    return metrics_dict

def log_hist_as_picture(y_true: torch.Tensor, y_pred: torch.Tensor,
                        ood_label=0, thr=None):
    print("OOD_LABEL = ", ood_label)
    print("y_true max", y_true.max(), y_true.shape)
    metrics_dict = get_metrics_dict(y_true, y_pred, thr, ood_label)

    c = ["red", "blue"]
    plt.figure(figsize=(20, 16))
    plt.title("Known classes vs OOD max logit distribution")
    for i, (names, hist) in enumerate(metrics_dict["hist"].items()):
        plt.hist(hist.cpu().detach().numpy(), bins=20, range=(0,1),
                 label=names, alpha=0.4, color=c[i % 2], density=True)
    plt.legend()
    plt.savefig("hist.png")
    plt.close()

def log_dict_with_writer(y_true: torch.Tensor, y_pred: torch.Tensor,
                         summary_writer: torch.utils.tensorboard.SummaryWriter,
                         thr=None, ood_label=0, global_step=None):
    """
    Log metrics to tensorboard with summary writer
    :param y_true: true labels of the objects, shape=(N,)
    :param torch.Tensor y_pred: logits, predictions of the model BEFORE the softmax function,
        shape=(N, n_classes + 1). Note that ood label is not the one on which we train.
    :param SummaryWriter summary_writer: a writer for logging metrics to tensorboard
    :param float thr: Value of the maximum probability below which we consider an object as ood
    :param ood_label: label which corresponds to an ood object
    :return:
    """
    metrics_dict = get_metrics_dict(y_true, y_pred, thr, ood_label)
    for names, hist in metrics_dict["hist"].items():
        summary_writer.add_histogram(names+str(global_step), hist)

    for names, scalars in metrics_dict["scalar"].items():
        summary_writer.add_histogram(names, scalars)
