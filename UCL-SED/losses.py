import torch
import torch.nn.functional as F

def common_loss(emb1, emb2):
    emb1 = emb1 - torch.mean(emb1, dim=0, keepdim=True)
    emb2 = emb2 - torch.mean(emb2, dim=0, keepdim=True)
    emb1 = torch.nn.functional.normalize(emb1, p=2, dim=1)
    emb2 = torch.nn.functional.normalize(emb2, p=2, dim=1)
    cov1 = torch.matmul(emb1, emb1.t())
    cov2 = torch.matmul(emb2, emb2.t())
    cost = torch.mean((cov1 - cov2)**2)
    return cost
def EUC_loss(alpha,u,true_labels,e):
    _, pred_label = torch.max(alpha, 1)
    true_indices = torch.where(pred_label == true_labels)
    false_indices = torch.where(pred_label != true_labels)
    S = torch.sum(alpha, dim=1, keepdim=True)
    p, _ = torch.max(alpha / S, 1)
    a = -0.01 * torch.exp(-(e + 1) / 10 * torch.log(torch.FloatTensor([0.01]))).cuda()
    annealing_coef = torch.min(
        torch.tensor(1.0, dtype=torch.float32),
        torch.tensor((e+1) / 10, dtype=torch.float32),
    )
    EUC_loss = -annealing_coef * torch.sum((p[true_indices]*(torch.log(1.000000001 - u[true_indices]).squeeze(
        -1)))) # -(1-annealing_coef)*torch.sum(((1-p[false_indices])*(torch.log(u[false_indices]).squeeze(-1))))




    return EUC_loss

def relu_evidence(y):
    return F.relu(y)


def exp_evidence(y):
    return torch.exp(torch.clamp(y, -10, 10))


def softplus_evidence(y):
    return F.softplus(y)


def kl_divergence(alpha, num_classes, device):
    ones = torch.ones([1, num_classes], dtype=torch.float32, device=device)
    sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
    first_term = (
        torch.lgamma(sum_alpha)
        - torch.lgamma(alpha).sum(dim=1, keepdim=True)
        + torch.lgamma(ones).sum(dim=1, keepdim=True)
        - torch.lgamma(ones.sum(dim=1, keepdim=True))
    )
    second_term = (
        (alpha - ones)
        .mul(torch.digamma(alpha) - torch.digamma(sum_alpha))
        .sum(dim=1, keepdim=True)
    )
    kl = first_term + second_term
    return kl


def kl_pred_divergence(alpha, y, num_classes, device):
    # max_alpha, _ = torch.max(alpha, 1)
    # ones = alpha*(1-y) + (max_alpha+1) * y
    ones = y + 0.01*torch.ones([1, num_classes], dtype=torch.float32, device=device)
    sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
    first_term = (
        torch.lgamma(sum_alpha)
        - torch.lgamma(alpha).sum(dim=1, keepdim=True)
        + torch.lgamma(ones).sum(dim=1, keepdim=True)
        - torch.lgamma(ones.sum(dim=1, keepdim=True))
    )
    second_term = (
        (alpha - ones)
        .mul(torch.digamma(alpha) - torch.digamma(sum_alpha))
        .sum(dim=1, keepdim=True)
    )
    kl = first_term + second_term
    return kl



def loglikelihood_loss(y, alpha, device):
    y = y.to(device)
    alpha = alpha.to(device)
    S = torch.sum(alpha, dim=1, keepdim=True)
    loglikelihood_err = torch.sum((y - (alpha / S)) ** 2, dim=1, keepdim=True)
    loglikelihood_var = torch.sum(
        alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True
    )
    loglikelihood = loglikelihood_err + loglikelihood_var
    return loglikelihood


def mse_loss(y, alpha, epoch_num, num_classes, annealing_step, device):
    y = y.to(device)
    alpha = alpha.to(device)
    loglikelihood = loglikelihood_loss(y, alpha, device)

    annealing_coef = torch.min(
        torch.tensor(1.0, dtype=torch.float32),
        torch.tensor(epoch_num / annealing_step, dtype=torch.float32),
    )

    kl_alpha = (alpha - 1) * (1 - y) + 1
    kl_div = annealing_coef * kl_divergence(kl_alpha, num_classes, device=device)
    return loglikelihood + kl_div


def edl_loss(func, y, true_labels,alpha, epoch_num, num_classes, annealing_step, device):
    y = y.to(device)
    alpha = alpha.to(device)
    S = torch.sum(alpha, dim=1, keepdim=True)

    A = torch.sum(y * (func(S) - func(alpha)), dim=1, keepdim=True)

    annealing_coef = torch.min(
        torch.tensor(1.0, dtype=torch.float32),
        torch.tensor((epoch_num+1) / 10, dtype=torch.float32),
    )


    _, pred_label = torch.max(alpha, 1)
    true_indices = torch.where(pred_label == true_labels)
    false_indices = torch.where(pred_label != true_labels)
    kl_alpha = (alpha - 1) * (1 - y) + 1
    kl_div = annealing_coef * kl_divergence(kl_alpha, num_classes, device=device)
    print("kl_div:",1*torch.mean(kl_div))
    print("A:",20*torch.mean(A))

    return 20*A + 1*kl_div 


def edl_mse_loss(alpha, target, epoch_num, num_classes, annealing_step, device):
    # evidence = relu_evidence(output)
    # alpha = evidence + 1
    loss = torch.mean(
        mse_loss(target, alpha, epoch_num, num_classes, annealing_step, device=device)
    )
    return loss


def edl_log_loss(alpha, target, epoch_num, num_classes, annealing_step, device):
    # evidence = relu_evidence(output)
    # alpha = evidence + 1
    loss = torch.mean(
        edl_loss(
            torch.log, target, alpha, epoch_num, num_classes, annealing_step, device
        )
    )
    return loss


def edl_digamma_loss(
    alpha, target,true_labels, epoch_num, num_classes, annealing_step, device):
    # evidence = relu_evidence(output)
    # alpha = evidence + 1

    loss = torch.mean(edl_loss(
            torch.digamma, target,true_labels, alpha, epoch_num, num_classes, annealing_step, device

    ))
    return loss
