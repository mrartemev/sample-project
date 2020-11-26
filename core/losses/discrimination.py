import torch


def interpolate(a, b):
    alpha = torch.rand(a.size(0), 1, 1, 1, device=a.device)
    inter = a + alpha * (b - a)
    return inter


def calculate_gradient_penalty(critic, x_real, x_fake):
    image = interpolate(x_real, x_fake).requires_grad_(True)
    pred = critic(image)
    if isinstance(pred, tuple):
        pred = pred[0]
    grad = torch.autograd.grad(
        outputs=pred, inputs=image,
        grad_outputs=torch.ones_like(pred),
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    grad = grad.view(grad.shape[0], -1)
    norm = grad.norm(2, dim=1)
    gp = ((norm - 1.0) ** 2).mean()
    return gp