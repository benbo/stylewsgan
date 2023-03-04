# This code was modified from its original source by Benedikt Boecking.
# The original source code can be found at https://github.com/NVlabs/stylegan2-ada-pytorch
# The original source code is distributed under the following license, which still applies to this modified version:

# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np
import torch
import torch.nn.functional as F
from torch_utils import training_stats
from torch_utils import misc
from torch_utils.ops import conv2d_gradfix
import torchmetrics

GPHASES = frozenset(
    ["Gmain", "Greg", "Gboth", "Dmain", "Dreg", "Dboth", "Info", "Ymap"]
)

# ----------------------------------------------------------------------------


class Loss:
    def accumulate_gradients(
        self, phase, real_img, real_c, gen_z, gen_c, sync, gain
    ):  # to be overridden by subclass
        raise NotImplementedError()


def soft_cross_entropy_noreduce(pred, soft_targets):
    """
    Cross entropy with soft targets (probabilities), while pred is assumed to be logits.
    """
    predlogsoftmax = F.log_softmax(pred, dim=1)
    return torch.sum(-soft_targets * predlogsoftmax, 1)


# ----------------------------------------------------------------------------


class StyleWSGANLoss(Loss):
    def __init__(
        self,
        device,
        G_mapping,
        G_synthesis,
        D,
        labelmodel,
        augment_pipe=None,
        style_mixing_prob=0.9,
        r1_gamma=10,
        pl_batch_shrink=2,
        pl_decay=0.01,
        pl_weight=2,
        cardinality=None,
        num_LFs=None,
    ):
        super().__init__()
        self.device = device
        self.G_mapping = G_mapping
        self.G_synthesis = G_synthesis
        self.D = D
        self.labelmodel = labelmodel
        self.augment_pipe = augment_pipe
        self.style_mixing_prob = style_mixing_prob
        self.r1_gamma = r1_gamma
        self.pl_batch_shrink = pl_batch_shrink
        self.pl_decay = pl_decay
        self.pl_weight = pl_weight
        self.pl_mean = torch.zeros([], device=device)
        self.n_classes = float(cardinality)
        self.num_LFs = num_LFs
        self.usefmap = False
        self.beta_dist = torch.distributions.beta.Beta(
            torch.tensor([0.5]), torch.tensor([0.5])
        )

    def run_G(self, z, c, sync):
        with misc.ddp_sync(self.G_mapping, sync):
            ws = self.G_mapping(z, c)
            if self.style_mixing_prob > 0:
                with torch.autograd.profiler.record_function("style_mixing"):
                    cutoff = torch.empty(
                        [], dtype=torch.int64, device=ws.device
                    ).random_(1, ws.shape[1])
                    cutoff = torch.where(
                        torch.rand([], device=ws.device) < self.style_mixing_prob,
                        cutoff,
                        torch.full_like(cutoff, ws.shape[1]),
                    )
                    ws[:, cutoff:] = self.G_mapping(
                        torch.randn_like(z), c, skip_w_avg_update=True
                    )[:, cutoff:]
        with misc.ddp_sync(self.G_synthesis, sync):
            img = self.G_synthesis(ws)
        return img, ws

    def run_D(self, img, sync, ret_valid=True, ret_code=True, map_code=False):
        if self.augment_pipe is not None:
            img = self.augment_pipe(img)
        with misc.ddp_sync(self.D, sync):
            logits, features, predicted_code, fmapped_pred_code = self.D(
                img, ret_valid=ret_valid, ret_code=ret_code, map_code=map_code
            )
        return logits, features, predicted_code, fmapped_pred_code

    def run_labelmodel(self, features, lambdas, sync):
        with misc.ddp_sync(self.labelmodel, sync):
            yestimate, yest_mapped, accs = self.labelmodel(
                features.detach(), lambdas, get_accuracies=True
            )

        return yestimate, yest_mapped, accs

    def accumulate_gradients(
        self,
        phase,
        real_img,
        real_c,
        gen_z,
        gen_c,
        gen_c_abs,
        sync,
        gain,
        tick,
        LFs,
        filteridx,
    ):
        assert phase in GPHASES
        do_Gmain = phase in ("Gmain", "Gboth")
        do_Dmain = phase in ("Dmain", "Dboth")
        do_Gpl = (phase in ("Greg", "Gboth")) and (self.pl_weight != 0)
        do_Dr1 = (phase in ("Dreg", "Dboth")) and (self.r1_gamma != 0)
        do_info = phase == "Info"
        do_ymap = phase == "Ymap"

        # Compute info loss
        if do_info:
            with torch.autograd.profiler.record_function("Info_forward"):
                gen_img, _ = self.run_G(gen_z, gen_c, sync=sync)
                _, _, predicted_code, _ = self.run_D(
                    gen_img, sync=sync, ret_valid=False, ret_code=True
                )
                # predicted_code are logits
                # gen_c_abs are the sampled discrete codes/labels (not one-hot)
                loss_info = F.cross_entropy(predicted_code, gen_c_abs)
                training_stats.report("Loss/Info_loss", loss_info)
            with torch.autograd.profiler.record_function("Info_backward"):
                # loss_info.mul(gain).backward()
                loss_info.backward()

        if do_ymap:
            # first, check if we have LF outputs in the batch
            # filteridx contains indices for samples where we have at least 1 LF vote
            filtersum = filteridx.sum()

            with torch.autograd.profiler.record_function("LM_forward"):
                real_img_tmp = real_img.detach().requires_grad_(False)

                _, features, predicted_code, predicted_code_mapped = self.run_D(
                    real_img_tmp,
                    sync=sync,
                    ret_valid=False,
                    ret_code=True,
                    map_code=True,
                )

                yestimate, yestimate_mapped, accs = self.run_labelmodel(
                    features, LFs, sync
                )
                lfloss = soft_cross_entropy_noreduce(
                    predicted_code_mapped, yestimate.detach()
                )
                mult = torch.zeros(lfloss.shape[0]).to(real_img.device)
                mult[filteridx] = 1.0
                denom = mult.sum() + 1e-8
                lfloss = (lfloss.flatten() * mult).sum() / denom
                lflosstmp = soft_cross_entropy_noreduce(
                    yestimate_mapped,
                    F.softmax(predicted_code.detach(), dim=1),
                )
                lflosstmp = (lflosstmp.flatten() * mult).sum() / denom
                lfloss += lflosstmp
                training_stats.report("Loss/LFloss", lfloss)

                # Add decaying loss term to keep labelmodel weights close to uniform in initial epochs.
                # Instead of epochs, we'll use ticks here, consider adjusting this in the future.
                decaylossparam = 0.1  # TODO make this a config parameter
                decayloss = (
                    self.n_classes
                    / (tick * decaylossparam + 1.0)
                    * F.mse_loss(accs, torch.ones_like(accs) * 0.5)
                    / self.num_LFs
                )
                training_stats.report("Loss/decayloss", decayloss)
                lfloss += decayloss

            if filtersum > 0:
                # log accuracy etc on non-abstains
                with torch.no_grad():
                    y_sub = real_c[filteridx].type(torch.int64)
                    pred_label_crisp = torch.argmax(yestimate[filteridx], 1)
                    intlabel = torch.argmax(y_sub, dim=1)  # convert from one-hot
                    acc = torchmetrics.functional.accuracy(pred_label_crisp, intlabel)
                    training_stats.report("Posterior/accuracy", acc)
            with torch.autograd.profiler.record_function("LM_backward"):
                lfloss.backward()

        # Gmain: Maximize logits for generated images.
        if do_Gmain:
            with torch.autograd.profiler.record_function("Gmain_forward"):
                gen_img, _gen_ws = self.run_G(
                    gen_z, gen_c, sync=(sync and not do_Gpl)
                )  # May get synced by Gpl.
                gen_logits, features, predicted_code, _ = self.run_D(
                    gen_img, sync=False, ret_valid=True, ret_code=False
                )
                training_stats.report("Loss/scores/fake", gen_logits)
                training_stats.report("Loss/signs/fake", gen_logits.sign())
                loss_Gmain = torch.nn.functional.softplus(-gen_logits)
                training_stats.report("Loss/G/loss", loss_Gmain)
            with torch.autograd.profiler.record_function("Gmain_backward"):
                loss_Gmain.mean().mul(gain).backward()

        # Gpl: Apply path length regularization.
        if do_Gpl:
            with torch.autograd.profiler.record_function("Gpl_forward"):
                batch_size = gen_z.shape[0] // self.pl_batch_shrink
                gen_img, gen_ws = self.run_G(
                    gen_z[:batch_size], gen_c[:batch_size], sync=sync
                )
                pl_noise = torch.randn_like(gen_img) / np.sqrt(
                    gen_img.shape[2] * gen_img.shape[3]
                )
                with torch.autograd.profiler.record_function(
                    "pl_grads"
                ), conv2d_gradfix.no_weight_gradients():
                    pl_grads = torch.autograd.grad(
                        outputs=[(gen_img * pl_noise).sum()],
                        inputs=[gen_ws],
                        create_graph=True,
                        only_inputs=True,
                    )[0]
                pl_lengths = pl_grads.square().sum(2).mean(1).sqrt()
                pl_mean = self.pl_mean.lerp(pl_lengths.mean(), self.pl_decay)
                self.pl_mean.copy_(pl_mean.detach())
                pl_penalty = (pl_lengths - pl_mean).square()
                training_stats.report("Loss/pl_penalty", pl_penalty)
                loss_Gpl = pl_penalty * self.pl_weight
                training_stats.report("Loss/G/reg", loss_Gpl)
            with torch.autograd.profiler.record_function("Gpl_backward"):
                (gen_img[:, 0, 0, 0] * 0 + loss_Gpl).mean().mul(gain).backward()

        # Dmain: Minimize logits for generated images.
        loss_Dgen = 0
        if do_Dmain:
            with torch.no_grad():
                gen_img, _ = self.run_G(gen_z, gen_c, sync=False)
            with torch.autograd.profiler.record_function("Dgen_forward"):
                gen_logits, _, _, _ = self.run_D(
                    gen_img, sync=False, ret_valid=True, ret_code=False
                )  # Gets synced by loss_Dreal.
                training_stats.report("Loss/scores/fake", gen_logits)
                training_stats.report("Loss/signs/fake", gen_logits.sign())
                loss_Dgen = torch.nn.functional.softplus(gen_logits)
            with torch.autograd.profiler.record_function("Dgen_backward"):
                loss_Dgen.mean().mul(gain).backward()

        # Dmain: Maximize logits for real images.
        # Dr1: Apply R1 regularization.
        if do_Dmain or do_Dr1:
            name = (
                "Dreal_Dr1" if do_Dmain and do_Dr1 else "Dreal" if do_Dmain else "Dr1"
            )
            with torch.autograd.profiler.record_function(name + "_forward"):
                real_img_tmp = real_img.detach().requires_grad_(do_Dr1)
                real_logits, features, predicted_code, _ = self.run_D(
                    real_img_tmp, sync=sync, ret_valid=True, ret_code=False
                )
                training_stats.report("Loss/scores/real", real_logits)
                training_stats.report("Loss/signs/real", real_logits.sign())

                loss_Dreal = 0
                if do_Dmain:
                    loss_Dreal = torch.nn.functional.softplus(
                        -real_logits
                    )  # -log(sigmoid(real_logits))
                    training_stats.report("Loss/D/loss", loss_Dgen + loss_Dreal)

                loss_Dr1 = 0
                if do_Dr1:
                    with torch.autograd.profiler.record_function(
                        "r1_grads"
                    ), conv2d_gradfix.no_weight_gradients():
                        r1_grads = torch.autograd.grad(
                            outputs=[real_logits.sum()],
                            inputs=[real_img_tmp],
                            create_graph=True,
                            only_inputs=True,
                        )[0]
                    r1_penalty = r1_grads.square().sum([1, 2, 3])
                    loss_Dr1 = r1_penalty * (self.r1_gamma / 2)
                    training_stats.report("Loss/r1_penalty", r1_penalty)
                    training_stats.report("Loss/D/reg", loss_Dr1)

            with torch.autograd.profiler.record_function(name + "_backward"):
                (real_logits * 0 + loss_Dreal + loss_Dr1).mean().mul(gain).backward()


# ----------------------------------------------------------------------------
