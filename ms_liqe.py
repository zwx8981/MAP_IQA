"""Multi-scale LIQE quality model used by the ms_liqe option."""

import torch
import clip
import torch.nn.functional as F
from torch import nn
from torchvision.transforms import Normalize

qualitys = ['bad', 'poor', 'fair', 'good', 'perfect']


class MSLIQE(nn.Module):
    def __init__(self, ckpt, device, scale=[1]):
        super(MSLIQE, self).__init__()
        self.model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
        checkpoint = torch.load(ckpt, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        joint_texts = torch.cat(
            [clip.tokenize(f"a photo with {q} quality") for q in qualitys]).to(device)
        with torch.no_grad():
            self.text_features = self.model.encode_text(joint_texts)
            self.text_features = self.text_features / self.text_features.norm(dim=1, keepdim=True)
        self.step = 32
        if len(scale) > 1:
            self.num_patch = 7
        else:
            self.num_patch = 15
        self.normalize = Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        self.device = device
        self.scale = scale

    def forward(self, x):
        x = x.to(self.device)
        batch_size = x.size(0)
        x = self.normalize(x)

        I_global = x
        I_global = F.interpolate(I_global, size=(224,224), mode='bicubic')
        I_global = I_global.unsqueeze(1)

        for scale in self.scale:
            if scale != 1:
                xs = F.interpolate(x,  scale_factor=scale, mode='bicubic')
            else:
                xs = x
            xs = xs.unfold(2, 224, self.step).unfold(3, 224, self.step).permute(0, 2, 3, 1, 4, 5).reshape(batch_size, -1, 3, 224, 224)

            sel_step = xs.size(1) // self.num_patch
            sel = torch.zeros(self.num_patch)
            for i in range(self.num_patch):
                sel[i] = sel_step * i
            sel = sel.long()
            xs = xs[:, sel, ...]
            I_global = torch.cat([I_global, xs], 1)

        x = I_global
        x = x.view(batch_size*x.size(1), x.size(2), x.size(3), x.size(4))

        image_features = self.model.encode_image(x)

        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.model.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ self.text_features.t()

        logits_per_image = logits_per_image.view(batch_size, 15, -1)

        logits_per_image = logits_per_image.mean(1)

        logits_per_image = F.softmax(logits_per_image/2, dim=1)




        quality = 1 * logits_per_image[:, 0] + 2 * logits_per_image[:, 1] + 3 * logits_per_image[:, 2] + \
                    4 * logits_per_image[:, 3] + 5 * logits_per_image[:, 4]
        return quality
