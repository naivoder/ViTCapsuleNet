import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import copy
import math

USE_CUDA = True if torch.cuda.is_available() else False

class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, num_classes=10, zero_head=False, vis=False):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = 'token'

        self.transformer = Transformer(img_size, vis)
        self.head = nn.Linear(768, num_classes)

    def forward(self, x, labels=None):
        x, _ = self.transformer(x)
        logits = self.head(x[:, 0])
        return logits


class ViTCapsNet(nn.Module):
    def __init__(self, img_size, num_classes=10):
        super(ViTCapsNet, self).__init__()
        self.vit = VisionTransformer(img_size=img_size, num_classes=num_classes)
        self.primary_capsules = PrimaryCaps(num_capsules=8, in_channels=768, out_channels=32, kernel_size=9, num_routes=32 * 6 * 6)
        self.digit_capsules = DigitCaps(num_capsules=num_classes, num_routes=32 * 6 * 6, in_channels=8, out_channels=16)
        self.decoder = Decoder(input_width=img_size, input_height=img_size, input_channel=3)
        self.mse_loss = nn.MSELoss()

    def forward(self, data):
        vit_output = self.vit(data)
        vit_output = vit_output.unsqueeze(2).unsqueeze(3)
        output = self.digit_capsules(self.primary_capsules(vit_output))
        recons, masked = self.decoder(output, data)
        return output, recons, masked

    def loss(self, data, x, target, recons):
        return self.margin_loss(x, target) + self.reconstruction_loss(data, recons)

    def margin_loss(self, x, labels):
        batch_size = x.size(0)
        v_c = torch.sqrt((x ** 2).sum(dim=2, keepdim=True))
        left = F.relu(0.9 - v_c).view(batch_size, -1)
        right = F.relu(v_c - 0.1).view(batch_size, -1)
        loss = labels * left + 0.5 * (1.0 - labels) * right
        loss = loss.sum(dim=1).mean()
        return loss

    def reconstruction_loss(self, data, recons):
        loss = self.mse_loss(recons.view(recons.size(0), -1), data.view(recons.size(0), -1))
        return loss * 0.0005


class Transformer(nn.Module):
    def __init__(self, img_size, vis):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(img_size=img_size)
        self.encoder = Encoder(vis)

    def forward(self, input_ids):
        embedding_output = self.embeddings(input_ids)
        encoded, attn_weights = self.encoder(embedding_output)
        return encoded, attn_weights


class Embeddings(nn.Module):
    def __init__(self, img_size, in_channels=3):
        super(Embeddings, self).__init__()
        img_size = (img_size, img_size)
        patch_size = (16, 16)
        n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
        self.patch_embeddings = nn.Conv2d(in_channels=in_channels, out_channels=768, kernel_size=patch_size, stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches + 1, 768))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 768))
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = self.patch_embeddings(x)
        x = x.flatten(2)
        x = x.transpose(-1, -2)
        x = torch.cat((cls_tokens, x), dim=1)
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings


class Encoder(nn.Module):
    def __init__(self, vis):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = nn.LayerNorm(768, eps=1e-6)
        for _ in range(12):
            layer = Block(vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights


class Block(nn.Module):
    def __init__(self, vis):
        super(Block, self).__init__()
        self.hidden_size = 768
        self.attention_norm = nn.LayerNorm(768, eps=1e-6)
        self.ffn_norm = nn.LayerNorm(768, eps=1e-6)
        self.ffn = Mlp()
        self.attn = Attention(vis)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h
        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights


class Attention(nn.Module):
    def __init__(self, vis):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = 12
        self.attention_head_size = int(768 / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(768, self.all_head_size)
        self.key = nn.Linear(768, self.all_head_size)
        self.value = nn.Linear(768, self.all_head_size)
        self.out = nn.Linear(768, 768)
        self.attn_dropout = nn.Dropout(0.0)
        self.proj_dropout = nn.Dropout(0.0)
        self.softmax = nn.Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights


class Mlp(nn.Module):
    def __init__(self):
        super(Mlp, self).__init__()
        self.fc1 = nn.Linear(768, 3072)
        self.fc2 = nn.Linear(3072, 768)
        self.act_fn = nn.GELU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class PrimaryCaps(nn.Module):
    def __init__(self, num_capsules=8, in_channels=768, out_channels=32, kernel_size=9, num_routes=32 * 6 * 6):
        super(PrimaryCaps, self).__init__()
        self.num_routes = num_routes
        self.capsules = nn.ModuleList([
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=2, padding=0)
            for _ in range(num_capsules)])

    def forward(self, x):
        u = [capsule(x) for capsule in self.capsules]
        u = torch.stack(u, dim=1)
        u = u.view(x.size(0), self.num_routes, -1)
        return self.squash(u)

    def squash(self, input_tensor):
        squared_norm = (input_tensor ** 2).sum(-1, keepdim=True)
        output_tensor = squared_norm * input_tensor / ((1. + squared_norm) * torch.sqrt(squared_norm))
        return output_tensor


class DigitCaps(nn.Module):
    def __init__(self, num_capsules=10, num_routes=32 * 6 * 6, in_channels=8, out_channels=16):
        super(DigitCaps, self).__init__()
        self.in_channels = in_channels
        self.num_routes = num_routes
        self.num_capsules = num_capsules
        self.W = nn.Parameter(torch.randn(1, num_routes, num_capsules, out_channels, in_channels))

    def forward(self, x):
        batch_size = x.size(0)
        x = torch.stack([x] * self.num_capsules, dim=2).unsqueeze(4)
        W = torch.cat([self.W] * batch_size, dim=0)
        u_hat = torch.matmul(W, x)

        b_ij = Variable(torch.zeros(1, self.num_routes, self.num_capsules, 1))
        if USE_CUDA:
            b_ij = b_ij.cuda()

        num_iterations = 3
        for iteration in range(num_iterations):
            c_ij = F.softmax(b_ij, dim=1)
            c_ij = torch.cat([c_ij] * batch_size, dim=0).unsqueeze(4)
            s_j = (c_ij * u_hat).sum(dim=1, keepdim=True)
            v_j = self.squash(s_j)

            if iteration < num_iterations - 1:
                a_ij = torch.matmul(u_hat.transpose(3, 4), torch.cat([v_j] * self.num_routes, dim=1))
                b_ij = b_ij + a_ij.squeeze(4).mean(dim=0, keepdim=True)

        return v_j.squeeze(1)

    def squash(self, input_tensor):
        squared_norm = (input_tensor ** 2).sum(-1, keepdim=True)
        output_tensor = squared_norm * input_tensor / ((1. + squared_norm) * torch.sqrt(squared_norm))
        return output_tensor


class Decoder(nn.Module):
    def __init__(self, input_width=28, input_height=28, input_channel=1):
        super(Decoder, self).__init__()
        self.input_width = input_width
        self.input_height = input_height
        self.input_channel = input_channel
        self.reconstraction_layers = nn.Sequential(
            nn.Linear(16 * 10, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, self.input_height * self.input_width * self.input_channel),
            nn.Sigmoid()
        )

    def forward(self, x, data):
        classes = torch.sqrt((x ** 2).sum(2))
        classes = F.softmax(classes, dim=0)

        _, max_length_indices = classes.max(dim=1)
        masked = Variable(torch.sparse.torch.eye(10))
        if USE_CUDA:
            masked = masked.cuda()
        masked = masked.index_select(dim=0, index=Variable(max_length_indices.squeeze(1).data))
        t = (x * masked[:, :, None, None]).view(x.size(0), -1)
        reconstructions = self.reconstraction_layers(t)
        reconstructions = reconstructions.view(-1, self.input_channel, self.input_width, self.input_height)
        return reconstructions, masked

if __name__=="__main__":
    img_size = 128
    num_classes = 10
    model = ViTCapsNet(img_size, num_classes)
    for name, _ in model.named_parameters():
        print(name)
