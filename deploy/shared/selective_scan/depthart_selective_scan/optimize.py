import hashlib
import types

import torch
import torch.nn.functional as F

from .integration import install_depthart


def parameter_fingerprint(model):
    """Hash state-dict names, shapes, dtypes and values for invariance checks."""
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def disable_unused_auxiliary_features(model):
    """Stop producing high/low-frequency debug outputs discarded by DPTHead.

    The model instance, modules, parameters and state-dict schema are untouched.
    Only the instance's forward method skips values that the released forward
    computes and immediately discards.
    """
    if not hasattr(model, "pretrained") or not hasattr(model, "depth_head"):
        raise TypeError("expected a DepthART relative TinyVimDepth model")
    if hasattr(model, "_depthart_original_forward"):
        return model

    model._depthart_original_forward = model.forward

    def forward_without_auxiliary(self, x):
        features = self.pretrained(x, return_high_freq=False)
        return self.depth_head(features=features, size_h=x.shape[-2], size_w=x.shape[-1])

    model.forward = types.MethodType(forward_without_auxiliary, model)
    return model


def restore_auxiliary_forward(model):
    original = getattr(model, "_depthart_original_forward", None)
    if original is not None:
        model.forward = original
        delattr(model, "_depthart_original_forward")
    return model


def optimize_depthart_inference(model, tvimblock_module, install_scan=True):
    """Apply parameter-preserving eager inference optimizations."""
    before = parameter_fingerprint(model)
    model.eval()
    disable_unused_auxiliary_features(model)
    if install_scan:
        install_depthart(tvimblock_module)
    after = parameter_fingerprint(model)
    if before != after:
        raise RuntimeError("optimization unexpectedly changed model parameters")
    return model


def cache_metric_camera_embeddings(model, K, height=448, width=448):
    """Cache fixed-camera Metric DepthART embeddings without changing parameters."""
    required = ("cam_embedder", "pretrained", "daa1", "daa2", "daa3", "daa4", "depth_head", "sfh")
    if not all(hasattr(model, name) for name in required):
        raise TypeError("expected a K-aware MetricDepthART model")
    if hasattr(model, "_depthart_metric_original_forward"):
        raise RuntimeError("metric camera embeddings are already cached")
    before = parameter_fingerprint(model)
    device = next(model.parameters()).device
    K = K.detach().to(device=device, dtype=torch.float32).contiguous()
    with torch.inference_mode():
        cameras = tuple(
            value.detach() for value in model.cam_embedder(K, height, width, device)
        )
    model._depthart_metric_original_forward = model.forward
    model._depthart_cached_cameras = cameras
    model._depthart_cached_K = K
    model._depthart_cached_shape = (height, width)

    def forward_with_cached_camera(self, image, K_unused=None):
        del K_unused
        if image.shape[-2:] != self._depthart_cached_shape:
            raise ValueError(
                f"cached camera shape is {self._depthart_cached_shape}, got {tuple(image.shape[-2:])}"
            )
        cameras = self._depthart_cached_cameras
        features = self.pretrained.forward_with_adapters(
            image,
            adapters=[self.daa1, self.daa2, self.daa3, self.daa4],
            cams=list(cameras),
        )
        depth = self.depth_head(features, image.shape[-2], image.shape[-1])
        scale = self.sfh(features[3], cameras[3])
        return (depth * scale.view(-1, 1, 1, 1) * self.max_depth).squeeze(1)

    model.forward = types.MethodType(forward_with_cached_camera, model)
    if parameter_fingerprint(model) != before:
        raise RuntimeError("camera caching unexpectedly changed model parameters")
    return model


def restore_metric_camera_forward(model):
    original = getattr(model, "_depthart_metric_original_forward", None)
    if original is not None:
        model.forward = original
        delattr(model, "_depthart_metric_original_forward")
        delattr(model, "_depthart_cached_cameras")
        delattr(model, "_depthart_cached_K")
        delattr(model, "_depthart_cached_shape")
    return model


def cache_metric_daa_attention_kv(model):
    """Cache fixed-camera DAA K/V projections without changing model state."""
    if not hasattr(model, "_depthart_cached_cameras"):
        raise RuntimeError("cache metric camera embeddings before caching DAA K/V")
    stages = (model.daa1, model.daa2, model.daa3, model.daa4)
    if any(hasattr(stage, "_depthart_original_forward") for stage in stages):
        raise RuntimeError("metric DAA K/V projections are already cached")
    before = parameter_fingerprint(model)

    for stage, camera in zip(stages, model._depthart_cached_cameras):
        attention = stage.cross_attention
        with torch.inference_mode():
            camera_projected = stage.proj_cam_pw(stage.proj_cam_dw(camera))
            camera_tokens = camera_projected.flatten(2).transpose(1, 2)
            context = attention.norm_attnctx(camera_tokens)
            key, value = (
                attention.kv(context)
                .view(context.shape[0], context.shape[1], 2, attention.num_heads, -1)
                .permute(0, 3, 1, 4, 2)
                .unbind(dim=-1)
            )
        stage._depthart_original_forward = stage.forward
        stage._depthart_cached_key = key.detach()
        stage._depthart_cached_value = value.detach()
        stage._depthart_cached_spatial_shape = camera.shape[-2:]

        def forward_with_cached_kv(self, feat, camera_unused=None):
            del camera_unused
            batch, channels, height, width = feat.shape
            if (height, width) != self._depthart_cached_spatial_shape:
                raise ValueError(
                    f"cached DAA shape is {self._depthart_cached_spatial_shape}, got {(height, width)}"
                )
            tokens = feat.flatten(2).transpose(1, 2)
            block = self.cross_attention
            query = block.q(block.norm_attnx(tokens))
            query = query.view(batch, height * width, block.num_heads, -1).permute(0, 2, 1, 3)
            key = self._depthart_cached_key
            value = self._depthart_cached_value
            if key.shape[0] != batch:
                key = key.expand(batch, -1, -1, -1)
                value = value.expand(batch, -1, -1, -1)
            if key.dtype != query.dtype:
                key = key.to(query.dtype)
                value = value.to(query.dtype)
            attended = F.scaled_dot_product_attention(
                query, key, value, dropout_p=block.dropout, attn_mask=None
            )
            attended = attended.permute(0, 2, 1, 3).reshape(batch, height * width, channels)
            output = block.ls1(block.out(attended)) + tokens
            output = block.ls2(block.mlp(output)) + output
            return output.transpose(1, 2).reshape(batch, channels, height, width)

        stage.forward = types.MethodType(forward_with_cached_kv, stage)

    if parameter_fingerprint(model) != before:
        raise RuntimeError("DAA K/V caching unexpectedly changed model parameters")
    return model


def restore_metric_daa_attention(model):
    for stage in (model.daa1, model.daa2, model.daa3, model.daa4):
        original = getattr(stage, "_depthart_original_forward", None)
        if original is None:
            continue
        stage.forward = original
        delattr(stage, "_depthart_original_forward")
        delattr(stage, "_depthart_cached_key")
        delattr(stage, "_depthart_cached_value")
        delattr(stage, "_depthart_cached_spatial_shape")
    return model


class CUDAGraphRunner:
    """Static-shape CUDA Graph runner that owns no model parameters."""

    def __init__(self, model, example, amp=False, warmup=50):
        if not example.is_cuda:
            raise ValueError("CUDA Graph requires a CUDA example tensor")
        self.model = model
        self.static_input = torch.empty_like(example)
        self.static_input.copy_(example)
        self.amp = amp
        self.graph = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream), torch.inference_mode():
            for _ in range(warmup):
                with torch.cuda.amp.autocast(enabled=amp, dtype=torch.float16):
                    self.static_output = model(self.static_input)
        stream.synchronize()
        torch.cuda.current_stream().wait_stream(stream)
        with torch.cuda.graph(self.graph), torch.inference_mode():
            with torch.cuda.amp.autocast(enabled=amp, dtype=torch.float16):
                self.static_output = model(self.static_input)

    def copy_input(self, value):
        self.static_input.copy_(value)

    def replay(self):
        """Replay and return a buffer overwritten by the next replay."""
        self.graph.replay()
        return self.static_output

    def __call__(self, value):
        self.copy_input(value)
        return self.replay()
