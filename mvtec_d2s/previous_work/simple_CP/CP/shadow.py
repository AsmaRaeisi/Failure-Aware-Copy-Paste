from .lib import *


class ShadowSynthesizer:
    """Physically-inspired soft-shadow generator.

    For every object instance it casts a directional, blurred ("penumbra")
    shadow plus a tight contact shadow, driven only by the instance mask.
    The result is a single-channel shadow map in [0, 1].

    Casting is *depth-aware*: the instance id encodes paste order (higher id
    was pasted later and therefore sits on top). A shadow may fall on the
    background or on objects *below* the caster, but never on objects that
    are stacked on top of it.
    """

    def __init__(self, shadow_dir=(0.5, 1.0), offset_scale=0.12,
                 penumbra=0.04, strength=0.60, contact=0.85,
                 jitter=0.15, cast_on_objects=True, seed=42):
        d = np.asarray(shadow_dir, np.float32)
        self.dir = d / (np.linalg.norm(d) + 1e-8)   # direction the shadow is cast
        self.offset_scale = offset_scale            # cast distance, fraction of obj size
        self.penumbra = penumbra                    # soft-edge blur, fraction of obj size
        self.strength = strength                    # opacity of the cast shadow
        self.contact = contact                      # opacity of the contact shadow
        self.jitter = jitter                        # per-object randomness
        self.cast_on_objects = cast_on_objects      # allow inter-object shadows
        self.rng = np.random.default_rng(seed)

    def __call__(self, instance_map):
        H, W = instance_map.shape
        shadow = np.zeros((H, W), np.float32)
        obj_ids = [int(i) for i in np.unique(instance_map) if i != 0]

        for oid in obj_ids:
            mask = (instance_map == oid).astype(np.float32)
            ys, xs = np.nonzero(mask)
            if xs.size == 0:
                continue
            size = max(xs.max() - xs.min(), ys.max() - ys.min()) + 1

            # per-object jittered cast direction & distance
            jit = 1.0 + self.rng.uniform(-self.jitter, self.jitter)
            off = self.dir * size * self.offset_scale * jit
            M = np.float32([[1, 0, off[0]], [0, 1, off[1]]])
            shifted = cv2.warpAffine(mask, M, (W, H))

            # penumbra (soft, offset) + contact (tight, under the object)
            sig_p = max(1.0, size * self.penumbra)
            sig_c = max(1.0, sig_p * 0.4)

            size_factor = np.clip(size / 400.0, 0.7, 1.3)

            penumbra = (
                cv2.GaussianBlur(shifted, (0, 0), sig_p)
                * self.strength
                * size_factor
            )

            contact = (
                cv2.GaussianBlur(mask, (0, 0), sig_c)
                * self.contact
                * size_factor
            )

            s = np.maximum(penumbra, contact)

            # Where this shadow is allowed to land (depth-aware):
            #   - always on the background (id == 0)
            #   - on objects *below* the caster (lower id) only if enabled
            #   - never on the caster itself or on objects on top of it (higher id)
            if self.cast_on_objects:
                receivable = (instance_map < oid)          # background + objects below
            else:
                receivable = (instance_map == 0)           # background only
            s *= receivable.astype(np.float32)

            shadow = np.maximum(shadow, s)

        return np.clip(shadow, 0.0, 1.0)


def apply_shadow(img, shadow, max_darken=0.6, tint=(0, 0, 0)):
    """Composite a shadow map onto a BGR image (multiplicative darkening)."""
    a = (shadow * max_darken)[..., None].astype(np.float32)
    tint = np.asarray(tint, np.float32)
    out = img.astype(np.float32) * (1.0 - a) + tint * a
    return np.clip(out, 0, 255).astype(np.uint8)
