"""Velocity goal semantics from HybridRobotics/supermimic's make_velocity_task.

Centered mean in world coordinates, project into the actual root body frame,
add fresh uniform command noise, clamp. Unlike upstream, smoothing is clamped
within each clip so adjacent concatenated motions cannot contaminate commands.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from supermimic.kinematics.rotations import quat_apply_inverse


@dataclass(frozen=True)
class VelocityTaskCfg:
    lin_vel_x_range: tuple[float, float] = (-1.5, 1.5)
    lin_vel_y_range: tuple[float, float] = (-0.5, 0.5)
    ang_vel_z_range: tuple[float, float] = (-3.14, 3.14)
    window_frames: int = 25
    command_noise: tuple[float, float, float] = (0.5, 0.5, 0.785)
    latent_bc_weight: float = 0.1
    velocity_fraction: float = 0.5

    def validate(self):
        ranges = (self.lin_vel_x_range, self.lin_vel_y_range, self.ang_vel_z_range)
        if any(
            len(r) != 2 or not torch.isfinite(torch.tensor(r)).all() or r[0] >= r[1] for r in ranges
        ):
            raise ValueError("Invalid velocity command ranges")
        if self.window_frames < 0 or len(self.command_noise) != 3:
            raise ValueError("Invalid velocity smoothing/noise shape")
        if (
            not torch.isfinite(torch.tensor(self.command_noise)).all()
            or min(self.command_noise) < 0
        ):
            raise ValueError("Invalid velocity command noise")
        if not 0 < self.velocity_fraction < 1 or not 0 <= self.latent_bc_weight < float("inf"):
            raise ValueError("Need both interfaces and a finite nonnegative BC weight")


def centered_mean(track, window):
    """One clip, arbitrary length (including shorter than the averaging window)."""
    if window <= 1:
        return track.clone()
    half = window // 2
    padded = F.pad(track.T.unsqueeze(0), (half, window - 1 - half), mode="replicate")
    return F.avg_pool1d(padded, window, stride=1)[0].T.contiguous()


class VelocityCommands:
    def __init__(self, library, cfg):
        cfg.validate()
        self.cfg, self.lib = cfg, library
        self.low = torch.tensor(
            [cfg.lin_vel_x_range[0], cfg.lin_vel_y_range[0], cfg.ang_vel_z_range[0]],
            device=library.device,
        )
        self.high = torch.tensor(
            [cfg.lin_vel_x_range[1], cfg.lin_vel_y_range[1], cfg.ang_vel_z_range[1]],
            device=library.device,
        )
        self.noise = torch.tensor(cfg.command_noise, device=library.device)
        # Extra six scalars per source frame on the SAME storage device as the
        # library. CPU-backed motion data never becomes a full GPU command table.
        track = torch.cat([library.root_lin_vel_w, library.root_ang_vel_w], dim=-1)
        self.track = torch.empty_like(track)
        for i, clip in enumerate(library.clips, 1):
            sl = slice(clip.start, clip.start + clip.length)
            self.track[sl] = centered_mean(track[sl], cfg.window_frames)
            if len(library.clips) >= 1000 and (i % 10000 == 0 or i == len(library.clips)):
                print(f"[VELOCITY] smoothed {i}/{len(library.clips)} clips", flush=True)

    def world_velocity(self, frame):
        rows = self.track[frame.detach().to(self.track.device)]
        asynchronous = (
            self.lib.library_transfer == "pinned"
            and frame.device.type == "cuda"
            and rows.device.type == "cpu"
        )
        if asynchronous:
            rows = rows.pin_memory()
        return rows.to(frame.device, non_blocking=asynchronous)

    def project(self, world_velocity, root_quat, *, noisy=True):
        v = quat_apply_inverse(root_quat, world_velocity[..., :3])
        w = quat_apply_inverse(root_quat, world_velocity[..., 3:])
        command = torch.stack([v[..., 0], v[..., 1], w[..., 2]], -1)
        if noisy:
            command = command + (torch.rand_like(command) * 2 - 1) * self.noise
        return command.clamp(self.low, self.high)

    def __call__(self, frame, root_quat, *, noisy=True):
        return self.project(self.world_velocity(frame), root_quat, noisy=noisy)
