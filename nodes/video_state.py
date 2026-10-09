"""
SAM3 Video State - Immutable state for stateless video tracking

This module provides immutable state structures that flow through ComfyUI nodes
for video tracking. Instead of global mutable state, all session data is encoded
in the output and reconstructed on-demand.

Key design principles:
1. All state is immutable (frozen dataclasses)
2. Adding prompts returns a NEW state (doesn't mutate)
3. Inference state is reconstructed on-demand
4. Frame directories live on disk, capped by total size (see create_session_frames)
"""
import logging
import os
import uuid
import shutil
import tempfile
import weakref
from collections import OrderedDict
import atexit
from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, Any, List
from pathlib import Path

log = logging.getLogger("sam3")


# =============================================================================
# Session frame directories: on real disk, capped by total size
# =============================================================================
#
# Each run of SAM3 Video Segmentation extracts its frames as JPEGs into a
# fresh, uniquely named directory of its own.  Nothing is ever identified by
# content: ComfyUI's own caching decides when the node re-runs, and a session
# can only ever read the directory it created, so it can't get another
# session's frames.
#
# These directories used to go into tempfile.gettempdir() (usually /tmp, RAM
# on many distros), were kept until process exit, and were orphaned forever
# after a crash.  Now:
#   * they live under SAM3_TEMP_DIR, default ~/.cache/comfyui-sam3 (real disk);
#   * their total size is capped at SAM3_FRAME_CACHE_GB (default 10 GB): when
#     a new session pushes the total over the cap, the least recently used
#     sessions are deleted (the newest one is always kept);
#   * leftovers from earlier runs are deleted the first time the root is used.

_SESSIONS = OrderedDict()       # session_uuid -> {"dir", "bytes"}, least recently used first
_EVICTION_LISTENERS = []        # callables(session_uuid) run when a session's frames go away
_ROOT = None
_CLEANUP_REGISTERED = False
_DIR_PREFIX = "sam3_"


def frame_cache_limit_bytes() -> int:
    try:
        gb = float(os.environ.get("SAM3_FRAME_CACHE_GB", "10"))
    except ValueError:
        gb = 10.0
    return int(max(gb, 0.0) * 1024**3)


def _temp_root() -> str:
    """Dedicated root for SAM3 scratch data; stale contents purged on first use."""
    global _ROOT
    if _ROOT is None:
        root = os.environ.get("SAM3_TEMP_DIR") or os.path.join(
            os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache"),
            "comfyui-sam3",
        )
        os.makedirs(root, exist_ok=True)
        # Everything under this root belongs to SAM3, so anything present now
        # was left behind by an earlier (possibly crashed) ComfyUI run.
        for name in os.listdir(root):
            if name.startswith(_DIR_PREFIX):
                path = os.path.join(root, name)
                if os.path.isdir(path) and not os.path.islink(path):
                    shutil.rmtree(path, ignore_errors=True)
                else:
                    try:
                        os.remove(path)
                    except OSError:
                        pass
                log.info(f"Removed stale SAM3 scratch item: {name}")
        _ROOT = root
        log.info(f"SAM3 video frame cache: {root} "
                 f"(limit {frame_cache_limit_bytes() / 1024**3:.1f} GB)")
    return _ROOT


def register_eviction_listener(fn):
    """fn(session_uuid) is called just before a session's frames are deleted,
    so caches that refer to the session can drop their entries."""
    _EVICTION_LISTENERS.append(fn)


def _drop_session(session_uuid: str):
    entry = _SESSIONS.pop(session_uuid, None)
    for fn in _EVICTION_LISTENERS:
        try:
            fn(session_uuid)
        except Exception as e:
            log.warning(f"SAM3 session eviction hook failed: {e}")
    if entry:
        shutil.rmtree(entry["dir"], ignore_errors=True)
        log.info(f"Discarded frames of video session {session_uuid[:8]} "
                 f"({entry['bytes'] / 1024**2:.0f} MB)")


def _cleanup_temp_dirs():
    """Clean up all session directories at process exit."""
    for session_uuid in list(_SESSIONS):
        _drop_session(session_uuid)


def _ensure_cleanup_registered():
    """Register cleanup function with atexit (once)."""
    global _CLEANUP_REGISTERED
    if not _CLEANUP_REGISTERED:
        atexit.register(_cleanup_temp_dirs)
        _CLEANUP_REGISTERED = True


def _dir_bytes(path: str) -> int:
    total = 0
    for entry in os.scandir(path):
        if entry.is_file(follow_symlinks=False):
            total += entry.stat(follow_symlinks=False).st_size
    return total


def create_session_frames(session_uuid: str, extract_fn) -> str:
    """
    Create the frame directory for a new session and fill it.

    extract_fn(dir) must write 00000.jpg, 00001.jpg, ... into dir.  If it
    fails or is cancelled, the directory is removed and nothing is recorded.
    Only after it succeeds are older sessions evicted to respect the size cap,
    so a cancelled run never costs other node chains their frames.
    """
    _ensure_cleanup_registered()
    temp_dir = tempfile.mkdtemp(prefix=f"{_DIR_PREFIX}{session_uuid[:8]}_", dir=_temp_root())
    try:
        extract_fn(temp_dir)
    except BaseException:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise
    _SESSIONS[session_uuid] = {"dir": temp_dir, "bytes": _dir_bytes(temp_dir)}

    limit = frame_cache_limit_bytes()
    total = sum(e["bytes"] for e in _SESSIONS.values())
    while total > limit and len(_SESSIONS) > 1:
        oldest = next(iter(_SESSIONS))
        if oldest == session_uuid:
            break
        total -= _SESSIONS[oldest]["bytes"]
        _drop_session(oldest)
    return temp_dir


def touch_session(session_uuid: str):
    """Mark a session as recently used (it is then evicted last)."""
    if session_uuid in _SESSIONS:
        _SESSIONS.move_to_end(session_uuid)


def cleanup_temp_dir(temp_dir: str):
    """Explicitly discard the session that owns temp_dir."""
    for session_uuid, entry in list(_SESSIONS.items()):
        if entry["dir"] == temp_dir:
            _drop_session(session_uuid)
            return
    try:
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
    except Exception:
        pass


def require_session_frames(video_state) -> None:
    """Raise a clear error if this state's frames were already discarded;
    otherwise mark the session as recently used."""
    if not os.path.isdir(video_state.temp_dir):
        raise RuntimeError(
            f"[SAM3 Video] The frames for video session {video_state.session_uuid[:8]} "
            f"were discarded because the frame cache exceeded its size limit "
            f"({frame_cache_limit_bytes() / 1024**3:.1f} GB). Re-run the SAM3 Video "
            f"Segmentation node (e.g. tweak one of its inputs), or raise the limit "
            f"with the environment variable SAM3_FRAME_CACHE_GB."
        )
    touch_session(video_state.session_uuid)


# =============================================================================
# Immutable State Structures
# =============================================================================

@dataclass(frozen=True)
class VideoPrompt:
    """
    Single prompt for a video frame (immutable).

    Attributes:
        frame_idx: Frame index where prompt is applied
        prompt_type: Type of prompt ("point", "box", "text")
        obj_id: Object ID for tracking
        data: Prompt-specific data (tuple for immutability)
    """
    frame_idx: int
    prompt_type: str  # "point", "box", "text"
    obj_id: int
    data: tuple  # Immutable representation of prompt data

    @classmethod
    def create_point(cls, frame_idx: int, obj_id: int, points: list, labels: list) -> 'VideoPrompt':
        """Create a point prompt."""
        return cls(
            frame_idx=frame_idx,
            prompt_type="point",
            obj_id=obj_id,
            data=(tuple(tuple(p) for p in points), tuple(labels))
        )

    @classmethod
    def create_box(cls, frame_idx: int, obj_id: int, box: list, is_positive: bool = True) -> 'VideoPrompt':
        """Create a box prompt."""
        return cls(
            frame_idx=frame_idx,
            prompt_type="box",
            obj_id=obj_id,
            data=(tuple(box), is_positive)
        )

    @classmethod
    def create_text(cls, frame_idx: int, obj_id: int, text: str) -> 'VideoPrompt':
        """Create a text prompt."""
        return cls(
            frame_idx=frame_idx,
            prompt_type="text",
            obj_id=obj_id,
            data=(text,)
        )


@dataclass(frozen=True)
class VideoConfig:
    """
    Video tracking configuration (immutable).

    Contains all the detection and tracking parameters.
    """
    score_threshold_detection: float = 0.3
    new_det_thresh: float = 0.4
    fill_hole_area: int = 16
    assoc_iou_thresh: float = 0.1
    det_nms_thresh: float = 0.1
    hotstart_unmatch_thresh: int = 8
    hotstart_dup_thresh: int = 8
    init_trk_keep_alive: int = 30
    hotstart_delay: int = 15
    decrease_keep_alive_empty: bool = False
    suppress_unmatched_globally: bool = False


@dataclass(frozen=True)
class SAM3VideoState:
    """
    Immutable video session state that flows through node outputs.

    All state needed to reconstruct inference is encoded here.
    This enables truly stateless nodes - no global mutable state needed.

    Attributes:
        session_uuid: Unique session identifier
        temp_dir: Path to temporary directory containing frames
        num_frames: Total number of video frames
        height: Video height in pixels
        width: Video width in pixels
        config: Tracking configuration
        prompts: Tuple of all prompts (immutable)
    """
    session_uuid: str
    temp_dir: str
    num_frames: int
    height: int
    width: int
    config: VideoConfig = field(default_factory=VideoConfig)
    prompts: Tuple[VideoPrompt, ...] = ()

    def with_prompt(self, prompt: VideoPrompt) -> 'SAM3VideoState':
        """
        Return new state with prompt added (immutable update).

        This is the key pattern - instead of mutating, we create a new state.

        Args:
            prompt: VideoPrompt to add

        Returns:
            New SAM3VideoState with prompt appended
        """
        return SAM3VideoState(
            session_uuid=self.session_uuid,
            temp_dir=self.temp_dir,
            num_frames=self.num_frames,
            height=self.height,
            width=self.width,
            config=self.config,
            prompts=self.prompts + (prompt,),
        )

    def with_config(self, **kwargs) -> 'SAM3VideoState':
        """
        Return new state with updated config.

        Args:
            **kwargs: Config parameters to update

        Returns:
            New SAM3VideoState with updated config
        """
        from dataclasses import asdict
        current_config = asdict(self.config)
        current_config.update(kwargs)
        new_config = VideoConfig(**current_config)

        return SAM3VideoState(
            session_uuid=self.session_uuid,
            temp_dir=self.temp_dir,
            num_frames=self.num_frames,
            height=self.height,
            width=self.width,
            config=new_config,
            prompts=self.prompts,
        )

    def get_prompts_for_frame(self, frame_idx: int) -> Tuple[VideoPrompt, ...]:
        """Get all prompts for a specific frame."""
        return tuple(p for p in self.prompts if p.frame_idx == frame_idx)

    def get_object_ids(self) -> Tuple[int, ...]:
        """Get all unique object IDs in prompts."""
        return tuple(sorted(set(p.obj_id for p in self.prompts)))

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize state to dict for ComfyUI compatibility.

        Returns:
            Dict representation of state
        """
        from dataclasses import asdict
        return {
            "session_uuid": self.session_uuid,
            "temp_dir": self.temp_dir,
            "num_frames": self.num_frames,
            "height": self.height,
            "width": self.width,
            "config": asdict(self.config),
            "prompts": [
                {
                    "frame_idx": p.frame_idx,
                    "prompt_type": p.prompt_type,
                    "obj_id": p.obj_id,
                    "data": p.data
                }
                for p in self.prompts
            ]
        }

    @staticmethod
    def _deep_to_tuple(obj):
        """Recursively convert lists to tuples (JSON deserializes tuples as lists)."""
        if isinstance(obj, list):
            return tuple(SAM3VideoState._deep_to_tuple(item) for item in obj)
        return obj

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'SAM3VideoState':
        """
        Deserialize state from dict.

        Args:
            d: Dict representation

        Returns:
            SAM3VideoState instance
        """
        prompts = tuple(
            VideoPrompt(
                frame_idx=p["frame_idx"],
                prompt_type=p["prompt_type"],
                obj_id=p["obj_id"],
                data=cls._deep_to_tuple(p["data"])
            )
            for p in d.get("prompts", [])
        )

        config = VideoConfig(**d.get("config", {}))

        return cls(
            session_uuid=d["session_uuid"],
            temp_dir=d["temp_dir"],
            num_frames=d["num_frames"],
            height=d["height"],
            width=d["width"],
            config=config,
            prompts=prompts,
        )


# =============================================================================
# Factory Functions
# =============================================================================

def create_video_state(
    video_frames,
    config: Optional[VideoConfig] = None,
    session_id: Optional[str] = None
) -> SAM3VideoState:
    """
    Create a new video state from video frames.

    This function:
    1. Generates a unique session UUID
    2. Extracts the frames as JPEG files into a directory of its own
    3. Returns immutable state

    Args:
        video_frames: ComfyUI image tensor [N, H, W, C]
        config: Optional tracking configuration
        session_id: Optional custom session ID (UUID generated if None)

    Returns:
        SAM3VideoState ready for use
    """
    import numpy as np
    from PIL import Image
    import comfy.model_management
    import comfy.utils

    session_uuid = session_id if session_id else str(uuid.uuid4())

    def extract(temp_dir):
        num_frames = video_frames.shape[0]
        log.info(f"Saving {num_frames} frames to {temp_dir}")
        pbar = comfy.utils.ProgressBar(num_frames)
        for i in range(num_frames):
            comfy.model_management.throw_exception_if_processing_interrupted()
            frame = video_frames[i].cpu().numpy()
            # Convert from [H, W, C] float32 0-1 to uint8 0-255
            frame = (frame * 255).astype(np.uint8)
            img = Image.fromarray(frame)
            img.save(os.path.join(temp_dir, f"{i:05d}.jpg"))
            pbar.update(1)
        log.info("Frames saved successfully")
        return num_frames, video_frames.shape[1], video_frames.shape[2]

    temp_dir = create_session_frames(session_uuid, extract)

    return SAM3VideoState(
        session_uuid=session_uuid,
        temp_dir=temp_dir,
        num_frames=video_frames.shape[0],
        height=video_frames.shape[1],
        width=video_frames.shape[2],
        config=config or VideoConfig(),
        prompts=(),
    )


def create_video_state_from_file(
    video_path: str,
    config: Optional[VideoConfig] = None,
    session_id: Optional[str] = None,
) -> SAM3VideoState:
    """
    Create a new video state from a video file.

    Memory-efficient: extracts frames one at a time using cv2,
    saving each as JPEG. Only 1 frame in RAM at a time.

    Args:
        video_path: Path to video file (mp4, mov, avi, mkv, webm)
        config: Optional tracking configuration
        session_id: Optional custom session ID (UUID generated if None)

    Returns:
        SAM3VideoState ready for use
    """
    import cv2
    from PIL import Image as PILImage
    import comfy.model_management
    import comfy.utils

    session_uuid = session_id if session_id else str(uuid.uuid4())

    def extract(temp_dir):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        try:
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            log.info(f"Extracting frames from {video_path} ({total_frames} frames, {width}x{height})")

            pbar = comfy.utils.ProgressBar(total_frames)
            frame_idx = 0
            while True:
                comfy.model_management.throw_exception_if_processing_interrupted()
                ret, frame = cap.read()
                if not ret:
                    break
                # BGR -> RGB, save as JPEG (only 1 frame in RAM at a time)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = PILImage.fromarray(frame_rgb)
                img.save(os.path.join(temp_dir, f"{frame_idx:05d}.jpg"))
                frame_idx += 1
                pbar.update(1)
        finally:
            cap.release()

        if frame_idx == 0:
            raise ValueError(f"No frames could be read from video: {video_path}")

        log.info(f"Extracted {frame_idx} frames to {temp_dir}")
        return frame_idx, height, width

    info = {}

    def extract_and_record(temp_dir):
        info["num_frames"], info["height"], info["width"] = extract(temp_dir)

    temp_dir = create_session_frames(session_uuid, extract_and_record)

    return SAM3VideoState(
        session_uuid=session_uuid,
        temp_dir=temp_dir,
        num_frames=info["num_frames"],
        height=info["height"],
        width=info["width"],
        config=config or VideoConfig(),
        prompts=(),
    )
