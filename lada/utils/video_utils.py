# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0
import csv
import dataclasses
import io
import json
import logging
import os
import re
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from fractions import Fraction
from functools import cache
from typing import Callable, Iterator, Tuple, Literal
from collections import deque
import heapq
import shlex

import av
import cv2
import torch
import numpy as np

from lada.utils import Image, Mask, VideoMetadata, os_utils

logger = logging.getLogger(__name__)

def read_video_frames(path: str, float32: bool = True, start_idx: int = 0, end_idx: int | None = None, normalize_neg1_pos1 = False, binary_frames=False) -> list[np.ndarray]:
    with VideoReaderOpenCV(path) as video_reader:
        frames = []
        i = 0
        while video_reader.isOpened():
            ret, frame = video_reader.read()
            if ret and (end_idx is None or i < end_idx):
                if binary_frames:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    frame = np.expand_dims(frame, axis=-1)
                if i >= start_idx:
                    if float32:
                        if normalize_neg1_pos1:
                            frame = (frame.astype(np.float32) / 255.0 - 0.5) / 0.5
                        else:
                            frame = frame.astype(np.float32) / 255.
                    frames.append(frame)
                i += 1
            else:
                break
    return frames

def resize_video_frames(frames: list, size: int | tuple[int, int]):
    resized = []
    target_size = size if isinstance(size, (list, tuple)) else (size, size)
    for frame in frames:
        if frame.shape[:2] == target_size:
            resized.append(frame)
        else:
            resized.append(cv2.resize(frame, (size, size), interpolation=cv2.INTER_LINEAR))
    return resized

def pad_to_compatible_size_for_video_codecs(imgs):
    # dims need to be divisible by 2 by most codecs. given the chroma / pix format dims must be divisible by 4
    h, w = imgs[0].shape[:2]
    pad_h = 0 if h % 4 == 0 else 4 - (h % 4)
    pad_w = 0 if w % 4 == 0 else 4 - (w % 4)
    if pad_h == 0 and pad_w == 0:
        return imgs
    else:
        return [np.pad(img, ((0, pad_h), (0, pad_w), (0,0))).astype(np.uint8) for img in imgs]

@contextmanager
def VideoReaderOpenCV(*args, **kwargs):
    cap = cv2.VideoCapture(*args, **kwargs)
    if not cap.isOpened():
        raise Exception(f"Unable to open video file:", *args)
    try:
        yield cap
    finally:
        cap.release()

class VideoReader:
    def __init__(self, file, nvidia_decode: bool = False):
        self.file = file
        self.nvidia_decode = nvidia_decode
        self.container = None

    def __enter__(self):
        # We currently do not pass through metadata to the output file so let's just ignore potential errors. Fixes #127
        # E.g. metadata could be encoded in CP936 instead of UTF-8 which would raise an error if we don't pass it in metadata_encoding.
        # If we use it in the future we have to consider non-default character encodings.
        hwaccel = None
        if self.nvidia_decode:
            hwaccel = av.codec.hwaccel.HWAccel('cuda', allow_software_fallback=True)
        self.container = av.open(self.file, metadata_errors='ignore', hwaccel=hwaccel)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.container.close()

    def frames(self) -> Iterator[Tuple[torch.Tensor, int]]:
        # Print to console via FFmpegs log callback instead of utilizing Pythons logging system
        # Unfortunately we need this to prevent deadlocks. On certain corrupt video files decode() would hang indefinitely after
        # encountering an error (always reproducible). See https://github.com/PyAV-Org/PyAV/issues/751 and https://codeberg.org/ladaapp/lada/issues/247
        # Alternatively, setting thread_type to 'SLICE' would also avoid the deadlock even with av logs enabled but may negatively impact performance.
        av.logging.restore_default_callback()
        av.logging.set_libav_level(av.logging.ERROR)
        self.container.streams.video[0].thread_type = 'SLICE' if self.nvidia_decode else 'AUTO'

        # Fault-tolerant frame decoding with frame duplication for corrupted frames
        # This approach mimics how ffmpeg CLI handles corrupted frames by duplicating the last good frame
        last_good_frame = None
        consecutive_errors = 0
        max_consecutive_errors = 10  # Prevent infinite loops on completely corrupted streams
        
        # Use packet-level decoding to handle corrupted frames properly
        vstream = self.container.streams.video[0]
        for packet in self.container.demux(vstream):
            try:
                frames = packet.decode()
                for frame in frames:
                    nd_frame = frame.to_ndarray(format='bgr24')
                    torch_frame = torch.from_numpy(nd_frame)
                    last_good_frame = (torch_frame, frame.pts)
                    consecutive_errors = 0
                    yield torch_frame, frame.pts
            except av.error.InvalidDataError as e:
                # Handle corrupted frames by duplicating the last good frame
                if last_good_frame is not None and consecutive_errors < max_consecutive_errors:
                    consecutive_errors += 1
                    logger.warning(f"Corrupted frame detected, duplicating last good frame ({consecutive_errors}/{max_consecutive_errors})")
                    yield last_good_frame[0], last_good_frame[1]
                else:
                    # No good frame available yet (first frame corrupt) or too many consecutive errors
                    # Re-raise the error instead of skipping to fail fast
                    raise Exception(f"Cannot handle corrupted frame: {'first frame corrupt' if last_good_frame is None else f'too many consecutive corrupted frames ({max_consecutive_errors})'}") from e
            except Exception as e:
                # For other unexpected errors, re-raise them
                raise

    def seek(self, offset_ns):
        offset = int((offset_ns / 1_000_000_000) * av.time_base)
        self.container.seek(offset)

def get_video_meta_data(path: str) -> VideoMetadata:
    cmd = ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-select_streams', 'v', '-show_streams', '-show_format', path]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, startupinfo=os_utils.get_subprocess_startup_info())
    out, err =  p.communicate()
    if p.returncode != 0:
        raise Exception(f"error running ffprobe: {err.strip()}. Code: {p.returncode}, cmd: {cmd}")
    json_output = json.loads(out)
    json_video_stream = json_output["streams"][0]
    json_video_format = json_output["format"]

    value = [int(num) for num in json_video_stream['avg_frame_rate'].split("/")]
    # Can be 0/0 for some files for ffprobe isn't able to determine the number of frames nb_frames
    average_fps = value[0]/value[1] if len(value) == 2 and value[1] != 0 else value[0]

    value = [int(num) for num in json_video_stream['r_frame_rate'].split("/")]
    fps = value[0]/value[1] if len(value) == 2 else value[0]
    fps_exact = Fraction(value[0], value[1])

    value = [int(num) for num in json_video_stream['time_base'].split("/")]
    time_base = Fraction(value[0], value[1])

    frame_count = json_video_stream.get('nb_frames')
    if not frame_count:
        # print("frame count ffmpeg", frame_count)
        cap = cv2.VideoCapture(path)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()
        # print("frame count opencv", frame_count)
    frame_count=int(frame_count)

    start_pts = json_video_stream.get('start_pts')

    metadata = VideoMetadata(
        video_file=path,
        video_height=int(json_video_stream['height']),
        video_width=int(json_video_stream['width']),
        video_fps=fps,
        average_fps=average_fps,
        video_fps_exact=fps_exact,
        codec_name=json_video_stream['codec_name'],
        frames_count=frame_count,
        duration=float(json_video_stream.get('duration', json_video_format['duration'])),
        time_base=time_base,
        start_pts=start_pts
    )
    return metadata

def offset_ns_to_frame_num(offset_ns, video_fps_exact):
    return int(Fraction(offset_ns, 1_000_000_000) * video_fps_exact)

def write_frames_to_video_file(frames: list[Image], output_path, fps: int | float | Fraction, codec='x264', preset='medium', crf=None):
    assert frames[0].ndim == 3
    width = frames[0].shape[1]
    height = frames[0].shape[0]
    ffmpeg_output = [
        'nice', '-n', '19', 'ffmpeg', '-y',
        '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{width}x{height}', '-r', f"{fps.numerator}/{fps.denominator}" if type(fps) == Fraction else str(fps),
        '-i', '-', '-an', '-preset', preset
    ]
    if codec == 'x265':
        ffmpeg_output.extend(['-tag:v', 'hvc1', '-vcodec', 'libx265', '-crf', str(crf) if crf else '18'])
    elif codec == 'x264':
        ffmpeg_output.extend(['-vcodec', 'libx264', '-crf', str(crf) if crf else '15'])
    ffmpeg_output.append(output_path)

    ffmpeg_process = subprocess.Popen(ffmpeg_output, stdin=subprocess.PIPE, stderr=subprocess.PIPE, stdout=subprocess.PIPE, startupinfo=os_utils.get_subprocess_startup_info())
    for frame in frames:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ffmpeg_process.stdin.write(frame.tobytes())
    ffmpeg_process.stdin.close()
    ffmpeg_process.wait()
    if ffmpeg_process.returncode != 0:
        print(f"ERROR when writing video via ffmpeg to file: {output_path}, return code: {ffmpeg_process.returncode}")
        print(f"stderr: {ffmpeg_process.stderr.read()}")

def write_masks_to_video_file(frames: list[Mask], output_path, fps: int | float | Fraction):
    #assert frames[0].ndim == 2
    width = frames[0].shape[1]
    height = frames[0].shape[0]
    ffmpeg_output = [
        'nice', '-n', '19', 'ffmpeg', '-y',
        '-f', 'rawvideo', '-pix_fmt', 'gray', '-s', f'{width}x{height}', '-r', f"{fps.numerator}/{fps.denominator}" if type(fps) == Fraction else str(fps),
        '-i', '-', '-an', '-vcodec', 'ffv1', '-level', '3', '-tag:v', 'ffv1',  output_path
    ]

    ffmpeg_process = subprocess.Popen(ffmpeg_output, stdin=subprocess.PIPE, stderr=subprocess.PIPE, startupinfo=os_utils.get_subprocess_startup_info())
    for frame in frames:
        try:
            ffmpeg_process.stdin.write(frame.tobytes())
        except Exception as e:
            print(f"ERROR when writing video via ffmpeg to file: {output_path}")
            print(f"exception: {e}")
            print(f"stderr: {ffmpeg_process.stderr.read()}")
            print(f"stdout: {ffmpeg_process.stdout.read()}")
            raise e
    ffmpeg_process.stdin.close()
    ffmpeg_process.wait()
    if ffmpeg_process.returncode != 0:
        print(f"ERROR when writing video via ffmpeg to file: {output_path}, return code: {ffmpeg_process.returncode}")
        print(f"stderr: {ffmpeg_process.stderr.read()}")
        print(f"stdout: {ffmpeg_process.stdout.read()}")

def process_video_v3(input_path, output_path, frame_processor: Callable[[Image], Image]):
    video_metadata = get_video_meta_data(input_path)
    video_reader = cv2.VideoCapture(input_path)
    video_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps=video_metadata.video_fps, frameSize=(video_metadata.video_width, video_metadata.video_height))
    while video_reader.isOpened():
        ret, frame = video_reader.read()
        if ret:
            processed_frame = frame_processor(frame)
            video_writer.write(processed_frame)
        else:
            break
    video_reader.release()
    video_writer.release()

def approx_memory(video_metadata: VideoMetadata, frames_count, assume_images=True, assume_masks=True):
    size = 0
    frame_size_image = video_metadata.video_width * video_metadata.video_height * 3 * 1
    frame_size_mask = video_metadata.video_width * video_metadata.video_height * 1 * 1
    if assume_images:
        size += frame_size_image * frames_count
    if assume_masks:
        size += frame_size_mask * frames_count
    return size

def approx_max_length_by_memory_limit(video_metadata: VideoMetadata, limit_in_megabytes, assume_images=True, assume_masks=True):
    frame_size_image = approx_memory(video_metadata, 1, assume_images=assume_images, assume_masks=assume_masks)
    max_length_frames = (limit_in_megabytes * 1024 * 1024) / frame_size_image
    max_length_seconds = int(max_length_frames / video_metadata.video_fps)
    return max_length_seconds

@dataclass
class EncodingPreset:
    name: str
    description: str
    user_preset: bool
    encoder_name: str
    encoder_options: str
    video_filter: str = ""

    def __hash__(self): return hash(self.name)

    def clone(self): return EncodingPreset(**dataclasses.asdict(self))

def get_default_preset_name():
    if os_utils.has_nvidia_gpu() and is_nvidia_cuda_encoding_available():
        return "hevc-nvidia-gpu-hq"
    if is_apple_videotoolbox_encoding_available():
        return "hevc-apple-gpu-balanced"
    if os_utils.has_intel_arc_gpu() and is_intel_qsv_encoding_available():
        return "hevc-intel-gpu-hq"
    return "h264-cpu-fast"

@cache
def get_encoding_presets() -> list[EncodingPreset]:
    presets = []
    encoding_presets_csv_path = os.path.join(os.path.dirname(__file__), 'encoding_presets.csv')
    if not os.path.exists(encoding_presets_csv_path):
        logger.warning("Could not find encoding_presets.csv!")
        return presets
    
    available_encoders_list = get_video_encoder_codecs()
    available_encoder_names = {e.name.lower() for e in available_encoders_list}
    has_intel_qsv = False
    if 'h264_qsv' in available_encoder_names:
        has_intel_qsv = is_intel_qsv_encoding_available()
    has_nvidia_nvenc = False
    if 'h264_nvenc' in available_encoder_names:
        has_nvidia_nvenc = is_nvidia_cuda_encoding_available()
    has_apple_vt = False
    if 'hevc_videotoolbox' in available_encoder_names or 'h264_videotoolbox' in available_encoder_names:
        has_apple_vt = is_apple_videotoolbox_encoding_available()

    with open(encoding_presets_csv_path, mode='r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile, delimiter='|')
        for row in reader:
            encoder_name = row["encoder_name"].lower()
            preset_name = row["preset_name"].lower()
            
            if encoder_name not in available_encoder_names:
                continue

            # Nvidia
            is_nvidia_preset = 'nvenc' in encoder_name or 'nvidia' in preset_name
            if is_nvidia_preset and not has_nvidia_nvenc:
                continue
            # Intel
            is_intel_preset = 'qsv' in encoder_name or 'intel' in preset_name
            if is_intel_preset and not has_intel_qsv:
                continue
            # Apple Video Toolbox
            is_apple_preset = 'videotoolbox' in encoder_name or 'apple' in preset_name
            if is_apple_preset and not has_apple_vt:
                continue

            video_filter = row.get("video_filter", "")
            preset = EncodingPreset(row["preset_name"], row["preset_description(translatable)"], False, row["encoder_name"], row["encoder_options"], video_filter)
            presets.append(preset)
        return presets

@cache
def is_intel_qsv_encoding_available() -> bool:
    if sys.platform == "win32":
        return _is_codec_hardware_acceleration_working('h264_qsv', 'qsv')
    else:
        # TODO: For some reason the method HWAccel.create() is not working for qsv when using official Linux binary wheel of PyAv 16.1.0
        # It throws a "Function not implemented" error regardless whether qsv is working or not (cuda/nvenc check works as expected)
        # It works when building PyAV locally against ffmpeg from ArchLinux on the same system
        # As a workaround let's encode a dummy frame to see if qsv is working
        # See issue: #297
        try:
            with av.logging.Capture():
                mem_file = io.BytesIO()
                with av.open(mem_file, mode='w', format='mp4') as container:
                    stream = container.add_stream('h264_qsv', rate=30)
                    stream.width = 64
                    stream.height = 64
                    stream.pix_fmt = 'nv12'
                    dummy_frame = av.VideoFrame(64, 64, format='nv12')
                    stream.encode(dummy_frame)
                    return True
        except Exception:
            return False

@cache
def is_nvidia_cuda_encoding_available() -> bool:
    return _is_codec_hardware_acceleration_working('h264_nvenc', 'cuda')

@cache
def is_nvidia_cuda_decoding_available() -> bool:
    return _is_codec_hardware_acceleration_working('h264_cuvid', 'cuda', 'r')

@cache
def is_apple_videotoolbox_encoding_available() -> bool:
    if sys.platform != "darwin":
        return False
    if _is_codec_hardware_acceleration_working('hevc_videotoolbox', 'videotoolbox'):
        return True
    # HWAccel check can fail in frozen/PyInstaller builds. Try encoding a dummy frame instead (same approach as QSV on Linux).
    try:
        with av.logging.Capture():
            mem_file = io.BytesIO()
            with av.open(mem_file, mode='w', format='mp4') as container:
                stream = container.add_stream('hevc_videotoolbox', rate=30)
                stream.width = 64
                stream.height = 64
                stream.pix_fmt = 'yuv420p'
                dummy_frame = av.VideoFrame(64, 64, format='yuv420p')
                stream.encode(dummy_frame)
                return True
    except Exception:
        return False

def _is_codec_hardware_acceleration_working(codec_name: str, hwaccel_device_type: str, codec_mode: Literal["r", "w"]='w') -> bool:
    try:
        with av.logging.Capture():
            hwaccel = av.codec.hwaccel.HWAccel(hwaccel_device_type, allow_software_fallback=False)
            codec = av.codec.Codec(codec_name, codec_mode)
            # Initialize hardware context. This will raise if hardware is not available (missing, driver or library issue)
            hwaccel.create(codec)
            return True
    except Exception:
        return False

class VideoWriter:
    def _parse_encoder_options(self, encoder_options: str):
        tokens = shlex.split(encoder_options)
        parsed_encoder_options = {
            tokens[i].lstrip("-"): tokens[i + 1]
            for i in range(0, len(tokens), 2)
        }
        return parsed_encoder_options

    @staticmethod
    def _parse_video_filter(video_filter: str) -> list[tuple[str, dict[str, str]]]:
        """Parse a video filter string into a list of (filter_name, kwargs) tuples.

        Supports ffmpeg libavfilter syntax:
        - Single filter with named params: ``fps=30`` → ``("fps", {"fps": "30"})``
        - Multiple named params: ``fps=30:round=near`` → ``("fps", {"fps": "30", "round": "near"})``
        - Filter chain: ``fps=30,eq=brightness=0.1`` → two linked filter nodes
        - Filter with no params: ``yadif`` → ``("yadif", {})``
        """
        if not video_filter or not video_filter.strip():
            return []

        filters = []
        for filter_str in video_filter.split(','):
            filter_str = filter_str.strip()
            if not filter_str:
                continue

            eq_pos = filter_str.find('=')
            if eq_pos == -1:
                # No params: filter name only (e.g. "yadif")
                filters.append((filter_str, {}))
            else:
                filter_name = filter_str[:eq_pos]
                args_str = filter_str[eq_pos + 1:]
                kwargs = {}
                # Split args on ':' but respect quoted values
                # Simple split — advanced filter graphs with nested expressions
                # would need the full ffmpeg filter parser, but common use cases
                # (fps, eq, scale, etc.) work with this.
                parts = args_str.split(':')
                for part in parts:
                    part = part.strip()
                    if not part:
                        continue
                    kv_eq = part.find('=')
                    if kv_eq != -1:
                        kwargs[part[:kv_eq]] = part[kv_eq + 1:]
                    else:
                        # Positional arg — use as value with key = filter_name
                        # e.g. "fps=30" → kwargs["fps"] = "30"
                        kwargs[filter_name] = part
                filters.append((filter_name, kwargs))
        return filters

    def _create_filter_graph(self, video_filter: str, width: int, height: int, time_base):
        """Create a PyAV filter graph from a video filter string."""
        graph = av.filter.Graph()

        # Buffer source — must match the frame format we push (rgb24 from ndarray)
        buffer_src = graph.add_buffer(width=width, height=height, format='rgb24', time_base=time_base)

        # Parse and add filter nodes
        parsed_filters = self._parse_video_filter(video_filter)
        nodes = [buffer_src]
        for filter_name, kwargs in parsed_filters:
            node = graph.add(filter_name, **kwargs)
            nodes.append(node)

        # Buffer sink
        buffersink = graph.add("buffersink")
        nodes.append(buffersink)

        # Link all nodes in sequence
        for i in range(len(nodes) - 1):
            nodes[i].link_to(nodes[i + 1])

        graph.configure()
        return graph

    def __init__(self, output_path, width, height, fps, encoder: str, encoder_options: str, time_base=None, mp4_fast_start=False, video_filter: str = ""):
        container_options = {}
        if mp4_fast_start and (output_path.lower().endswith(".mp4") or output_path.lower().endswith(".mov")):
            container_options["movflags"] = "+frag_keyframe+empty_moov+faststart"

        output_container = av.open(output_path, "w", options=container_options)
        video_stream_out: av.VideoStream = output_container.add_stream(encoder, fps)

        self.is_qsv_encoder = 'qsv' in encoder.lower()

        if encoder == "libsvtav1" and "SVT_LOG" not in os.environ:
            # Suppress logging default info messages
            os.environ["SVT_LOG"] = "1"

        target_pix_fmt = 'yuv420p'
        if self.is_qsv_encoder:
            target_pix_fmt = 'nv12'

        video_stream_out.pix_fmt = target_pix_fmt
        video_stream_out.codec_context.pix_fmt = target_pix_fmt

        video_stream_out.width = width
        video_stream_out.height = height
        video_stream_out.thread_count = 0
        video_stream_out.thread_type = 3
        video_stream_out.time_base = time_base

        # up until PyAV 15.5.0 it was enough to set these settings on the stream only.
        video_stream_out.codec_context.width = width
        video_stream_out.codec_context.height = height
        video_stream_out.codec_context.thread_count = 0
        video_stream_out.codec_context.thread_type = 3
        video_stream_out.codec_context.time_base = time_base

        stream_options = self._parse_encoder_options(encoder_options)
        # hevc_videotoolbox needs tag 'hvc1' for compatibility (e.g. Safari in MP4/MOV); preset -tag:v is often ignored by PyAV/ffmpeg
        if encoder == 'hevc_videotoolbox':
            stream_options['tag'] = 'hvc1'
        video_stream_out.options = stream_options
        self.output_container = output_container
        self.video_stream = video_stream_out

        # Preserve the original time_base before the encoder opens and potentially
        # overrides video_stream.time_base with its own internal value (e.g. 1/16000
        # instead of 1/1000). Frames must use the original time_base so that their
        # PTS values are interpreted correctly by the encoder.
        self._time_base = time_base

        # Video filter graph (None when no filter is configured)
        self.filter_graph = None
        if video_filter and video_filter.strip():
            self.filter_graph = self._create_filter_graph(video_filter.strip(), width, height, time_base)

        # Buffers for reordering frames
        self.BUFFER_MAX_SIZE = 30
        self.pts_heap = []
        self.frame_queue = deque()
        self.pts_set = set()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.release()

    def _drain_filter_graph(self):
        """Pull all available frames from the filter graph and encode them.

        Called after vpush() to consume any output the filter has produced.
        BlockingIOError means the filter is still buffering — this is normal
        and not an error condition, so we catch it and return.
        """
        while True:
            try:
                filtered_frame = self.filter_graph.vpull()
                out_packet = self.video_stream.encode(filtered_frame)
                if out_packet:
                    self.output_container.mux(out_packet)
            except av.error.BlockingIOError:
                # Filter is still buffering — no output available yet.
                # This is the normal "not ready" path, not an error.
                break
            except (av.error.EOFError, EOFError):
                break

    def _process_buffer(self, flush_all=False):
        """Processes the buffer to encode frames."""
        if len(self.frame_queue) > (self.BUFFER_MAX_SIZE / 2) or (flush_all and self.frame_queue):
            frame_to_encode = self.frame_queue.popleft()
            pts_to_assign = heapq.heappop(self.pts_heap)
            self.pts_set.remove(pts_to_assign)

            out_frame = av.VideoFrame.from_ndarray(frame_to_encode, format='rgb24')
            out_frame.pts = pts_to_assign
            out_frame.time_base = self._time_base

            if self.filter_graph is not None:
                self.filter_graph.vpush(out_frame)
                self._drain_filter_graph()
            else:
                out_packet = self.video_stream.encode(out_frame)
                if out_packet:
                    self.output_container.mux(out_packet)


    def write(self, frame, frame_pts=None, bgr2rgb=False):
        # We add the frame and its pts given by PyAV (FFmpeg) to a FIFO queue and a min heap, respectively.
        # Upon a call to write(), if the buffer is full, we pop the head of the queue and the smallest PTS and pair
        # those together. This operation is a no-op for "nicely behaved" videos, where frames and PTS are decoded
        # in linear order. However, it appears several problematic videos exist such that the frames are given in
        # linear order, but the PTS associated with the frames are not. This strategy is used to avoid prompting
        # the user to identify a framerate ahead of time, and uses the timing of the existing PTS, but reorders the PTS.
        #
        # See https://codeberg.org/ladaapp/lada/pulls/33 for more information/discussion.
        if isinstance(frame, torch.Tensor):
            frame = frame.cpu().numpy()
        if bgr2rgb:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if frame_pts not in self.pts_set:
            heapq.heappush(self.pts_heap, frame_pts)
            self.frame_queue.append(frame)
            self.pts_set.add(frame_pts)

        self._process_buffer()

    def release(self):
        while len(self.frame_queue) > 0:
            self._process_buffer(flush_all=True)
        # Flush the filter graph
        if self.filter_graph is not None:
            try:
                self.filter_graph.vpush(None)
                # After EOF, pull all remaining frames from the filter
                while True:
                    try:
                        filtered_frame = self.filter_graph.vpull()
                        out_packet = self.video_stream.encode(filtered_frame)
                        if out_packet:
                            self.output_container.mux(out_packet)
                    except (av.error.EOFError, EOFError):
                        break
                    except av.error.BlockingIOError:
                        # Should not happen after EOF, but handle defensively
                        break
            except Exception:
                logger.warning("Error on flushing filter graph. Ignoring...")
        # Flush the encoder
        try:
            out_packet = self.video_stream.encode(None)
            if out_packet:
                self.output_container.mux(out_packet)
        except:
            # TODO: For half of my test files flushing QSV encoders fail here with "Application provided invalid, non monotonically increasing dts to muxer in stream"
            # This doesn't happen with libx264 or NVENC encoders. The restored file plays fine so let's ignore it for now.
            if self.is_qsv_encoder:
                logger.warning("Error on flushing QSV encoder. Ignoring...")
            else:
                raise
        self.output_container.close()

def is_video_file(file_path):
    SUPPORTED_VIDEO_FILE_EXTENSIONS = {".asf", ".avi", ".m4v", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg", ".ts", ".wmv",
                                       ".webm"}

    file_ext = os.path.splitext(file_path)[1]
    return file_ext.lower() in SUPPORTED_VIDEO_FILE_EXTENSIONS

@dataclass
class Encoder:
    name: str
    long_name: str
    hardware_encoder: bool
    hardware_devices: set[str]

    def __hash__(self): return hash(self.name)

def get_human_readable_hardware_device_name(device_type_name: str) -> str:
    if device_type_name == 'qsv':
        return 'Intel QSV'
    elif device_type_name == 'amf':
        return 'AMD AMF'
    elif device_type_name == 'cuda':
        return 'Nvidia CUDA'
    elif device_type_name == 'videotoolbox':
        return 'Apple VideoToolbox'
    return device_type_name

def get_video_encoder_codecs() -> list[Encoder]:
    codecs = set()
    for name in av.codec.codecs_available:
        try:
            codec = av.codec.Codec(name, "w")
        except ValueError:
            continue
        if codec.type != 'video':
            continue
        if re.search(r'\bimage\b', codec.long_name, re.IGNORECASE):
            continue
        codec_long_name = codec.long_name.lower()
        whitelist_video_codecs = ['hevc', 'h265', "h.265", "h264", "h.264", "vp9", "av1", "ffmpeg video codec #1", "huffyuv", "prores", "mpeg-2"]
        whitelist_hardware_devices = ['qsv', 'cuda', 'amf', 'videotoolbox']
        if not any(name in codec_long_name for name in whitelist_video_codecs):
            continue
        is_hardware_encoder = codec.hardware_configs is not None and len(codec.hardware_configs) > 0
        hardware_devices = set([hwconfig.device_type.name for hwconfig in filter(lambda hwconfig: hwconfig.device_type.name in whitelist_hardware_devices, codec.hardware_configs)] if is_hardware_encoder else [] if is_hardware_encoder else [])
        encoder = Encoder(codec.name, codec.long_name, is_hardware_encoder, hardware_devices)
        codecs.add(encoder)
    return sorted(list(codecs), key=lambda e: e.name)

class VideoThumbnailer:

    def __init__(self, video_path: str, thumb_width: int, thumb_height: int):
        self.video_path = video_path
        self.cap = None
        self.thumb_width = thumb_width
        self.thumb_height = thumb_height
        self._frame_cache = {} # LRU cache for recently accessed frames to avoid re-seeking for nearby timestamps
        self._cache_max_size = 60
        self._cache_access_order = []  # Track access order for LRU

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def open(self):
        if self.cap is None:
            self.cap = cv2.VideoCapture(self.video_path)
            if not self.cap.isOpened():
                raise Exception(f"Unable to open video file: {self.video_path}")

    def close(self):
        if self.cap:
            self.cap.release()
            self.cap = None
        self._frame_cache.clear()
        self._cache_access_order.clear()

    def _get_fallback_thumbnail(self):
        return np.zeros(shape=(self.thumb_height, self.thumb_width, 3), dtype=np.uint8)

    def _get_cached_thumbnail(self, timestamp_ms: float) -> np.ndarray | None:
        """Get frame from cache if available and recent enough"""
        # Round to nearest 100ms for caching (avoids too many cache entries)
        cache_key = round(timestamp_ms / 100) * 100

        if cache_key in self._frame_cache:
            # Move to end of access order (most recently used)
            if cache_key in self._cache_access_order:
                self._cache_access_order.remove(cache_key)
            self._cache_access_order.append(cache_key)
            return self._frame_cache[cache_key].copy()
        return None

    def _cache_thumbnail(self, timestamp_ms: float, frame: np.ndarray):
        """Add frame to cache with LRU eviction"""
        cache_key = round(timestamp_ms / 100) * 100

        # Remove from access order if already exists
        if cache_key in self._cache_access_order:
            self._cache_access_order.remove(cache_key)

        # Add to cache
        self._frame_cache[cache_key] = frame.copy()
        self._cache_access_order.append(cache_key)

        # Evict least recently used if cache is full
        if len(self._frame_cache) > self._cache_max_size:
            oldest_key = self._cache_access_order.pop(0)
            del self._frame_cache[oldest_key]

    def get_thumbnail(self, timestamp_ns: int) -> np.ndarray:
        try:
            # Convert nanoseconds to milliseconds for OpenCV
            timestamp_ms = timestamp_ns / 1_000_000

            cached_thumbnail = self._get_cached_thumbnail(timestamp_ms)
            if cached_thumbnail is not None:
                return cached_thumbnail

            self.cap.set(cv2.CAP_PROP_POS_MSEC, timestamp_ms)

            ret, frame = self.cap.read()

            if ret and frame is not None:
                thumbnail = cv2.resize(frame, (self.thumb_width, self.thumb_height), interpolation=cv2.INTER_LINEAR)
                self._cache_thumbnail(timestamp_ms, thumbnail)

                return thumbnail

            return self._get_fallback_thumbnail()

        except Exception as e:
            logger.error(f"Error generating thumbnail at {timestamp_ns}: {e}")
            return self._get_fallback_thumbnail()
