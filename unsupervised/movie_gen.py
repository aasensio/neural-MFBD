import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterator, Literal, Optional

import cv2
import ffmpeg
import matplotlib.animation as ani
import matplotlib.pyplot as plt
import numpy as np


def as_uint8(arr: np.ndarray) -> np.ndarray:
    if np.issubdtype(arr.dtype, np.integer):
        return arr.astype(np.uint8)
    if np.issubdtype(arr.dtype, np.floating):
        return (arr * 255).round().astype(np.uint8)
    raise NotImplementedError(f"Unsupported dtype: {arr.dtype}")


@dataclass
class VideoProps:
    frame_width: int
    frame_height: int
    frame_count: int
    fps: int

    @classmethod
    def from_file_opencv(cls, fpath: str | Path) -> "VideoProps":
        cap = cv2.VideoCapture(str(fpath))

        return cls(
            frame_width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            frame_height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            fps=int(cap.get(cv2.CAP_PROP_FPS)),
        )

    @classmethod
    def from_file_ffmpeg(cls, fpath: str | Path) -> "VideoProps":
        probe = ffmpeg.probe(str(fpath))

        for stream in probe["streams"]:
            if stream["codec_type"] == "video":
                width, height, count = stream["width"], stream["height"], int(stream["nb_frames"])
                fps_num, fps_denom = stream["r_frame_rate"].split("/")
                assert fps_denom == "1", f"Unexpected frame rate: {stream['r_frame_rate']}"
                fps = int(fps_num)
                return cls(
                    frame_width=width,
                    frame_height=height,
                    frame_count=count,
                    fps=fps,
                )

        raise ValueError(f"Could not parse video properties from video in {fpath}")


def read_video_ffmpeg(
    in_file: str | Path,
    output_fmt: str = "rgb24",
    channels: int = 3,
) -> Iterator[np.ndarray]:
    """Function that reads a video to a stream of numpy arrays using FFMPEG.

    Args:
        in_file: The input video to read
        output_fmt: The output image format
        channels: Number of output channels for each video frame

    Yields:
        Frames from the video as numpy arrays with shape (H, W, C)
    """

    props = VideoProps.from_file_ffmpeg(in_file)

    stream = ffmpeg.input(str(in_file))
    stream = ffmpeg.output(stream, "pipe:", format="rawvideo", pix_fmt=output_fmt, r=props.fps)
    stream = ffmpeg.run_async(stream, pipe_stdout=True)

    while True:
        in_bytes = stream.stdout.read(props.frame_width * props.frame_height * channels)
        if not in_bytes:
            break
        yield np.frombuffer(in_bytes, np.uint8).reshape((props.frame_height, props.frame_width, channels))

    stream.stdout.close()
    stream.wait()


def read_video_opencv(in_file: str | Path) -> Iterator[np.ndarray]:
    """Reads a video as a stream using OpenCV.

    Args:
        in_file: The input video to read

    Yields:
        Frames from the video as numpy arrays with shape (H, W, C)
    """

    cap = cv2.VideoCapture(str(in_file))

    while True:
        ret, buffer = cap.read()
        if not ret:
            cap.release()
            return
        yield buffer


def write_video_opencv(
    itr: Iterator[np.ndarray],
    out_file: str | Path,
    fps: int = 30,
    codec: str = "MP4V",
) -> None:
    """Function that writes a video from a stream of numpy arrays using OpenCV.

    Args:
        itr: The image iterator, yielding images with shape (H, W, C).
        out_file: The path to the output file.
        fps: Frames per second for the video.
        codec: FourCC code specifying OpenCV video codec type. Examples are
            MPEG, MP4V, DIVX, AVC1, H236.
    """

    first_img = next(itr)
    height, width, _ = first_img.shape

    fourcc = cv2.VideoWriter_fourcc(*codec)
    stream = cv2.VideoWriter(str(out_file), fourcc, fps, (width, height))

    def write_frame(img: np.ndarray) -> None:
        stream.write(as_uint8(img))

    write_frame(first_img)
    for img in itr:
        write_frame(img)

    stream.release()
    cv2.destroyAllWindows()


def write_video_ffmpeg(
    itr: Iterator[np.ndarray],
    out_file: str | Path,
    fps: int = 30,
    out_fps: int = 30,
    vcodec: str = "libx264",
    input_fmt: str = "rgb24",
    output_fmt: str = "yuv420p",
) -> None:
    """Function that writes an video from a stream of numpy arrays using FFMPEG.

    Args:
        itr: The image iterator, yielding images with shape (H, W, C).
        out_file: The path to the output file.
        fps: Frames per second for the video.
        out_fps: Frames per second for the saved video.
        vcodec: The video codec to use for the output video
        input_fmt: The input image format
        output_fmt: The output image format
    """

    first_img = next(itr)
    height, width, _ = first_img.shape

    stream = ffmpeg.input("pipe:", format="rawvideo", pix_fmt=input_fmt, s=f"{width}x{height}", r=fps)
    stream = ffmpeg.output(stream, str(out_file), pix_fmt=output_fmt, vcodec=vcodec, r=out_fps)
    stream = ffmpeg.overwrite_output(stream)
    stream = ffmpeg.run_async(stream, pipe_stdin=True)

    def write_frame(img: np.ndarray) -> None:
        stream.stdin.write(as_uint8(img).tobytes())

    # Writes all the video frames to the file.
    write_frame(first_img)
    for img in itr:
        write_frame(img)

    stream.stdin.close()
    stream.wait()


def write_video_matplotlib(
    itr: Iterator[np.ndarray],
    out_file: str | Path,
    dpi: int = 50,
    fps: int = 30,
    title: str = "Video",
    comment: Optional[str] = None,
    writer: str = "ffmpeg",
) -> None:
    """Function that writes an video from a stream of input tensors.

    Args:
        itr: The image iterator, yielding images with shape (H, W, C).
        out_file: The path to the output file.
        dpi: Dots per inch for output image.
        fps: Frames per second for the video.
        title: Title for the video metadata.
        comment: Comment for the video metadata.
        writer: The Matplotlib video writer to use (if you use the
            default one, make sure you have `ffmpeg` installed on your
            system).
    """

    first_img = next(itr)
    height, width, _ = first_img.shape
    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi))

    # Ensures that there's no extra space around the image.
    fig.subplots_adjust(
        left=0,
        bottom=0,
        right=1,
        top=1,
        wspace=None,
        hspace=None,
    )

    # Creates the writer with the given metadata.
    writer_obj = ani.writers[writer]
    metadata = {
        "title": title,
        "artist": __name__,
        "comment": comment,
    }
    mpl_writer = writer_obj(
        fps=fps,
        metadata={k: v for k, v in metadata.items() if v is not None},
    )

    with mpl_writer.saving(fig, out_file, dpi=dpi):
        im = ax.imshow(as_uint8(first_img), interpolation="nearest")
        mpl_writer.grab_frame()

        for img in itr:
            im.set_data(as_uint8(img))
            mpl_writer.grab_frame()


Reader = Literal["ffmpeg", "opencv"]
Writer = Literal["ffmpeg", "matplotlib", "opencv"]

READERS: Dict[Reader, Callable[[str | Path], Iterator[np.ndarray]]] = {
    "ffmpeg": read_video_ffmpeg,
    "opencv": read_video_opencv,
}

WRITERS: Dict[Writer, Callable[[Iterator[np.ndarray], str | Path], None]] = {
    "ffmpeg": write_video_ffmpeg,
    "matplotlib": write_video_matplotlib,
    "opencv": write_video_opencv,
}

# Remove the FFMPEG reader and writer if FFMPEG is not available in the system.
if not shutil.which("ffmpeg"):
    READERS.pop("ffmpeg")
    WRITERS.pop("ffmpeg")
    WRITERS.pop("matplotlib")