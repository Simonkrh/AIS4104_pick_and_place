#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import cv2


DICTIONARIES = {
    "4x4_50": cv2.aruco.DICT_4X4_50,
    "4x4_100": cv2.aruco.DICT_4X4_100,
    "5x5_50": cv2.aruco.DICT_5X5_50,
    "5x5_100": cv2.aruco.DICT_5X5_100,
    "6x6_50": cv2.aruco.DICT_6X6_50,
    "6x6_100": cv2.aruco.DICT_6X6_100,
}

DEFAULT_SQUARES_X = 7
DEFAULT_SQUARES_Y = 5
DEFAULT_SQUARE_LENGTH_MM = 32.5
DEFAULT_MARKER_LENGTH_MM = 25.0
DEFAULT_MARGIN_MM = 8.0
DEFAULT_DPI = 300
DEFAULT_DICTIONARY = "5x5_100"
DEFAULT_OUTPUT = "calibration/charuco_board_a4_7x5_32p5mm.png"


def mm_to_px(mm: float, dpi: int) -> int:
    return max(1, round(mm / 25.4 * dpi))


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate the printable ChArUco board PNG. Defaults are tuned for "
            "the current A4 7x5 board used in this project."
        )
    )
    parser.add_argument(
        "--squares-x",
        type=int,
        default=DEFAULT_SQUARES_X,
        help="Number of squares along X.",
    )
    parser.add_argument(
        "--squares-y",
        type=int,
        default=DEFAULT_SQUARES_Y,
        help="Number of squares along Y.",
    )
    parser.add_argument(
        "--square-length-mm",
        type=float,
        default=DEFAULT_SQUARE_LENGTH_MM,
        help="Square side length in millimeters.",
    )
    parser.add_argument(
        "--marker-length-mm",
        type=float,
        default=DEFAULT_MARKER_LENGTH_MM,
        help="ArUco marker side length in millimeters.",
    )
    parser.add_argument(
        "--margin-mm",
        type=float,
        default=DEFAULT_MARGIN_MM,
        help="White margin around the board in millimeters.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=DEFAULT_DPI,
        help="Raster DPI for printing.",
    )
    parser.add_argument(
        "--dictionary",
        choices=sorted(DICTIONARIES.keys()),
        default=DEFAULT_DICTIONARY,
        help="ArUco dictionary.",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="Output PNG path.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.marker_length_mm >= args.square_length_mm:
        raise SystemExit("--marker-length-mm must be smaller than --square-length-mm")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    dictionary = cv2.aruco.getPredefinedDictionary(DICTIONARIES[args.dictionary])
    board = cv2.aruco.CharucoBoard(
        (args.squares_x, args.squares_y),
        args.square_length_mm / 1000.0,
        args.marker_length_mm / 1000.0,
        dictionary,
    )

    board_width_mm = args.squares_x * args.square_length_mm
    board_height_mm = args.squares_y * args.square_length_mm
    total_width_mm = board_width_mm + 2.0 * args.margin_mm
    total_height_mm = board_height_mm + 2.0 * args.margin_mm

    image_size = (
        mm_to_px(total_width_mm, args.dpi),
        mm_to_px(total_height_mm, args.dpi),
    )
    margin_px = mm_to_px(args.margin_mm, args.dpi)

    image = board.generateImage(image_size, marginSize=margin_px, borderBits=1)
    if not cv2.imwrite(str(output), image):
        raise SystemExit(f"Failed to write board image to {output}")

    print(f"Saved ChArUco board to {output}.")
    print(f"Squares are {args.squares_x} by {args.squares_y}.")
    print(f"Square length is {args.square_length_mm:.2f} mm.")
    print(f"Marker length is {args.marker_length_mm:.2f} mm.")
    print(f"Dictionary is {args.dictionary}.")
    print(
        "Board size without margin is "
        f"{board_width_mm:.1f} mm by {board_height_mm:.1f} mm."
    )
    print(
        "Canvas size with margin is "
        f"{total_width_mm:.1f} mm by {total_height_mm:.1f} mm."
    )
    print(f"Raster size is {image_size[0]} px by {image_size[1]} px at {args.dpi} DPI.")
    print("Print at 100 percent scale with no page scaling.")


if __name__ == "__main__":
    main()
