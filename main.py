import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib
import matplotlib.pyplot as plt


def main():
    input_path = Path("input.txt")
    output_path = Path("output.txt")

    if not input_path.exists():
        raise SystemExit("Не найден input.txt")
    if not output_path.exists():
        raise SystemExit("Не найден output.txt. Сначала запустите CUDA/C++ программу.")

    with input_path.open("r", encoding="utf-8") as input_file:
        n, al, ar, cl, cr, u = map(float, input_file.readline().split())

    with output_path.open("r", encoding="utf-8") as output_file:
        lines = [line.strip() for line in output_file]

    try:
        thickness_line = lines.index("Толщина покрытия на катоде:") + 1
    except ValueError as exc:
        raise SystemExit("В output.txt не найдена строка: Толщина покрытия на катоде:") from exc

    thickness = list(map(float, lines[thickness_line].split()))
    x = list(range(int(cl), int(cr) + 1))

    plt.plot(x, thickness, marker="o")
    plt.xlabel("i")
    plt.ylabel("Толщина покрытия на катоде")
    plt.title("График толщины покрытия на катоде")
    plt.grid(True)

    if matplotlib.get_backend().lower() == "agg":
        image_path = Path("thickness.png")
        plt.savefig(image_path, dpi=150, bbox_inches="tight")
        print(f"График сохранен в {image_path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
