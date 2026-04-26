import sys

from main import main


if __name__ == "__main__":
    if "--mode" not in sys.argv:
        sys.argv.extend(["--mode", "deep"])
    if "--detector" not in sys.argv:
        sys.argv.extend(["--detector", "opencv"])
    main()
