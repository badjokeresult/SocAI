import sys
import os


def main():
    if sys.argv[1] == "train":
        from training import train
        train()
    elif sys.argv[1] == "detect":
        launch_func = None
        if os.name() == "nt":
            from ntservice import launch_service
            launch_func = launch_service
        else:
            from systemdservice import launch_service
            launch_func = launch_service
        launch_func()


if __name__ == "__main__":
    main()
