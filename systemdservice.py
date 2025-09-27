def launch_service():
    import systemd.daemon

    print("Starting the daemon up...")
    systemd.daemon.notify("READY=1")
    print("Daemon was launched successfully")

    while True:
        pass