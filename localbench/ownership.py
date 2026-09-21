import socket


def check_port(port):
    sock = socket.socket()
    try:
        sock.bind(("127.0.0.1", port))
    except OSError as e:
        raise RuntimeError("occupied_port: do not claim or terminate existing listener") from e
    finally:
        sock.close()


def validate_process(saved, observed):
    if not saved or not observed or saved.get("owner") != "localbench":
        return False
    return saved.get("pid") == observed.get("pid") and saved.get("created") == observed.get("created") and saved.get("command_hash") == observed.get("command_hash")


def validate_container(labels, run_id, job_id):
    return labels.get("localbench.owner") == "localbench" and labels.get("localbench.run") == run_id and labels.get("localbench.job") == job_id
