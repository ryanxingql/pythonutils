import subprocess


def exists_remote(host, port, path):
    """Test if a file exists at path on a host accessible with SSH.
    https://stackoverflow.com/questions/14392432/checking-a-file-existence-on-a-remote-ssh-server-using-python

    Input:
    host, port: for SSH.
    path: file path.

    Return:
    0: exist.
    1: not exist.
    """
    cmd_ = f'ssh -p {port} {host} test -f {path}'
    status = subprocess.call(cmd_, shell=True)
    if status == 0:
        return True
    if status == 1:
        return False
    raise Exception('SSH failed')
