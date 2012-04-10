""" Utility for running R commands without Rpy """

from subprocess import check_call, PIPE, CalledProcessError

from ..tmpdirs import InTemporaryDirectory

def Rcmd(cmd, stderr=None):
    """ Execute R commands in string `cmd`, return terminal output

    Parameters
    ----------
    cmd : str
        string containing R command(s) to execute
    stderr : None or file-like, optional
        Pipe for stderr. None does not trap stderr. Use PIPE from the
        ``subprocess`` module to supress stderr output

    Returns
    -------
    output : str
        R console output
    """
    with InTemporaryDirectory():
        cmd = "options(width=1000)\n" + cmd
        open('in.R', 'wt').write(cmd)
        check_call('R CMD BATCH --slave --no-timing in.R out.txt',
                   stderr=stderr, shell=True)
        res = open('out.txt', 'rt').read().strip()
    return res


try:
    Rcmd('', stderr=PIPE)
except CalledProcessError:
    have_R = False
else:
    have_R = True
