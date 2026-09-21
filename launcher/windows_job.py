"""Own one Windows process and its descendants without searching by PID.

The job handle is never inherited. Closing its final handle terminates every
process in the job, including descendants created after the initial launch.
"""

from __future__ import annotations

import ctypes
from ctypes import wintypes
import os
import subprocess
import threading
from collections.abc import Mapping, Sequence


_SIZE_T = ctypes.c_size_t
_CREATE_SUSPENDED = 0x00000004
_CREATE_UNICODE_ENVIRONMENT = 0x00000400
_EXTENDED_STARTUPINFO_PRESENT = 0x00080000
_CREATE_NO_WINDOW = 0x08000000
_STARTF_USESTDHANDLES = 0x00000100
_HANDLE_FLAG_INHERIT = 0x00000001
_DUPLICATE_SAME_ACCESS = 0x00000002
_PROC_THREAD_ATTRIBUTE_HANDLE_LIST = 0x00020002
_JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
_WAIT_OBJECT_0 = 0
_WAIT_TIMEOUT = 258
_INVALID_DWORD = 0xFFFFFFFF


class _STARTUPINFOW(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD), ("lpReserved", wintypes.LPWSTR),
        ("lpDesktop", wintypes.LPWSTR), ("lpTitle", wintypes.LPWSTR),
        ("dwX", wintypes.DWORD), ("dwY", wintypes.DWORD),
        ("dwXSize", wintypes.DWORD), ("dwYSize", wintypes.DWORD),
        ("dwXCountChars", wintypes.DWORD), ("dwYCountChars", wintypes.DWORD),
        ("dwFillAttribute", wintypes.DWORD), ("dwFlags", wintypes.DWORD),
        ("wShowWindow", wintypes.WORD), ("cbReserved2", wintypes.WORD),
        ("lpReserved2", ctypes.POINTER(wintypes.BYTE)),
        ("hStdInput", wintypes.HANDLE), ("hStdOutput", wintypes.HANDLE),
        ("hStdError", wintypes.HANDLE),
    ]


class _STARTUPINFOEXW(ctypes.Structure):
    _fields_ = [("StartupInfo", _STARTUPINFOW), ("lpAttributeList", wintypes.LPVOID)]


class _PROCESS_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("hProcess", wintypes.HANDLE), ("hThread", wintypes.HANDLE),
        ("dwProcessId", wintypes.DWORD), ("dwThreadId", wintypes.DWORD),
    ]


class _JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_longlong),
        ("PerJobUserTimeLimit", ctypes.c_longlong),
        ("LimitFlags", wintypes.DWORD),
        ("MinimumWorkingSetSize", _SIZE_T), ("MaximumWorkingSetSize", _SIZE_T),
        ("ActiveProcessLimit", wintypes.DWORD), ("Affinity", _SIZE_T),
        ("PriorityClass", wintypes.DWORD), ("SchedulingClass", wintypes.DWORD),
    ]


class _IO_COUNTERS(ctypes.Structure):
    _fields_ = [(name, ctypes.c_ulonglong) for name in (
        "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
        "ReadTransferCount", "WriteTransferCount", "OtherTransferCount",
    )]


class _JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", _JOBOBJECT_BASIC_LIMIT_INFORMATION),
        ("IoInfo", _IO_COUNTERS), ("ProcessMemoryLimit", _SIZE_T),
        ("JobMemoryLimit", _SIZE_T), ("PeakProcessMemoryUsed", _SIZE_T),
        ("PeakJobMemoryUsed", _SIZE_T),
    ]


def _load_kernel32():
    if os.name != "nt":
        return None
    api = ctypes.WinDLL("kernel32", use_last_error=True)
    signatures = {
        "CreateJobObjectW": ([wintypes.LPVOID, wintypes.LPCWSTR], wintypes.HANDLE),
        "SetInformationJobObject": ([wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD], wintypes.BOOL),
        "SetHandleInformation": ([wintypes.HANDLE, wintypes.DWORD, wintypes.DWORD], wintypes.BOOL),
        "CloseHandle": ([wintypes.HANDLE], wintypes.BOOL),
        "GetCurrentProcess": ([], wintypes.HANDLE),
        "DuplicateHandle": ([wintypes.HANDLE, wintypes.HANDLE, wintypes.HANDLE, ctypes.POINTER(wintypes.HANDLE), wintypes.DWORD, wintypes.BOOL, wintypes.DWORD], wintypes.BOOL),
        "InitializeProcThreadAttributeList": ([wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD, ctypes.POINTER(_SIZE_T)], wintypes.BOOL),
        "UpdateProcThreadAttribute": ([wintypes.LPVOID, wintypes.DWORD, _SIZE_T, wintypes.LPVOID, _SIZE_T, wintypes.LPVOID, ctypes.POINTER(_SIZE_T)], wintypes.BOOL),
        "DeleteProcThreadAttributeList": ([wintypes.LPVOID], None),
        "CreateProcessW": ([wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.LPVOID, wintypes.LPVOID, wintypes.BOOL, wintypes.DWORD, wintypes.LPVOID, wintypes.LPCWSTR, ctypes.POINTER(_STARTUPINFOEXW), ctypes.POINTER(_PROCESS_INFORMATION)], wintypes.BOOL),
        "AssignProcessToJobObject": ([wintypes.HANDLE, wintypes.HANDLE], wintypes.BOOL),
        "ResumeThread": ([wintypes.HANDLE], wintypes.DWORD),
        "WaitForSingleObject": ([wintypes.HANDLE, wintypes.DWORD], wintypes.DWORD),
        "GetExitCodeProcess": ([wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)], wintypes.BOOL),
        "TerminateProcess": ([wintypes.HANDLE, wintypes.UINT], wintypes.BOOL),
    }
    for name, (arguments, result) in signatures.items():
        function = getattr(api, name)
        function.argtypes = arguments
        function.restype = result
    return api


_kernel32 = _load_kernel32()


def _error(operation: str) -> OSError:
    error = ctypes.WinError(ctypes.get_last_error())
    error.strerror = f"{operation}: {error.strerror}"
    return error


def _close_handle(handle):
    if handle and not _kernel32.CloseHandle(handle):
        raise _error("CloseHandle")


def _environment_block(env: Mapping[str, str] | None):
    if env is None:
        return None
    entries = []
    for key, value in env.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise TypeError("environment names and values must be strings")
        if not key or "\0" in key or "\0" in value or "=" in key[1:]:
            raise ValueError("invalid environment variable")
        entries.append((key, value))
    entries.sort(key=lambda pair: pair[0].upper())
    return ctypes.create_unicode_buffer("\0".join(f"{key}={value}" for key, value in entries) + "\0\0")


def _duplicate_inheritable(descriptor: int):
    import msvcrt

    duplicate = wintypes.HANDLE()
    current = _kernel32.GetCurrentProcess()
    if not _kernel32.DuplicateHandle(
        current, msvcrt.get_osfhandle(descriptor), current,
        ctypes.byref(duplicate), 0, True, _DUPLICATE_SAME_ACCESS,
    ):
        raise _error("DuplicateHandle")
    return duplicate.value


class OwnedJob:
    """A single launch with deterministic ownership of its entire process tree.

    ``spawn`` returns the initial PID for display only. ``alive`` checks the
    retained initial process handle. Even if that process exits, ``close`` must
    still be called to terminate any remaining descendants and release handles.
    Stdout and stderr share the supplied writable file or file descriptor; the
    caller continues to own that file. Input and omitted output go to NUL.
    """

    def __init__(self):
        self._lock = threading.RLock()
        self._job_handle = None
        self._process_handle = None
        self._spawned = False
        self.pid = None
        if _kernel32 is None:
            raise OSError("OwnedJob requires Windows")
        self._job_handle = _kernel32.CreateJobObjectW(None, None)
        if not self._job_handle:
            raise _error("CreateJobObjectW")
        try:
            if not _kernel32.SetHandleInformation(self._job_handle, _HANDLE_FLAG_INHERIT, 0):
                raise _error("SetHandleInformation")
            limits = _JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
            limits.BasicLimitInformation.LimitFlags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            if not _kernel32.SetInformationJobObject(
                self._job_handle, _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
                ctypes.byref(limits), ctypes.sizeof(limits),
            ):
                raise _error("SetInformationJobObject")
        except BaseException:
            self.close()
            raise

    def spawn(self, argv: Sequence[str], env: Mapping[str, str] | None = None, stdout=None) -> int:
        if isinstance(argv, (str, bytes)) or not argv:
            raise ValueError("argv must be a nonempty sequence of arguments")
        arguments = [os.fsdecode(os.fspath(argument)) for argument in argv]
        if not arguments[0] or any("\0" in argument for argument in arguments):
            raise ValueError("argv contains an empty executable or a NUL")
        command = ctypes.create_unicode_buffer(subprocess.list2cmdline(arguments))
        environment = _environment_block(env)
        with self._lock:
            if not self._job_handle or self._spawned:
                raise RuntimeError("OwnedJob permits exactly one process launch")
            self._spawned = True
            inherited = []
            descriptors = []
            attribute_list = None
            initialized = False
            process = _PROCESS_INFORMATION()
            try:
                input_fd = os.open(os.devnull, os.O_RDONLY | getattr(os, "O_BINARY", 0))
                descriptors.append(input_fd)
                inherited.append(_duplicate_inheritable(input_fd))
                if stdout is None:
                    output_fd = os.open(os.devnull, os.O_WRONLY | getattr(os, "O_BINARY", 0))
                    descriptors.append(output_fd)
                elif isinstance(stdout, int):
                    output_fd = stdout
                else:
                    stdout.flush()
                    output_fd = stdout.fileno()
                inherited.append(_duplicate_inheritable(output_fd))
                size = _SIZE_T()
                _kernel32.InitializeProcThreadAttributeList(None, 1, 0, ctypes.byref(size))
                if not size.value:
                    raise _error("InitializeProcThreadAttributeList")
                attribute_list = ctypes.create_string_buffer(size.value)
                if not _kernel32.InitializeProcThreadAttributeList(attribute_list, 1, 0, ctypes.byref(size)):
                    raise _error("InitializeProcThreadAttributeList")
                initialized = True
                handle_list = (wintypes.HANDLE * len(inherited))(*inherited)
                if not _kernel32.UpdateProcThreadAttribute(
                    attribute_list, 0, _PROC_THREAD_ATTRIBUTE_HANDLE_LIST,
                    handle_list, ctypes.sizeof(handle_list), None, None,
                ):
                    raise _error("UpdateProcThreadAttribute")
                startup = _STARTUPINFOEXW()
                startup.StartupInfo.cb = ctypes.sizeof(startup)
                startup.StartupInfo.dwFlags = _STARTF_USESTDHANDLES
                startup.StartupInfo.hStdInput = inherited[0]
                startup.StartupInfo.hStdOutput = inherited[1]
                startup.StartupInfo.hStdError = inherited[1]
                startup.lpAttributeList = ctypes.cast(attribute_list, wintypes.LPVOID)
                flags = (_CREATE_SUSPENDED | _CREATE_NO_WINDOW |
                         _CREATE_UNICODE_ENVIRONMENT | _EXTENDED_STARTUPINFO_PRESENT)
                if not _kernel32.CreateProcessW(
                    arguments[0], command, None, None, True, flags,
                    environment, None, ctypes.byref(startup), ctypes.byref(process),
                ):
                    raise _error("CreateProcessW")
                self._process_handle = process.hProcess
                self.pid = int(process.dwProcessId)
                if not _kernel32.AssignProcessToJobObject(self._job_handle, self._process_handle):
                    raise _error("AssignProcessToJobObject")
                if _kernel32.ResumeThread(process.hThread) == _INVALID_DWORD:
                    raise _error("ResumeThread")
                return self.pid
            except BaseException:
                # Before assignment, closing the job cannot kill the suspended
                # child. Terminate that exact original handle on every failure.
                # A signal can interrupt ctypes after CreateProcess has filled
                # the result structure but before Python stores the handle.
                if process.hProcess and not self._process_handle:
                    self._process_handle = process.hProcess
                try:
                    if self._process_handle and _kernel32.WaitForSingleObject(self._process_handle, 0) != _WAIT_OBJECT_0:
                        if not _kernel32.TerminateProcess(self._process_handle, 1):
                            if _kernel32.WaitForSingleObject(self._process_handle, 0) != _WAIT_OBJECT_0:
                                raise _error("TerminateProcess")
                finally:
                    self.close()
                raise
            finally:
                try:
                    _close_handle(process.hThread)
                finally:
                    if initialized:
                        _kernel32.DeleteProcThreadAttributeList(attribute_list)
                    try:
                        for handle in inherited:
                            _close_handle(handle)
                    finally:
                        for descriptor in descriptors:
                            os.close(descriptor)

    def alive(self) -> bool:
        with self._lock:
            if not self._process_handle:
                return False
            result = _kernel32.WaitForSingleObject(self._process_handle, 0)
            if result == _WAIT_OBJECT_0:
                return False
            if result == _WAIT_TIMEOUT:
                return True
            raise _error("WaitForSingleObject")

    def exit_code(self) -> int | None:
        """Return the original process result, or None while it is running."""
        with self._lock:
            if not self._process_handle:
                raise RuntimeError("OwnedJob has no retained launched process")
            result = _kernel32.WaitForSingleObject(self._process_handle, 0)
            if result == _WAIT_TIMEOUT:
                return None
            if result != _WAIT_OBJECT_0:
                raise _error("WaitForSingleObject")
            code = wintypes.DWORD()
            if not _kernel32.GetExitCodeProcess(self._process_handle, ctypes.byref(code)):
                raise _error("GetExitCodeProcess")
            return int(code.value)

    def close(self):
        """Close the noninherited job, terminating only its owned descendants."""
        with self._lock:
            if self._job_handle:
                _close_handle(self._job_handle)
                self._job_handle = None
            if self._process_handle:
                try:
                    result = _kernel32.WaitForSingleObject(self._process_handle, 10000)
                    if result == _WAIT_TIMEOUT:
                        raise TimeoutError("owned process did not exit after job closure")
                    if result != _WAIT_OBJECT_0:
                        raise _error("WaitForSingleObject")
                finally:
                    _close_handle(self._process_handle)
                    self._process_handle = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def __del__(self):
        try:
            self.close()
        except BaseException:
            pass
