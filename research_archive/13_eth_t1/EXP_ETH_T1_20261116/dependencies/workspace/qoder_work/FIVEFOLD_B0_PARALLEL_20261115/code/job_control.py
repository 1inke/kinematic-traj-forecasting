"""Windows child containment: terminating the owner closes and kills its child job."""
import ctypes
from ctypes import wintypes as wt
import subprocess


class BasicLimits(ctypes.Structure):
    _fields_ = [('PerProcessUserTimeLimit', ctypes.c_int64), ('PerJobUserTimeLimit', ctypes.c_int64),
                ('LimitFlags', wt.DWORD), ('MinimumWorkingSetSize', ctypes.c_size_t),
                ('MaximumWorkingSetSize', ctypes.c_size_t), ('ActiveProcessLimit', wt.DWORD),
                ('Affinity', ctypes.c_size_t), ('PriorityClass', wt.DWORD), ('SchedulingClass', wt.DWORD)]


class IoCounters(ctypes.Structure):
    _fields_ = [(name, ctypes.c_uint64) for name in ('ReadOperationCount', 'WriteOperationCount',
                  'OtherOperationCount', 'ReadTransferCount', 'WriteTransferCount', 'OtherTransferCount')]


class ExtendedLimits(ctypes.Structure):
    _fields_ = [('BasicLimitInformation', BasicLimits), ('IoInfo', IoCounters),
                ('ProcessMemoryLimit', ctypes.c_size_t), ('JobMemoryLimit', ctypes.c_size_t),
                ('PeakProcessMemoryUsed', ctypes.c_size_t), ('PeakJobMemoryUsed', ctypes.c_size_t)]


class ChildJob:
    def __init__(self):
        self.api = ctypes.WinDLL('kernel32', use_last_error=True)
        self.api.CreateJobObjectW.argtypes = [ctypes.c_void_p, wt.LPCWSTR]
        self.api.CreateJobObjectW.restype = wt.HANDLE
        self.api.SetInformationJobObject.argtypes = [wt.HANDLE, ctypes.c_int, ctypes.c_void_p, wt.DWORD]
        self.api.SetInformationJobObject.restype = wt.BOOL
        self.api.AssignProcessToJobObject.argtypes = [wt.HANDLE, wt.HANDLE]
        self.api.AssignProcessToJobObject.restype = wt.BOOL
        self.api.TerminateJobObject.argtypes = [wt.HANDLE, wt.UINT]
        self.api.CloseHandle.argtypes = [wt.HANDLE]
        self.handle = self.api.CreateJobObjectW(None, None)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        limits = ExtendedLimits()
        limits.BasicLimitInformation.LimitFlags = 0x00002000  # KILL_ON_JOB_CLOSE, no memory limit.
        if not self.api.SetInformationJobObject(self.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
            raise ctypes.WinError(ctypes.get_last_error())

    def popen(self, command, **kwargs):
        # Suspend before assignment so the venv redirector cannot spawn outside containment.
        child = subprocess.Popen(command, creationflags=subprocess.CREATE_NO_WINDOW | 0x00000004, **kwargs)
        try:
            if not self.api.AssignProcessToJobObject(self.handle, wt.HANDLE(int(child._handle))):
                raise ctypes.WinError(ctypes.get_last_error())
            native = ctypes.WinDLL('ntdll')
            native.NtResumeProcess.argtypes = [wt.HANDLE]
            native.NtResumeProcess.restype = ctypes.c_long
            code = native.NtResumeProcess(wt.HANDLE(int(child._handle)))
            if code != 0:
                raise OSError('NtResumeProcess status ' + hex(code & 0xffffffff))
        except BaseException:
            child.kill(); child.wait()
            raise
        return child

    def terminate(self):
        if self.handle:
            self.api.TerminateJobObject(self.handle, 1)

    def close(self):
        if self.handle:
            self.api.CloseHandle(self.handle)
            self.handle = None
