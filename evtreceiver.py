import concurrent.futures
import sys

from datetime import datetime

import win32api
import win32con
import win32evtlog
import win32evtlogutil
import win32security


class EvtLogReceiver:
    def __init__(self, start_datetime, end_datetime, computers, log_types=None):
        self.start_datetime = start_datetime if start_datetime else datetime(1970, 1, 1)
        self.end_datetime = end_datetime if end_datetime else datetime(2038, 1, 19)
        if not computers:
            raise ValueError("Computers list should be provided")
        self.computers = computers
        self.log_types = log_types if log_types else ("Security", "Microsoft-Windows-Sysmon%4Operational")

    def launch(self):
        events = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            future_computers = {executor.submit(self.__read_log_from_computer(log_type, computer)): (log_type, computer) for log_type in self.log_types for computer in self.computers}
            for future in concurrent.futures.as_completed(future_computers):
                events.extend(future.result())
        return events

    def __read_log_from_computer(self, log_type, computer):
        h = win32evtlog.OpenEventLog(computer, log_type)
        records_num = win32evtlog.GetNumberOfEventLogRecords(h)
        actual_num = 0
        records = []
        while True:
            objects = win32evtlog.ReadEventLog(h, win32evtlog.EVENTLOG_BACKWARDS_READ | win32evtlog.EVENTLOG_SEQUENTIAL_READ, 0)
            if not objects:
                break
            records.extend(objects)
            actual_num += len(objects)
        
        if actual_num < records_num:
            print(f"Could not capture all records, expected {records_num} got {actual_num}", file=sys.stderr)
        else:
            print("All records was captured", file=sys.stdout)
        
        return records


def ReadLog(computer, logType="Application", dumpEachRecord=0):
    # read the entire log back.
    h = win32evtlog.OpenEventLog(computer, logType)
    numRecords = win32evtlog.GetNumberOfEventLogRecords(h)
    # print(f"There are {numRecords} records")

    num = 0
    while 1:
        objects = win32evtlog.ReadEventLog(
            h,
            win32evtlog.EVENTLOG_BACKWARDS_READ | win32evtlog.EVENTLOG_SEQUENTIAL_READ,
            0,
        )
        if not objects:
            break
        for object in objects:
            # get it for testing purposes, but don't print it.
            msg = win32evtlogutil.SafeFormatMessage(object, logType)
            if object.Sid is not None:
                try:
                    domain, user, typ = win32security.LookupAccountSid(
                        computer, object.Sid
                    )
                    sidDesc = f"{domain}/{user}"
                except win32security.error:
                    sidDesc = str(object.Sid)
                user_desc = f"Event associated with user {sidDesc}"
            else:
                user_desc = None
            if dumpEachRecord:
                print(
                    "Event record from {!r} generated at {}".format(
                        object.SourceName, object.TimeGenerated.Format()
                    )
                )
                if user_desc:
                    print(user_desc)
                try:
                    print(msg)
                except UnicodeError:
                    print("(unicode error printing message: repr() follows...)")
                    print(repr(msg))

        num += len(objects)

    if numRecords == num:
        print("Successfully read all", numRecords, "records")
    else:
        print(
            "Couldn't get all records - reported %d, but found %d" % (numRecords, num)
        )
        print(
            "(Note that some other app may have written records while we were running!)"
        )
    win32evtlog.CloseEventLog(h)