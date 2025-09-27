from datetime import datetime

import xmltodict
from Evtx.Evtx import Evtx

from evtlog import EvtLog


class EvtLogReceiver:
    def __init__(self, start_datetime, end_datetime, log_path=None):
        self.start_datetime = start_datetime if start_datetime else datetime(1970, 1, 1)
        self.end_datetime = end_datetime if end_datetime else datetime(2038, 1, 19)
        self.log_path = log_path if log_path else r"C:\Windows\System32\Winevt\Logs\ForwardedEvents.evtx"
    
    def receive(self):
        with Evtx(self.log_path) as log:
            for event in log.records():
                dict_event = xmltodict.parse(event)
                yield EvtLog(dict_event["Event"]["System"]["EventID"]["#text"],
                             dict_event["Event"]["System"]["TimeCreated"]["@SystemTime"],
                             dict_event["Event"]["System"]["Channel"],
                             dict_event["Event"]["System"]["Computer"],
                             dict_event["Event"]["EventData"]["Data"]).get_values()