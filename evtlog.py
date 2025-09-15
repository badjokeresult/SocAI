from datetime import datetime
from hashlib import sha256
from functools import reduce


class EvtLog:
    def __init__(self, event_id, time_created, channel, computer, event_data):
        self.event_id = int(event_id) if event_id else 0
        
        dt = datetime.strptime(time_created, "%Y-%m-%d %H:%M:%S.%f%z") if time_created else datetime(1970, 1, 1)
        self.weekday = dt.weekday()
        self.time_of_day = dt.hour * 3600 + dt.minute * 60 + dt.second
        
        self.channel = int(sha256(channel.encode("utf-8")).hexdigest(), 16) % 2**64 if channel else 0
        self.computer = int(sha256(computer.encode("utf-8")).hexdigest(), 16) % 2**64 if computer else 0
        event_datas = []
        #print(event_data)
        for ed in event_data:
            event_datas.append(hash(EventData(ed)))
        self.event_data = reduce(lambda x, y: x + y, event_datas) % 2 ** 64 if event_datas else 0
    
    def __repr__(self):
        return f"EvtLog(event_id={self.event_id}, weekday={self.weekday}, time_of_day={self.time_of_day}, channel={self.channel}, computer={self.computer}, event_data={self.event_data})"
    
    def get_values(self):
        return [self.event_id, self.weekday, self.time_of_day, self.channel, self.computer, self.event_data]


class EventData:
    def __init__(self, data):
        self.name = int(sha256(data["@Name"].encode("utf-8")).hexdigest(), 16) if data["@Name"] else 0
        if "#text" in data:
            self.text = int(sha256(data["#text"].encode("utf-8")).hexdigest(), 16) if data["#text"] else 0
        else:
            self.text = 0
    
    def __hash__(self):
        return (self.name + self.text) % 2 ** 64