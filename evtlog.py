class EvtLog:
    def __init__(self,
                 time_created,
                 event_id,
                 log_level,
                 provider,
                 channel,
                 computer,
                 chunk_number,
                 user_id,
                 map_description,
                 user_name,
                 remote_host,
                 payloads,
                 exe_info,
                 hidden_record,
                 src_file,
                 keywords,
                 extra_data_offset,
                 payload):
        self.time_created = time_created
        self.event_id = event_id
        self.log_level = log_level
        self.provider = provider
        self.channel = channel
        self.computer = computer
        self.chunk_number = chunk_number
        self.user_id = user_id
        self.map_description = map_description
        self.user_name = user_name
        self.remote_host = remote_host
        self.payloads = payloads
        self.exe_info = exe_info
        self.hidden_record = hidden_record
        self.src_file = src_file
        self.keywords = keywords
        self.extra_data_offset = extra_data_offset
        self.payload = payload
    
    