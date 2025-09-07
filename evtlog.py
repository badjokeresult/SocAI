from hashlib import sha256
from datetime import datetime


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
                 exe_info,
                 hidden_record,
                 src_file,
                 keywords,
                 extra_data_offset,
                 payload_dict):
        self.time_created = int(datetime.strptime(time_created, "%Y-%m-%d %H:%M:%S.%f").timestamp()) if time_created else 0
        self.event_id = int(event_id) if event_id else 0
        self.log_level = int(sha256(log_level.encode("utf-8")).hexdigest(), 16) if log_level else 0
        self.provider = int(sha256(provider.encode("utf-8")).hexdigest(), 16) if provider else 0
        self.channel = int(sha256(channel.encode("utf-8")).hexdigest(), 16) if channel else 0
        self.computer = int(sha256(computer.encode("utf-8")).hexdigest(), 16) if computer else 0
        self.chunk_number = int(chunk_number) if chunk_number else 0
        self.user_id = int(sha256(user_id.encode("utf-8")).hexdigest(), 16) if user_id else 0
        self.map_description = int(sha256(map_description.encode("utf-8")).hexdigest(), 16) if map_description else 0
        self.user_name = int(sha256(user_name.encode("utf-8")).hexdigest(), 16) if user_name else 0
        self.remote_host = int(sha256(remote_host.encode("utf-8")).hexdigest(), 16) if remote_host else 0
        self.payload = int(sha256(EvtLogPayloadSelector.get_payload_type(channel)(**payload_dict)).hexdigest(), 16) if payload_dict else None
        self.exe_info = int(sha256(exe_info.encode("utf-8")).hexdigest(), 16) if exe_info else 0
        self.hidden_record = 1 if hidden_record else 0
        self.src_file = int(sha256(src_file.encode("utf-8")).hexdigest(), 16) if src_file else 0
        self.keywords = int(sha256(keywords.encode("utf-8")).hexdigest(), 16) if keywords else 0
        self.extra_data_offset = int(extra_data_offset)
    
    def __iter__(self):
        return iter([v for v in self.__dict__.values()])


class EvtLogPayloadSelector:
    @staticmethod
    def get_payload_type(channel):
        if channel == "Microsoft-Windows-Sysmon/Operational":
            return SysmonEvtLogPayload
        elif channel == "Security":
            return SecurityEvtLogPayload
        return None


class SysmonEvtLogPayload:
    def __init__(self, **kwargs):
        self.rule_name = int(sha256(kwargs["rule_name"].encode("utf-8")).hexdigest(), 16) if "rule_name" in kwargs else 0
        self.md5 = int(kwargs["md5_hexdigest"], 16) if "md5_hexdigest" in kwargs else 0
        self.sha256 = int(kwargs["sha256_hexdigest"], 16) if "sha256_hexdigest" in kwargs else 0
        self.imphash = int(kwargs["imphash_hexdigest"], 16) if "imphash_hexdigest" in kwargs else 0
        self.process = int(sha256(kwargs["parent_process"].encode("utf-8")).hexdigest(), 16) if "parent_process" in kwargs else 0
        self.parent_command_line = int(sha256(kwargs["parent_command_line"].encode("utf-8")).hexdigest(), 16) if "parent_command_line" in kwargs else 0


    def __bytes__(self):
        attrs = [self.rule_name,
                 self.md5,
                 self.sha256,
                 self.imphash,
                 self.process,
                 self.parent_command_line]
        
        return b"".join(a.to_bytes(32, byteorder="little", signed=False) for a in attrs)


class SecurityEvtLogPayload:
    def __init__(self,
                 src,
                 dst,
                 protocol,
                 pid,
                 direction,
                 layer_name):
        self.src = int(sha256(src.encode("utf-8")).hexdigest(), 16) if src else 0
        self.dst = int(sha256(dst.encode("utf-8")).hexdigest(), 16) if dst else 0
        self.protocol = int(sha256(protocol.encode("utf-8")).hexdigest(), 16) if protocol else 0
        self.pid = int(pid) if pid else 0
        self.direction = int(sha256(direction.encode("utf-8")).hexdigest(), 16) if direction else 0
        self.layer_name = int(sha256(layer_name.encode("utf-8")).hexdigest(), 16) if layer_name else 0
    
    def __bytes__(self):
        attrs = [self.src,
                 self.dst,
                 self.protocol,
                 self.pid,
                 self.direction,
                 self.layer_name]
        
        return b"".join(a.to_bytes(32, byteorder="little", signed=False) for a in attrs)