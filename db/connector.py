import asyncio

import psycopg

from Evtx.Evtx import Evtx
import win32evtlog

async def get_forwarded_event_from(path):
    with Evtx(path) as log:
        record = log.get_record()


async def save_event(event, **conn_values):
    pass


async def main():
    # with await psycopg.AsyncConnection.connect(conn_str) as conn:
    #     with await conn.execute(exec_str) as cursor:
    #         result = await cursor.fetchone()
    #         print(result)
    evtlog_path = r"C:\Windows\System32\Winevt\Logs\ForwardedEvents.evtx"
    conn_values = {
        "dbname": "evtlogs",
        "host": "localhost",
        "user": "socai",
        "password": "ch[JN60L47}0",
        "port": "5432"
    }
    
    for event in await get_forwarded_event_from(evtlog_path):
        await save_event(event, **conn_values)


if __name__ == "__main__":
    main()
