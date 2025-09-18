namespace EvtxPrepare;

class Program
{
    public static async Task Main(string[] args)
    {
        var host = "localhost";
        var port = 5432;
        var database = "evtlogs";
        var username = "socai";
        var password = "ch[JN60L47}0";

        var logPath = @"C:\Windows\System32\Winevt\Logs\ForwardedEvents.evtx";
        var records = new List<LocalEventRecord>();
        foreach (var record in LocalEventLogReader.GetEventRecords(logPath))
            records.Add(new LocalEventRecord(record));

        using var db = new LocalEventLogSaver(host, port, database, username, password);
        await Task.Run(() => db.SaveRecordsAsync(records));
    }
}