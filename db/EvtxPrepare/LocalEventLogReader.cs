using System.Diagnostics.Eventing.Reader;

namespace EvtxPrepare
{
    public static class LocalEventLogReader
    {
        public static IEnumerable<EventRecord> GetEventRecords(string path)
        {
            using var reader = new EventLogReader(path, PathType.FilePath);
            EventRecord record;
            while ((record = reader.ReadEvent()) != null)
            {
                yield return record;
            }
        }
    }
}
