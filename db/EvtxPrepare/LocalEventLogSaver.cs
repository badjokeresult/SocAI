using System.Diagnostics.Eventing.Reader;

using Microsoft.EntityFrameworkCore;

namespace EvtxPrepare
{
    public class LocalEventLogSaver : DbContext
    {
        private readonly string _host;
        private readonly int _port;
        private readonly string _database;
        private readonly string _username;
        private readonly string _password;

        public DbSet<LocalEventRecord> Events { get; set; }
        
        public LocalEventLogSaver(string host, int port, string database, string username, string password)
        {
            Database.EnsureCreated();
            _host = host;
            _port = port;
            _database = database;
            _username = username;
        }

        protected override void OnConfiguring(DbContextOptionsBuilder optionsBuilder)
        {
            optionsBuilder.UseNpgsql($"Host={_host};Port={_port};Database={_database};Username={_username};Password={_password}");
        }

        public async Task SaveRecordsAsync(IEnumerable<LocalEventRecord> records)
        {
            await Events.AddRangeAsync(records);
            await SaveChangesAsync();
        }

    }

    public class LocalEventRecord
    {
        public long Id { get; set; }
        public string MachineName { get; set; }
        public string LogName { get; set; }
        public List<string> Properties { get; set; }
        public DateTime TimeCreated { get; set; }
        public string TaskName { get; set; }
        public string ProviderName { get; set; }

        public LocalEventRecord(EventRecord record)
        {
            if (!record.RecordId.HasValue)
                throw new ArgumentException("Record ID value cannot be null");
            if (!record.TimeCreated.HasValue)
                throw new ArgumentException("Creation time cannot be null");

            var properties = record.Properties.Select(x => x.Value.ToString()).Where(x => x != null).Select(x => x!).ToList()!;
            
            Id = record.RecordId.Value;
            MachineName = record.MachineName;
            LogName = record.LogName;
            Properties = properties;
            TimeCreated = record.TimeCreated.Value;
            TaskName = record.TaskDisplayName;
            ProviderName = record.ProviderName;
        }
    }
}
