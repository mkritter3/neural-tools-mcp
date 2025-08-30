# What Is Stored in the Memory Database?

## Current Database: `.claude/memory/dynamic-memory.db`

### What Gets Stored (with new UserPromptSubmit hook):

1. **User Prompts** - Each prompt you send to Claude
   - Stored as a summary (first 100 chars)
   - Tagged with conversation ID
   - Timestamped for temporal tracking

2. **Conversation Context** - Metadata about interactions
   - Conversation ID for grouping related prompts
   - Timestamps for temporal relevance
   - Token counts for budget management

3. **NOT Currently Stored** (but could be added):
   - Claude's responses (would need PostResponse hook)
   - Full prompt content (currently just summaries)
   - Semantic embeddings (for better search)

### Storage Tiers:
- **HOT** (< 24 hours) - Recent conversations, instantly accessible
- **WARM** (1-7 days) - Recent history, compressed
- **COLD** (7-30 days) - Archived, heavily compressed
- **Auto-expired** (> 30 days) - Automatically deleted

### Current Status:
- **2 test entries** from earlier testing
- **UserPromptSubmit hook** configured to store new prompts
- **PreCompact/SessionStart hooks** configured to retrieve and inject

### What This Enables:
1. **Continuous Learning** - Every conversation adds to memory
2. **Context Preservation** - Across sessions and compactions
3. **No Token Waste** - Storage happens without injection
4. **Project-Local History** - All memory stays with the project

### What's Missing:
- Claude's responses aren't being stored yet
- Semantic embeddings for better retrieval
- Full conversation threads (prompt + response pairs)

### To See What's Stored:
```bash
# View recent memories
sqlite3 .claude/memory/dynamic-memory.db \
  "SELECT summary, datetime(timestamp, 'unixepoch') FROM memory_chunks_v2 ORDER BY timestamp DESC LIMIT 10;"

# Count by tier
sqlite3 .claude/memory/dynamic-memory.db \
  "SELECT storage_tier, COUNT(*) FROM memory_chunks_v2 GROUP BY storage_tier;"
```