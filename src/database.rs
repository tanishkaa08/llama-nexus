use rusqlite::{Connection, Result};
use serde::{Deserialize, Serialize};

// FIX: Add 'pub' to make this struct visible to main.rs
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

// FIX: Add 'pub' to make this function visible to main.rs
pub fn connect() -> Result<Connection> {
    let conn = Connection::open("chat_history.db")?;
    conn.execute(
        "CREATE TABLE IF NOT EXISTS chat_history (
            session_id TEXT NOT NULL,
            role       TEXT NOT NULL,
            content    TEXT NOT NULL,
            timestamp  INTEGER NOT NULL
        )",
        [],
    )?;
    Ok(conn)
}

// FIX: Add 'pub'
pub fn get_history(conn: &Connection, session_id: &str) -> Result<Vec<ChatMessage>> {
    let mut stmt = conn.prepare(
        "SELECT role, content FROM chat_history WHERE session_id = ?1 ORDER BY timestamp ASC",
    )?;
    let msg_iter = stmt.query_map([session_id], |row| {
        Ok(ChatMessage {
            role: row.get(0)?,
            content: row.get(1)?,
        })
    })?;

    let mut history = Vec::new();
    for msg in msg_iter {
        history.push(msg?);
    }
    Ok(history)
}

// FIX: Add 'pub'
pub fn save_message(conn: &Connection, session_id: &str, message: &ChatMessage) -> Result<()> {
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();

    conn.execute(
        "INSERT INTO chat_history (session_id, role, content, timestamp) VALUES (?1, ?2, ?3, ?4)",
        &[
            &session_id.to_string(),
            &message.role,
            &message.content,
            &timestamp.to_string(),
        ],
    )?;
    Ok(())
}