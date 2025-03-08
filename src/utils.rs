use once_cell::sync::OnceCell;

// Global log configuration
pub(crate) static LOG_DESTINATION: OnceCell<String> = OnceCell::new();

// Helper macro for dual logging (to both stdout and log file)
#[macro_export]
macro_rules! dual_log {
    ($level:expr, $($arg:tt)+) => {{
        let msg = format!($($arg)+);
        if $crate::utils::LOG_DESTINATION.get().map_or(false, |d| d == "both") {
            println!("{}: {}", $level, msg);
        }
        match $level {
            "INFO" => tracing::info!("{}", msg),
            "WARN" => tracing::warn!("{}", msg),
            "ERROR" => tracing::error!("{}", msg),
            "DEBUG" => tracing::debug!("{}", msg),
            _ => tracing::trace!("{}", msg),
        }
    }};
}

// Convenience macros for each log level
#[macro_export]
macro_rules! dual_info {
    ($($arg:tt)+) => { $crate::dual_log!("INFO", $($arg)+) };
}

#[macro_export]
macro_rules! dual_warn {
    ($($arg:tt)+) => { $crate::dual_log!("WARN", $($arg)+) };
}

#[macro_export]
macro_rules! dual_error {
    ($($arg:tt)+) => { $crate::dual_log!("ERROR", $($arg)+) };
}

#[macro_export]
macro_rules! dual_debug {
    ($($arg:tt)+) => { $crate::dual_log!("DEBUG", $($arg)+) };
}
