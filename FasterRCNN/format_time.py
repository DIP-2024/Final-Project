def format_time(time_sec):
    h = time_sec // 3600
    m = (time_sec % 3600) // 60
    s = time_sec % 60
    return f"{int(h)}h {int(m)}m {int(s)}s"