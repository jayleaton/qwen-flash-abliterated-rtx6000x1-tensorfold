    const value = if (std.c.getenv("TF_EXAMPLE_DRAFT_HEAD")) |v| std.mem.span(v) else "bf16";
