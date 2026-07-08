# Filter+aggregate a Google Books v3 1-gram stream into per-decade counts.
# Input:  token \t year,match_count,volume_count \t ...
# Output: lowercased_token \t decade \t summed_match_count
BEGIN { FS = "\t" }
{
    if ($1 !~ /^[A-Za-z]+$/) next
    w = tolower($1)
    delete dec
    for (f = 2; f <= NF; f++) {
        split($f, y, ",")
        d = int(y[1] / 10) * 10
        if (d < 1800) continue
        dec[d] += y[2]
    }
    for (d in dec) acc[w "\t" d] += dec[d]
}
END { for (k in acc) print k "\t" acc[k] }
