# Filter+aggregate a Google Books v3 5-gram stream into per-decade
# (center, context) co-occurrence counts, restricted to a vocabulary.
# Usage: awk -f ngram_cooc.awk vocab.txt - < fivegram_stream
# vocab.txt: one lowercase word per line.
# Input:  w1 w2 w3 w4 w5 \t year,match_count,volume_count \t ...
# Output: decade \t center \t context \t summed_match_count
# The center of each 5-gram is w3; contexts are w1,w2,w4,w5 (window +-2).
# Every corpus position is the center of exactly one 5-gram, so this yields
# unbiased window-2 co-occurrence counts (the HistWords construction).
FNR == NR { vocab[$1] = 1; next }
BEGIN { FS = "\t" }
{
    n = split($1, tok, " ")
    if (n != 5) next
    ok = 1
    for (i = 1; i <= 5; i++) {
        if (tok[i] !~ /^[A-Za-z]+$/) { ok = 0; break }
        tok[i] = tolower(tok[i])
    }
    if (!ok) next
    if (!(tok[3] in vocab)) next
    delete dec
    for (f = 2; f <= NF; f++) {
        split($f, y, ",")
        d = int(y[1] / 10) * 10
        if (d < 1800) continue
        dec[d] += y[2]
    }
    for (d in dec) {
        w = dec[d]
        for (i = 1; i <= 5; i++) {
            if (i == 3 || !(tok[i] in vocab)) continue
            acc[d "\t" tok[3] "\t" tok[i]] += w
        }
    }
}
END { for (k in acc) print k "\t" acc[k] }
