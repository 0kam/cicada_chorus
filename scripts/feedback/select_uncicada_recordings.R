library(tidyverse)
setwd("/home/okamoto/cicada_chorus")

list.files(
    "predictions/",
    pattern = ".csv",
    recursive = TRUE,
    full.names = TRUE
) %>%
map_dfr(read_csv) %>%
pivot_longer(
    cols = -c("file_name"),
    names_to = "species",
    values_to = "exists"
) %>%
group_by(file_name) %>%
summarise(
    exists = max(exists)
) %>%
filter(exists == 0) %>%
sample_n(500) %>%
mutate(site_name = str_extract(file_name, "NE\\d{2}_[a-z]*")) %>%
mutate(
    out_file = str_c("data/uncicada/", site_name, "_", basename(file_name))
) %>%
select(
    file_name,
    out_file
) -> df

library(reticulate)
torchaudio <- import("torchaudio")

for (i in seq_len(nrow(df))) {
    loaded <- torchaudio$load(df$file_name[i])
    audio <- loaded[[1]]$mean(0L)$unsqueeze(0L)
    sr <- loaded[[2]]
    torchaudio$save(
        df$out_file[i],
        audio,
        sr
    )
}
