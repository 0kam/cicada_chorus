library(tidyverse)
library(soundecology)
setwd("~/cicada_chorus/")

dirs <- list.dirs("/media/HDD10TB/cicadasong2023/", recursive = F) %>%
  str_subset("NE")

map(dirs,
    function(d){
      st <- basename(d)
      out <- str_c("predictions/sound_index/", st, "_", index, ".csv")
      multiple_sounds(
        directory = str_c(d, "/AUDIO/"),
        resultfile = out,
        soundindex = "ndsi",
        no_cores = 1
      )
    }
)
