library(tidyverse)
library(lubridate)
library(hms)

setwd("/home/okamoto/cicada_chorus/")

plot_all <- function(csv_path) {
  site_name <- str_remove(basename(csv_path), ".csv")
  df <- read_csv(csv_path) %>%
    select(-kumazemi) %>%
    rename(
      ツクツクボウシ = tsukutsukuboushi,
      ニイニイゼミ = niiniizemi,
      ミンミンゼミ = minminzemi,
      ヒグラシ = higurashi,
      アブラゼミ = aburazemi
    ) %>%
    mutate(
      datetime = file_name %>%
        str_extract("\\d{8}_\\d{4}") %>%
        ymd_hm()
    ) %>%
    select(-file_name) %>%
    pivot_longer(cols = -datetime, names_to = "species", values_to = "is_calling") %>%
    mutate(is_calling = as.logical(is_calling)) %>%
    mutate(
      date = date(datetime),
      time = as_hms(datetime)
    ) 
  
  species <- df %>%
    pull(species) %>%
    unique()
  
  p_all <- df %>%
    filter(is_calling) %>%
    ggplot(aes(x = date, y = time, color = species)) +
    geom_jitter(size = 3, alpha = 0.5) +
    theme_bw()
  
  p_daily <- 
    df %>%
    group_by(date, species) %>%
    summarise(freq = mean(as.integer(is_calling))) %>%
    ggplot(aes(x = date, y = species, fill = species, alpha = freq)) +
    geom_tile() +
    theme_bw() +
    theme(
      text = element_text(size = 15),
      axis.text.x = element_text(angle = 45, hjust = 1),
  　　plot.title = element_text(hjust = 0.5)  
    ) + 
    scale_alpha_continuous(name = "鳴き声の頻度") +
    labs(title = "セミの鳴き声頻度の季節変化", x = "日付", y = "セミの種類") +
    guides(fill = "none")
  
  p_hourly <- df %>%
    mutate(hour = hour(datetime)) %>%
    group_by(hour, species) %>%
    summarise(freq = mean(as.integer(is_calling))) %>%
    ggplot(aes(x = hour, y = species, fill = species, alpha = freq)) +
    geom_tile() +
    theme_bw() +
    theme(
      text = element_text(size = 15),
      axis.text.x = element_text(angle = 45, hjust = 1)
    ) +
    scale_alpha_continuous(name = "鳴き声の頻度") +
    labs(title = "セミの鳴き声頻度の時間変化", x = "日付", y = "セミの種類") +
    guides(fill = "none") +
    theme(plot.title = element_text(hjust = 0.5))
  
  out_d <- dirname(csv_path)
  ggsave(str_c(out_d, "/", site_name, "_all.png"), p_all, width = 16, height = 16)
  ggsave(str_c(out_d, "/", site_name, "_daily.png"), p_daily, width = 8, height = 4)
  ggsave(str_c(out_d, "/", site_name, "_hourly.png"), p_hourly, width = 8, height = 4)
}

list.files("predictions/resnet50_spectrogram_20230824230657/",
           pattern = ".csv", full.names = T) %>%
  map(plot_all)


csv_path <- "predictions/resnet50_spectrogram_20230824230657//NE04_yamao.csv"
kumazemi <- read_csv(csv_path) %>%
  filter(kumazemi == 1) %>%
  sample_n(30)

copy_to <- map(basename(kumazemi %>% pull(file_name)), function(f) str_c(out_d, "/kumazemis/", f)) %>%
  unlist()

file.copy(kumazemi %>% pull(file_name), copy_to)

read_csv(csv_path) %>%
  filter(kumazemi == 1) %>%
  pull(minminzemi)
