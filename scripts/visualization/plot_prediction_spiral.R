library(tidyverse)
library(lubridate)
library(hms)
library(slider)

setwd("/home/okamoto/cicada_chorus/")

plot_spiral <- function(csv_path, min_counts, th = 0.8) {
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
    group_by(file_name) %>%
    summarise(across(-c(index), function(v) {sum(v > th)})) %>%
    mutate(
      datetime = file_name %>%
        str_extract("\\d{8}_\\d{4}") %>%
        ymd_hm()
    ) %>%
    select(-file_name) %>%
    pivot_longer(cols = -datetime, names_to = "species", values_to = "is_calling") %>%
    left_join(min_counts, by = "species") %>%
    mutate(is_calling = (is_calling >= min_counts)) %>%
    mutate(
      date = date(datetime),
      hour = hour(datetime),
      time = as_hms(datetime)
    )
  
  species <- df %>%
    pull(species) %>%
    unique()
  
   
  p <- df %>%
    group_by(species, hour, date) %>%
    summarise(freq = mean(as.integer(is_calling))) %>%
    ungroup() %>%
    ggplot(aes(x = hour, y = date, fill = species, alpha = freq)) +
    geom_tile() +
    scale_x_continuous(
      breaks = 0:7 * 3, minor_breaks = 0:23,
      labels = str_c(0:7 * 3, "時")
    ) +
    scale_y_date(
      date_breaks = "2 weeks"
    ) +
    labs(
      x = "",
      y = "",
      alpha = "鳴き声の頻度",
      fill = "セミの種類"
    ) +
    theme_minimal() +
    theme(
      axis.text = element_text(size = 18),
      legend.text = element_text(size = 18),
      legend.title = element_text(size = 22),
      strip.text = element_text(size = 22)
    ) +
    coord_polar(start = -360 / 48 * pi / 180) + # 30分左に回すと真上が0時になる
    facet_wrap(~species)
  
  out_d <- dirname(csv_path)
  ggsave(str_c(out_d, "/", site_name, "_spiral.png"), p, width = 20, height = 12)
  p
}

csv_path <- "predictions/resnet50_spectrogram_20230824230657/NE01_biohazard.csv"
min_counts <- tibble(
  species = c("アブラゼミ", "ツクツクボウシ", "ニイニイゼミ", "ヒグラシ", "ミンミンゼミ"),
  min_counts = c(80, 3, 80, 1, 5)
)
th <- 0.6

plot_spiral(csv_path, min_counts, th)
