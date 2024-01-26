library(tidyverse)
library(lubridate)

set.seed(123)
setwd("/media/okamoto/HDD10TB/cicadasong2023")

df <- tibble(
  fname = list.files(pattern = ".wav", recursive = T, full.names = T)
  ) %>%
  mutate(
    datetime = ymd_hm(str_extract(fname, "\\d{8}_\\d{4}"))
  ) %>%
  mutate(
    month = month(datetime)
  ) %>%
  mutate(site_name = str_extract(fname, "NE\\d{2}")) %>%
  mutate(name = basename(fname))

# 幕内さんデータおこし1回目（2023/10/04）
df1 <- df %>%
  filter(month == 8) %>%
  sample_n(100)

df1 <- df1 %>%
  mutate(
    fname_to = str_c("test_data_1st/", site_name, "_", name)
  )

file.copy(df1$fname, df1$fname_to)

df1 %>%
  write_csv("makuuchi_data_1st.csv")

# 幕内さんデータおこし2回目（????/??/??）
df1 <- read_csv("makuuchi_data_1st.csv")

df %>%
  filter(!(fname %in% df1$fname))
